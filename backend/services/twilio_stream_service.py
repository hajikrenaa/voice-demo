"""
Twilio Bidirectional Media Stream — OpenAI Realtime API

Ultra-low latency voice agent.

Mode 1 (ElevenLabs OFF — default):
  Twilio mulaw 8kHz ──► g711_ulaw ──► OpenAI Realtime (speech-to-speech)
  Twilio mulaw 8kHz ◄── g711_ulaw ◄──┘
  Zero audio conversion. ~300ms latency.

Mode 2 (ElevenLabs ON):
  Twilio mulaw 8kHz ──► g711_ulaw ──► OpenAI Realtime (text-only response)
                                            │ text (sentence-streamed)
  Twilio mulaw 8kHz ◄── PCM→mulaw  ◄── TTS streaming (ElevenLabs or OpenAI)
"""

import asyncio
import audioop
import base64
import io
import json
import logging
import time
from typing import Optional

import websockets
from openai import AsyncOpenAI
from pydub import AudioSegment

from config import Config
from services.elevenlabs_tts_service import ElevenLabsTTSService

logger = logging.getLogger(__name__)


def mp3_to_mulaw(mp3_bytes: bytes) -> bytes:
    """Convert mp3 audio to raw mulaw 8kHz for Twilio."""
    audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    audio = audio.set_channels(1).set_frame_rate(8000).set_sample_width(2)
    return audioop.lin2ulaw(audio.raw_data, 2)


class TwilioRealtimeHandler:
    """
    Bridges Twilio ↔ OpenAI Realtime API.

    When use_elevenlabs=False:
      - g711_ulaw in/out — zero conversion, ~300ms latency.
    When use_elevenlabs=True:
      - g711_ulaw input, text output from OpenAI
      - Sentence-by-sentence → TTS streaming → mulaw → Twilio
      - Uses ordered queue so sentences never overlap or reorder.
    """

    OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"

    def __init__(self, use_elevenlabs: bool = False, active_script: dict = None):
        self.stream_sid: Optional[str] = None
        self.call_sid: Optional[str] = None
        self.openai_ws = None
        self._connected = False
        self._twilio_ws = None
        self.use_elevenlabs = use_elevenlabs
        self._elevenlabs_tts = ElevenLabsTTSService() if use_elevenlabs else None
        self._elevenlabs_available = True
        self._openai_tts_client = None
        self._response_text_buffer = ""
        self._tts_queue: asyncio.Queue = asyncio.Queue(maxsize=20)
        self._tts_worker_task = None
        self._receive_task: Optional[asyncio.Task] = None
        # Script passed directly from main.py at call time
        self._script: Optional[dict] = active_script
        self._goodbye_detected = False
        self._hangup_task: Optional[asyncio.Task] = None
        # Interruption tracking
        self._current_ai_transcript = ""
        self._ai_is_responding = False  # True while OpenAI is generating audio
        self._tts_playing = False  # True while TTS audio is being sent to Twilio
        # Echo cooldown: keeps the echo gate active AFTER AI finishes generating,
        # because Twilio is still playing audio and the caller's phone echoes it back.
        # Without this, echoed audio triggers ghost responses.
        self._echo_gate_until = 0.0  # timestamp until which echo gate stays active
        self._ECHO_COOLDOWN = 1.0  # seconds after last AI audio chunk (echo round-trip ~0.3-0.6s)

    # ── Twilio message handling ─────────────────────────────

    async def handle_twilio_message(self, twilio_ws, message: dict):
        """Process a message from Twilio's media stream."""
        event = message.get("event")

        if event == "connected":
            logger.info("Twilio media stream connected")

        elif event == "start":
            start_data = message.get("start", {})
            self.stream_sid = start_data.get("streamSid")
            self.call_sid = start_data.get("callSid")
            self._twilio_ws = twilio_ws

            custom_params = start_data.get("customParameters", {})
            elevenlabs_param = custom_params.get("elevenlabs", "false")
            if elevenlabs_param.lower() == "true":
                self.use_elevenlabs = True
                self._elevenlabs_tts = ElevenLabsTTSService()
                self._elevenlabs_available = True
                self._tts_worker_task = asyncio.create_task(self._tts_worker())

            mode = "ElevenLabs" if self.use_elevenlabs else "built-in"
            has_script = "with script" if self._script else "no script"
            logger.info(f"Stream started - SID: {self.stream_sid}, Call: {self.call_sid}, Voice: {mode}, {has_script}")
            await self._connect_openai()

        elif event == "media":
            payload = message.get("media", {}).get("payload", "")
            if payload and self._connected:
                await self._forward_audio(payload)

        elif event == "mark":
            pass

        elif event == "stop":
            logger.info(f"Twilio stream stopped - Call: {self.call_sid}")
            await self._full_cleanup()

    # ── OpenAI Realtime connection ──────────────────────────

    async def _connect_openai(self):
        """Connect to OpenAI Realtime API with a connection timeout."""
        try:
            model = Config.REALTIME_MODEL
            url = f"{self.OPENAI_REALTIME_URL}?model={model}"
            headers = {
                "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1",
            }

            self.openai_ws = await asyncio.wait_for(
                websockets.connect(
                    url,
                    additional_headers=headers,
                    ping_interval=30,
                    ping_timeout=10,
                    max_size=2**20,
                    compression=None,
                ),
                timeout=10.0,
            )
            self._connected = True
            logger.info(f"Connected to OpenAI Realtime API ({model})")

            await self._configure_session()
            self._receive_task = asyncio.create_task(self._receive_openai_events())

            # Trigger the AI to speak first — both inbound and outbound calls.
            await self._trigger_initial_response()

        except asyncio.TimeoutError:
            logger.error("OpenAI Realtime connection timed out (10s)")
            self._connected = False
        except Exception as e:
            logger.error(f"Failed to connect to OpenAI Realtime: {e}")
            self._connected = False

    async def _trigger_initial_response(self):
        """Send response.create so the AI speaks the welcome message immediately."""
        try:
            ws = self.openai_ws
            if ws:
                await ws.send(json.dumps({"type": "response.create"}))
                logger.info("Triggered initial AI response")
        except Exception as e:
            logger.error(f"Failed to trigger initial response: {e}")

    def _build_prompt(self) -> str:
        """Build system prompt from script config or use default."""
        if not self._script:
            return Config.SYSTEM_PROMPT

        s = self._script
        parts = []

        parts.append(
            "RULES:\n"
            "- Be quick and responsive. Keep replies to 1-2 short sentences. Sound snappy and natural.\n"
            "- Callers may have Indian or South Asian accents. Listen carefully to names and details.\n"
            "- If you didn't catch something, ask: \"Sorry, could you say that again?\" or \"Could you spell that for me?\"\n"
            "- NEVER guess or fabricate names, emails, numbers. Only use what the caller confirmed.\n"
            "- NAME CORRECTION: When corrected, FORGET the old name completely. Only use the new one. Repeat it back to confirm. Never reference the old name again.\n"
            "- Always repeat back names/emails for confirmation before moving on.\n"
            "- At recap, use ONLY the LAST confirmed version of each detail."
        )

        parts.append(
            "You are a real sales executive on an outbound call. Sound genuinely human — warm, confident, expressive.\n"
            "React with real emotion: enthusiasm, empathy, curiosity. Use natural filler words: \"Oh!\", \"Ah\", \"Right\", \"Got it\".\n"
            "Max 1-2 sentences per turn. Never sound robotic or scripted.\n"
            "BACK-CHANNEL FROM CALLER — if you hear \"hmm\", \"uh-huh\", \"yeah\", \"okay\" "
            "while you are mid-sentence, the caller is just acknowledging — continue naturally. Do NOT treat acknowledgments as answers."
        )

        if s.get("behaviour"):
            parts.append(s['behaviour'])

        if s.get("welcome"):
            parts.append(
                f'Start with: "{s["welcome"]}"\n'
                f"Deliver the full welcome naturally. If interrupted, handle it, then finish your intro."
            )

        questions = s.get("questions", [])
        if questions:
            q_list = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(questions))
            parts.append(f"Ask these questions naturally, one by one:\n{q_list}")

        if s.get("goal"):
            parts.append(f"Goal: {s['goal']}")

        parts.append(
            "CONVERSATION:\n"
            "- Ask questions one by one. React naturally, then move on.\n"
            "- For names: repeat back and confirm. If unclear, ask to spell using words (A for Alpha, S for Sierra).\n"
            "- For emails: repeat back full email. Confirm domain.\n"
            "- For numbers: repeat back digit by digit.\n"
            "- Never move on until confirmed. If wrong, FORGET the old version and ask fresh.\n"
            "- If caller says just \"hmm\"/\"yeah\"/\"okay\" after your question — they're thinking, not answering. Wait or gently re-ask.\n"
            "- Recap only final confirmed details. End with \"Goodbye\". English only."
        )
        return "\n\n".join(parts)

    async def _configure_session(self):
        """Configure session — g711_ulaw format, server VAD, tuned for phone accuracy."""
        if self.use_elevenlabs:
            modalities = ["text"]
        else:
            modalities = ["text", "audio"]

        prompt = self._build_prompt()
        logger.info(f"System prompt ({len(prompt)} chars): {prompt[:100]}...")

        session_config = {
            "type": "session.update",
            "session": {
                "modalities": modalities,
                "instructions": prompt,
                "voice": Config.REALTIME_VOICE,
                "input_audio_format": "g711_ulaw",
                "output_audio_format": "g711_ulaw",
                "temperature": 0.8,
                "max_response_output_tokens": 100,
                "input_audio_transcription": {
                    "model": "whisper-1",
                    "language": "en",
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.45,
                    "prefix_padding_ms": 200,
                    "silence_duration_ms": 300,
                    "create_response": True,
                },
            },
        }
        await self.openai_ws.send(json.dumps(session_config))
        mode = "ElevenLabs TTS" if self.use_elevenlabs else "built-in voice"
        logger.info(f"Session configured — {mode}, g711_ulaw, VAD(0.45/200/300), temp=0.8, max=100tok")

    # Echo gate: audio energy (RMS) below this threshold is treated as
    # echo/noise and dropped when the AI is speaking or during echo cooldown.
    # Phone echo ≈ 500–1500 RMS, direct caller speech ≈ 2000–15000+ RMS.
    ECHO_GATE_RMS = 1500

    def _is_echo_gate_active(self) -> bool:
        """Check if echo gate should be filtering audio right now."""
        if self._ai_is_responding or self._tts_playing:
            return True
        # Cooldown: Twilio buffers audio, caller phone echoes it back with delay.
        # Keep gate active for a few seconds after AI stops generating.
        if time.monotonic() < self._echo_gate_until:
            return True
        return False

    async def _forward_audio(self, mulaw_b64: str):
        """Forward Twilio mulaw 8kHz directly to OpenAI (g711_ulaw — zero conversion).

        Echo gate is active while AI audio is playing AND during cooldown after.
        This prevents the caller's phone echo from triggering ghost responses.
        """
        try:
            if self._is_echo_gate_active():
                mulaw_bytes = base64.b64decode(mulaw_b64)
                pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
                rms = audioop.rms(pcm_8k, 2)
                if rms < self.ECHO_GATE_RMS:
                    return

            ws = self.openai_ws
            if not ws:
                return
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": mulaw_b64,
            }))
        except Exception as e:
            logger.error(f"Error forwarding audio: {e}")

    # ── OpenAI event processing ─────────────────────────────

    async def _receive_openai_events(self):
        """Receive events from OpenAI and forward to Twilio."""
        if not self.openai_ws:
            return
        try:
            async for raw in self.openai_ws:
                try:
                    event = json.loads(raw)
                    await self._handle_openai_event(event)
                except json.JSONDecodeError:
                    pass
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"OpenAI connection closed: {e}")
        except asyncio.CancelledError:
            logger.info("OpenAI receive task cancelled")
        except Exception as e:
            logger.error(f"Error receiving OpenAI events: {e}")
        finally:
            self._connected = False

    async def _handle_openai_event(self, event: dict):
        """Process an event from OpenAI Realtime API."""
        t = event.get("type", "")

        if t == "session.created":
            logger.info("OpenAI session created")

        elif t == "session.updated":
            logger.info("OpenAI session updated")

        elif t == "response.created":
            self._ai_is_responding = True

        elif t == "response.audio.delta":
            audio_b64 = event.get("delta", "")
            if audio_b64 and self._twilio_ws and self.stream_sid:
                try:
                    await self._twilio_ws.send_json({
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {"payload": audio_b64},
                    })
                    # Refresh echo cooldown — Twilio buffers and plays this audio,
                    # and the caller's phone will echo it back ~1-3s later.
                    self._echo_gate_until = time.monotonic() + self._ECHO_COOLDOWN
                except Exception as e:
                    logger.error(f"Error sending audio to Twilio: {e}")

        elif t == "response.audio_transcript.delta":
            self._current_ai_transcript += event.get("delta", "")

        elif t == "response.audio_transcript.done":
            transcript = event.get("transcript", "")
            if transcript:
                logger.info(f"AI said: {transcript[:80]}...")
                if self._contains_goodbye(transcript):
                    self._schedule_hangup(12.0)
            self._current_ai_transcript = ""

        elif t == "response.text.delta":
            if self.use_elevenlabs:
                self._response_text_buffer += event.get("delta", "")
                self._flush_sentences()
            self._current_ai_transcript += event.get("delta", "")

        elif t == "response.text.done":
            full_text = event.get("text", "")
            if self.use_elevenlabs and self._response_text_buffer.strip():
                text = self._response_text_buffer.strip()
                self._response_text_buffer = ""
                logger.info(f"[flush-final] {text[:60]}")
                await self._enqueue_tts(text)
            if full_text and self._contains_goodbye(full_text):
                self._schedule_hangup(12.0)
            self._current_ai_transcript = ""

        elif t == "response.done":
            status = event.get("response", {}).get("status", "")
            self._ai_is_responding = False
            if status == "cancelled":
                # User interrupted — clear text buffer and drain TTS queue
                self._response_text_buffer = ""
                self._current_ai_transcript = ""
                self._drain_tts_queue()
                logger.info("Response cancelled (user interrupted)")
            else:
                self._current_ai_transcript = ""
                logger.debug(f"Response complete (status={status})")

        elif t == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            if transcript:
                logger.info(f"User said: \"{transcript}\"")
            else:
                logger.warning("User audio transcription was EMPTY — likely garbled or too quiet")

        elif t == "conversation.item.input_audio_transcription.failed":
            error = event.get("error", {})
            logger.error(f"Transcription FAILED: {error.get('message', 'unknown')}")

        elif t == "input_audio_buffer.speech_started":
            self._cancel_hangup()
            if self._ai_is_responding or self._tts_playing or self._is_echo_gate_active():
                logger.info("User interrupted — clearing audio immediately")
                self._drain_tts_queue()
                self._response_text_buffer = ""
                # Clear Twilio audio buffer IMMEDIATELY — no delay
                asyncio.create_task(self._gentle_clear(0.0))

        elif t == "error":
            error = event.get("error", {})
            logger.error(f"OpenAI error: {error.get('message', 'Unknown')} | {error}")

    def _flush_sentences(self):
        """Extract ALL complete sentences from the buffer and queue them for TTS."""
        changed = True
        while changed:
            changed = False
            for sep in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                if sep in self._response_text_buffer:
                    parts = self._response_text_buffer.split(sep, 1)
                    sentence = parts[0] + sep.strip()
                    self._response_text_buffer = parts[1] if len(parts) > 1 else ""
                    if sentence.strip():
                        logger.info(f"[flush-sentence] {sentence.strip()[:60]}")
                        asyncio.get_event_loop().call_soon(
                            lambda s=sentence.strip(): self._tts_queue.put_nowait(s)
                            if not self._tts_queue.full() else None
                        )
                    changed = True
                    break

    async def _enqueue_tts(self, text: str):
        """Enqueue text for TTS with backpressure (bounded queue)."""
        try:
            self._tts_queue.put_nowait(text)
        except asyncio.QueueFull:
            logger.warning(f"TTS queue full, dropping: {text[:40]}")

    def _drain_tts_queue(self):
        """Drain all pending TTS items on interruption."""
        drained = 0
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break
        if drained:
            logger.info(f"Drained {drained} items from TTS queue (user interrupted)")

    # ── TTS worker (ordered queue) ──────────────────────────

    async def _tts_worker(self):
        """Process TTS queue sequentially — sentences play in order."""
        logger.info("TTS worker started")
        while True:
            try:
                text = await self._tts_queue.get()
                if text is None:
                    break
                self._tts_playing = True
                await self._synthesize_tts(text)
                self._tts_playing = False
            except asyncio.CancelledError:
                logger.info("TTS worker cancelled")
                break
            except Exception as e:
                self._tts_playing = False
                logger.error(f"TTS worker error: {e}", exc_info=True)
        self._tts_playing = False
        logger.info("TTS worker stopped")

    async def _synthesize_tts(self, text: str):
        """Try ElevenLabs first, fall back to streaming OpenAI TTS."""
        if not self._twilio_ws or not self.stream_sid:
            return

        if self._elevenlabs_tts and self._elevenlabs_available:
            try:
                mp3_audio = await self._elevenlabs_tts.synthesize(text)
                logger.info(f"ElevenLabs OK ({len(mp3_audio)} bytes): {text[:50]}")
                mulaw = mp3_to_mulaw(mp3_audio)
                await self._send_mulaw_to_twilio(mulaw)
                return
            except Exception as e:
                logger.warning(f"ElevenLabs failed — switching to OpenAI TTS: {e}")
                self._elevenlabs_available = False

        await self._stream_openai_tts(text)

    # ── OpenAI streaming TTS ────────────────────────────────

    async def _stream_openai_tts(self, text: str):
        """Stream OpenAI TTS → PCM 24kHz → resample 8kHz → mulaw → Twilio."""
        try:
            if not self._openai_tts_client:
                self._openai_tts_client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)

            async with self._openai_tts_client.audio.speech.with_streaming_response.create(
                model=Config.TTS_MODEL,
                voice=Config.REALTIME_VOICE,
                input=text,
                response_format="pcm",
            ) as response:
                ratecv_state = None
                pcm_buffer = b""
                chunks_sent = 0
                PCM_CHUNK = 4800  # 100ms at 24kHz 16-bit mono

                async for data in response.iter_bytes(chunk_size=PCM_CHUNK):
                    pcm_buffer += data

                    while len(pcm_buffer) >= PCM_CHUNK:
                        pcm_chunk = pcm_buffer[:PCM_CHUNK]
                        pcm_buffer = pcm_buffer[PCM_CHUNK:]

                        resampled, ratecv_state = audioop.ratecv(
                            pcm_chunk, 2, 1, 24000, 8000, ratecv_state
                        )
                        mulaw = audioop.lin2ulaw(resampled, 2)
                        mulaw_b64 = base64.b64encode(mulaw).decode("utf-8")
                        if self._twilio_ws and self.stream_sid:
                            await self._twilio_ws.send_json({
                                "event": "media",
                                "streamSid": self.stream_sid,
                                "media": {"payload": mulaw_b64},
                            })
                            self._echo_gate_until = time.monotonic() + self._ECHO_COOLDOWN
                            chunks_sent += 1

                # Flush remaining
                if len(pcm_buffer) >= 2:
                    usable = len(pcm_buffer) - (len(pcm_buffer) % 2)
                    if usable > 0:
                        resampled, ratecv_state = audioop.ratecv(
                            pcm_buffer[:usable], 2, 1, 24000, 8000, ratecv_state
                        )
                        mulaw = audioop.lin2ulaw(resampled, 2)
                        mulaw_b64 = base64.b64encode(mulaw).decode("utf-8")
                        if self._twilio_ws and self.stream_sid:
                            await self._twilio_ws.send_json({
                                "event": "media",
                                "streamSid": self.stream_sid,
                                "media": {"payload": mulaw_b64},
                            })
                            self._echo_gate_until = time.monotonic() + self._ECHO_COOLDOWN
                            chunks_sent += 1

            logger.info(f"Streamed TTS ({chunks_sent} chunks): {text[:50]}")

        except Exception as e:
            logger.error(f"OpenAI TTS streaming failed: {e}", exc_info=True)

    # ── ElevenLabs mulaw delivery ───────────────────────────

    async def _send_mulaw_to_twilio(self, mulaw_audio: bytes):
        """Send mulaw audio to Twilio in ~400ms chunks."""
        if not self._twilio_ws or not self.stream_sid:
            return
        CHUNK_SIZE = 3200  # 400ms at 8kHz mulaw
        for i in range(0, len(mulaw_audio), CHUNK_SIZE):
            chunk = mulaw_audio[i:i + CHUNK_SIZE]
            chunk_b64 = base64.b64encode(chunk).decode("utf-8")
            try:
                await self._twilio_ws.send_json({
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": chunk_b64},
                })
                self._echo_gate_until = time.monotonic() + self._ECHO_COOLDOWN
            except Exception as e:
                logger.error(f"Error sending chunk to Twilio: {e}")
                break

    # ── Gentle audio stop ─────────────────────────────────

    async def _gentle_clear(self, delay: float):
        """Wait a moment for current word to finish, then clear Twilio audio."""
        try:
            await asyncio.sleep(delay)
            if self._twilio_ws and self.stream_sid:
                await self._twilio_ws.send_json({
                    "event": "clear",
                    "streamSid": self.stream_sid,
                })
                logger.debug("Twilio audio cleared gently")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Gentle clear failed (likely disconnected): {e}")

    # ── Auto-hangup detection ──────────────────────────────

    def _contains_goodbye(self, text: str) -> bool:
        """Check if the AI's response is a clear farewell."""
        lower = text.lower().strip().rstrip("!.,;")
        goodbye_phrases = ["goodbye", "good bye", "bye bye", "bye-bye",
                           "have a great day goodbye", "take care goodbye",
                           "talk to you later"]
        for phrase in goodbye_phrases:
            if lower.endswith(phrase):
                logger.info(f"Goodbye detected in AI response: '{phrase}'")
                return True
        return False

    def _schedule_hangup(self, delay: float):
        """Schedule a hangup — cancellable if user speaks again."""
        self._cancel_hangup()
        self._goodbye_detected = True
        self._hangup_task = asyncio.create_task(self._hangup_after_delay(delay))

    def _cancel_hangup(self):
        """Cancel pending hangup because user is still talking."""
        if self._hangup_task and not self._hangup_task.done():
            self._hangup_task.cancel()
            self._goodbye_detected = False
            logger.info("Hangup cancelled — user is still speaking")

    async def _hangup_after_delay(self, delay: float):
        """Wait for AI audio to finish, then hang up. Uses thread to avoid blocking event loop."""
        try:
            logger.info(f"Auto-hangup scheduled in {delay}s (cancels if user speaks)")
            await asyncio.sleep(delay)

            if self.call_sid:
                # Run synchronous Twilio client in a thread to avoid blocking the event loop
                await asyncio.to_thread(self._sync_hangup_call, self.call_sid)
                logger.info(f"Call {self.call_sid} hung up automatically after goodbye")
            else:
                logger.warning("No call_sid available for auto-hangup")
        except asyncio.CancelledError:
            logger.info("Hangup was cancelled — conversation continues")
        except Exception as e:
            logger.error(f"Auto-hangup failed: {e}")

    @staticmethod
    def _sync_hangup_call(call_sid: str):
        """Synchronous Twilio hangup — runs in a thread."""
        from twilio.rest import Client as TwilioClient
        client = TwilioClient(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)
        client.calls(call_sid).update(status="completed")

    # ── Cleanup ─────────────────────────────────────────────

    async def _full_cleanup(self):
        """Clean up all resources: TTS worker, hangup task, receive task, OpenAI WS, TTS client."""
        # Cancel hangup timer
        self._cancel_hangup()

        # Stop TTS worker
        if self._tts_worker_task and not self._tts_worker_task.done():
            self._drain_tts_queue()
            await self._tts_queue.put(None)  # sentinel
            try:
                await asyncio.wait_for(self._tts_worker_task, timeout=3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._tts_worker_task.cancel()

        # Cancel OpenAI receive task
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except (asyncio.CancelledError, Exception):
                pass

        # Close OpenAI WebSocket
        await self._disconnect_openai()

        # Close OpenAI TTS client
        if self._openai_tts_client:
            try:
                await self._openai_tts_client.close()
            except Exception:
                pass
            self._openai_tts_client = None

    async def _disconnect_openai(self):
        """Close the OpenAI WebSocket."""
        if self.openai_ws:
            try:
                await self.openai_ws.close()
            except Exception:
                pass
            self.openai_ws = None
            self._connected = False
            logger.info("Disconnected from OpenAI Realtime API")


# Backward compat alias
TwilioStreamHandler = TwilioRealtimeHandler
