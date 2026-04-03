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
        self._tts_queue: asyncio.Queue = asyncio.Queue()
        self._tts_worker_task = None
        # Script passed directly from main.py at call time
        self._script: Optional[dict] = active_script
        self._goodbye_detected = False
        self._hangup_task: Optional[asyncio.Task] = None
        # Interruption tracking
        self._current_ai_transcript = ""  # What AI is currently saying
        self._ai_is_responding = False  # True while AI is generating a response
        # Audio resampling state (mulaw 8kHz → pcm16 24kHz)
        self._ratecv_state = None

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
                # Test ElevenLabs before committing to it
                ok = await self._test_elevenlabs()
                if ok:
                    self.use_elevenlabs = True
                    self._elevenlabs_tts = ElevenLabsTTSService()
                    self._elevenlabs_available = True
                    self._tts_worker_task = asyncio.create_task(self._tts_worker())
                else:
                    logger.warning("ElevenLabs unavailable — falling back to built-in voice")
                    self.use_elevenlabs = False

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
            # Stop TTS worker
            if self._tts_worker_task:
                await self._tts_queue.put(None)
            await self._disconnect_openai()

    # ── ElevenLabs health check ────────────────────────────

    async def _test_elevenlabs(self) -> bool:
        """Quick test if ElevenLabs API works. Returns False if no credits or bad key."""
        try:
            tts = ElevenLabsTTSService()
            audio = await tts.synthesize("test")
            return len(audio) > 0
        except Exception as e:
            logger.warning(f"ElevenLabs test failed: {e}")
            return False

    # ── OpenAI Realtime connection ──────────────────────────

    async def _connect_openai(self):
        """Connect to OpenAI Realtime API."""
        try:
            model = Config.REALTIME_MODEL
            url = f"{self.OPENAI_REALTIME_URL}?model={model}"
            headers = {
                "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1",
            }

            self.openai_ws = await websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=30,
                ping_timeout=10,
                max_size=2**20,
                compression=None,
            )
            self._connected = True
            logger.info(f"Connected to OpenAI Realtime API ({model})")

            await self._configure_session()
            asyncio.create_task(self._receive_openai_events())

            # Trigger the AI to speak first (outbound call — we initiate)
            if self._script:
                await self._trigger_initial_response()

        except Exception as e:
            logger.error(f"Failed to connect to OpenAI Realtime: {e}")
            self._connected = False

    async def _trigger_initial_response(self):
        """Send response.create so the AI speaks the welcome message immediately."""
        try:
            await self.openai_ws.send(json.dumps({
                "type": "response.create",
            }))
            logger.info("Triggered initial AI response (outbound call)")
        except Exception as e:
            logger.error(f"Failed to trigger initial response: {e}")

    def _build_prompt(self) -> str:
        """Build system prompt from script config or use default."""
        if not self._script:
            return Config.SYSTEM_PROMPT

        s = self._script
        parts = []

        # ── CORE RULES (compact — saves tokens every turn) ──
        parts.append(
            "RULES:\n"
            "- NEVER guess or fabricate names, emails, numbers. Only use what caller confirmed.\n"
            "- If unsure, ask again: \"Sorry, could you repeat that?\"\n"
            "- Always repeat back names/emails for confirmation. If wrong, ask fresh.\n"
            "- When corrected, discard old version permanently. Use only the new one.\n"
            "- At recap, use only the LAST confirmed version of each detail.\n"
            "- Keep responses to 1-2 short sentences. Be concise."
        )

        # ── Personality ──
        parts.append(
            "You are a real sales executive on an outbound call. Sound human — warm, confident, natural.\n"
            "Max 1-2 sentences per turn. Never sound scripted.\n"
            "ACTIVE LISTENING — while the caller is speaking, use back-channel sounds naturally:\n"
            "  \"Mm-hmm\", \"Right\", \"Uh-huh\", \"Sure\", \"Okay\", \"Got it\", \"Yeah\"\n"
            "After they finish, react before moving on: \"Oh nice\", \"Great\", \"Okay perfect\".\n"
            "BACK-CHANNEL FROM CALLER — if you hear \"hmm\", \"uh-huh\", \"yeah\", \"okay\" "
            "while you are mid-sentence, the caller is just acknowledging — they are NOT answering a question. "
            "Continue what you were saying naturally. Do NOT treat acknowledgments as answers."
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
            "- Ask questions one by one. React naturally before moving on.\n"
            "- For names: repeat back and confirm. If unclear, ask to spell with words (H for Hotel).\n"
            "- For emails: confirm domain (gmail/yahoo/etc), repeat full email back.\n"
            "- For numbers: repeat back digit by digit.\n"
            "- Never move on until caller confirms. If wrong, ask again fresh.\n"
            "- If caller says just \"hmm\"/\"yeah\"/\"okay\" after your question, that's NOT an answer — "
            "they're thinking. Wait, or gently prompt: \"Take your time\" or repeat the question.\n"
            "- When corrected, discard old version, use only the new one.\n"
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
                "input_audio_format": "pcm16",
                "output_audio_format": "g711_ulaw",
                "temperature": 0.4,
                "max_response_output_tokens": 80,
                "input_audio_transcription": {
                    "model": "whisper-1",
                    "language": "en",
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.45,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 700,
                    "create_response": True,
                },
            },
        }
        await self.openai_ws.send(json.dumps(session_config))
        mode = "ElevenLabs TTS" if self.use_elevenlabs else "built-in voice"
        logger.info(f"Session configured — {mode}, pcm16/g711_ulaw, VAD(0.45/300/700), temp=0.4, max=80tok")

    # Echo gate: audio energy (RMS) below this threshold is treated as
    # echo/noise and dropped when the AI is speaking.
    # Phone echo ≈ 500–2000 RMS, direct caller speech ≈ 3000–15000+ RMS.
    ECHO_GATE_RMS = 2500

    async def _forward_audio(self, mulaw_b64: str):
        """Convert Twilio mulaw 8kHz → pcm16 24kHz and forward to OpenAI.

        When the AI is speaking, applies an echo gate: incoming audio whose
        energy is below ECHO_GATE_RMS is dropped (it's just the AI's voice
        leaking through the caller's microphone or background noise).
        Only loud-enough audio (actual caller speech) passes through.
        """
        try:
            mulaw_bytes = base64.b64decode(mulaw_b64)
            # mulaw → linear PCM 16-bit
            pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)

            # ── Echo gate: active only while AI is speaking ──
            if self._ai_is_responding:
                rms = audioop.rms(pcm_8k, 2)
                if rms < self.ECHO_GATE_RMS:
                    return  # drop — echo or background noise

            # Upsample 8kHz → 24kHz (stateful for seamless chunk joins)
            pcm_24k, self._ratecv_state = audioop.ratecv(
                pcm_8k, 2, 1, 8000, 24000, self._ratecv_state
            )
            pcm_b64 = base64.b64encode(pcm_24k).decode("utf-8")

            await self.openai_ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": pcm_b64,
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
            # Built-in voice: stream g711_ulaw straight to Twilio
            audio_b64 = event.get("delta", "")
            if audio_b64 and self._twilio_ws and self.stream_sid:
                try:
                    await self._twilio_ws.send_json({
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {"payload": audio_b64},
                    })
                except Exception as e:
                    logger.error(f"Error sending audio to Twilio: {e}")

        elif t == "response.audio_transcript.delta":
            self._current_ai_transcript += event.get("delta", "")

        elif t == "response.audio_transcript.done":
            transcript = event.get("transcript", "")
            if transcript:
                logger.info(f"AI said: {transcript[:80]}...")
                if self._contains_goodbye(transcript):
                    self._schedule_hangup(5.0)
            self._current_ai_transcript = ""

        elif t == "response.text.delta":
            if self.use_elevenlabs:
                self._response_text_buffer += event.get("delta", "")
                self._flush_sentence()
            self._current_ai_transcript += event.get("delta", "")

        elif t == "response.text.done":
            full_text = event.get("text", "")
            if self.use_elevenlabs and self._response_text_buffer.strip():
                text = self._response_text_buffer.strip()
                self._response_text_buffer = ""
                logger.info(f"[flush-final] {text[:60]}")
                self._tts_queue.put_nowait(text)
            if full_text and self._contains_goodbye(full_text):
                self._schedule_hangup(6.0)
            self._current_ai_transcript = ""

        elif t == "response.done":
            status = event.get("response", {}).get("status", "")
            self._ai_is_responding = False
            self._current_ai_transcript = ""
            if status == "cancelled":
                logger.info("Response cancelled (user spoke during AI speech)")
            else:
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
            if self._ai_is_responding:
                # User spoke while AI is speaking — clear buffered audio
                # after a short delay so current word finishes naturally
                logger.info("User spoke during AI response — clearing audio")
                asyncio.create_task(self._gentle_clear(0.15))

        elif t == "error":
            error = event.get("error", {})
            logger.error(f"OpenAI error: {error.get('message', 'Unknown')} | {error}")

    def _flush_sentence(self):
        """Extract complete sentences from the buffer and queue them for TTS."""
        for sep in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
            if sep in self._response_text_buffer:
                parts = self._response_text_buffer.split(sep, 1)
                sentence = parts[0] + sep.strip()
                self._response_text_buffer = parts[1] if len(parts) > 1 else ""
                if sentence.strip():
                    logger.info(f"[flush-sentence] {sentence.strip()[:60]}")
                    self._tts_queue.put_nowait(sentence.strip())
                break

    # ── TTS worker (ordered queue) ──────────────────────────

    async def _tts_worker(self):
        """Process TTS queue sequentially — sentences play in order."""
        logger.info("TTS worker started")
        while True:
            try:
                text = await self._tts_queue.get()
                if text is None:
                    break
                await self._synthesize_tts(text)
            except Exception as e:
                logger.error(f"TTS worker error: {e}", exc_info=True)
        logger.info("TTS worker stopped")

    async def _synthesize_tts(self, text: str):
        """Try ElevenLabs first, fall back to streaming OpenAI TTS."""
        if not self._twilio_ws or not self.stream_sid:
            return

        # --- ElevenLabs ---
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

        # --- Fallback: streaming OpenAI TTS (PCM → mulaw) ---
        await self._stream_openai_tts(text)

    # ── OpenAI streaming TTS ────────────────────────────────

    async def _stream_openai_tts(self, text: str):
        """
        Stream OpenAI TTS → PCM 24kHz → resample 8kHz → mulaw → Twilio.
        Streaming means audio starts playing while still generating.
        """
        try:
            if not self._openai_tts_client:
                self._openai_tts_client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)

            async with self._openai_tts_client.audio.speech.with_streaming_response.create(
                model=Config.TTS_MODEL,
                voice=Config.REALTIME_VOICE,
                input=text,
                response_format="pcm",  # 24kHz 16-bit mono
            ) as response:
                ratecv_state = None
                pcm_buffer = b""
                chunks_sent = 0
                # 4800 bytes = 100ms at 24kHz 16-bit mono
                PCM_CHUNK = 4800

                async for data in response.iter_bytes(chunk_size=PCM_CHUNK):
                    pcm_buffer += data

                    while len(pcm_buffer) >= PCM_CHUNK:
                        pcm_chunk = pcm_buffer[:PCM_CHUNK]
                        pcm_buffer = pcm_buffer[PCM_CHUNK:]

                        # Resample 24kHz → 8kHz (stateful for seamless joins)
                        resampled, ratecv_state = audioop.ratecv(
                            pcm_chunk, 2, 1, 24000, 8000, ratecv_state
                        )
                        mulaw = audioop.lin2ulaw(resampled, 2)
                        mulaw_b64 = base64.b64encode(mulaw).decode("utf-8")
                        await self._twilio_ws.send_json({
                            "event": "media",
                            "streamSid": self.stream_sid,
                            "media": {"payload": mulaw_b64},
                        })
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
                        await self._twilio_ws.send_json({
                            "event": "media",
                            "streamSid": self.stream_sid,
                            "media": {"payload": mulaw_b64},
                        })
                        chunks_sent += 1

            logger.info(f"Streamed TTS ({chunks_sent} chunks): {text[:50]}")

        except Exception as e:
            logger.error(f"OpenAI TTS streaming failed: {e}", exc_info=True)

    # ── ElevenLabs mulaw delivery ───────────────────────────

    async def _send_mulaw_to_twilio(self, mulaw_audio: bytes):
        """Send mulaw audio to Twilio in ~400ms chunks."""
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
        except Exception:
            pass

    # ── Auto-hangup detection ──────────────────────────────

    def _contains_goodbye(self, text: str) -> bool:
        """Check if the AI's response ends with a goodbye."""
        lower = text.lower().strip().rstrip("!.,")
        goodbye_words = ["goodbye", "good bye", "bye bye", "bye"]
        for phrase in goodbye_words:
            if lower.endswith(phrase):
                logger.info("Goodbye detected in AI response")
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
        """Wait for AI audio to finish, then hang up. Cancelled if user speaks."""
        try:
            logger.info(f"Auto-hangup scheduled in {delay}s (cancels if user speaks)")
            await asyncio.sleep(delay)

            if self.call_sid:
                from twilio.rest import Client as TwilioClient
                client = TwilioClient(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)
                client.calls(self.call_sid).update(status="completed")
                logger.info(f"Call {self.call_sid} hung up automatically after goodbye")
            else:
                logger.warning("No call_sid available for auto-hangup")
        except asyncio.CancelledError:
            logger.info("Hangup was cancelled — conversation continues")
        except Exception as e:
            logger.error(f"Auto-hangup failed: {e}")

    # ── Cleanup ─────────────────────────────────────────────

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
