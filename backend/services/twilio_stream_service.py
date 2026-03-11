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
        self._was_interrupted = False  # Flag: response was cut short

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
                ping_interval=20,
                ping_timeout=10,
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

        # Natural sales executive personality
        parts.append(
            "You are a real sales executive making an outbound call. You sound exactly like a human — "
            "warm, confident, conversational. Never admit you're AI. "
            "If asked, laugh: \"Haha no, I'm very much real!\"\n"
            "Talk like a real person on a call — short, natural, flowing. "
            "Use reactions: \"Oh nice\", \"Okay great\", \"Got it\", \"Sure sure\", \"Ah okay\", \"Perfect\". "
            "Max 1-2 sentences per turn. Never sound scripted or robotic."
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
            "HOW TO HAVE THE CONVERSATION:\n"
            "- Talk like a real sales person, not a form-filler. Weave questions into natural conversation.\n"
            "- After they answer, react naturally first (\"Oh that's great\", \"Okay nice\") then move to next topic.\n"
            "- If they say something that isn't an answer (\"hold on\", \"what?\", a question back), handle it and ask again naturally.\n"
            "- Don't just fire questions — make it a conversation. Connect their answers to the next question.\n\n"
            "GETTING NAMES & EMAILS RIGHT:\n"
            "- When they say a name, repeat it back naturally: \"Prajit, right? Just to make sure I have the spelling right, "
            "could you spell that out for me? Like P for Peter, R for...?\"\n"
            "- If they spell raw letters and it's unclear, say naturally: \"Sorry, this connection — "
            "could you give me a word for each letter? Like P for Peter?\"\n"
            "- For email: \"And what's a good email? ... Got it, let me just confirm — that's prajit at gmail dot com? "
            "Could you spell the part before @ for me? Just to be safe.\"\n"
            "- For numbers: repeat back naturally: \"So that's 98110-01639, right?\"\n"
            "- If they say it's wrong, don't panic: \"Oh sorry about that, could you say that one more time for me?\"\n"
            "- After 2 wrong attempts, go slow naturally: \"Tell you what, let's take it one letter at a time so I get it perfect. First letter?\"\n"
            "- NEVER accept something they haven't confirmed. But confirm naturally, not robotically.\n\n"
            "WHEN INTERRUPTED:\n"
            "- Stop, listen, respond to them.\n"
            "- Then pick up where you were: \"Anyway, so...\" or \"Coming back to what I was asking...\"\n"
            "- Never skip or forget where you were.\n\n"
            "WRAPPING UP:\n"
            "- After all info collected, quickly recap: \"So just to make sure I have everything — [details]. That all sound right?\"\n"
            "- Fix anything wrong.\n"
            "- \"Is there anything else you'd like to know?\" — if yes, help them. If no:\n"
            "- \"Perfect, thanks so much for your time. Really appreciate it. Have a great day! Goodbye.\"\n"
            "- If no response: try once more. Still nothing: \"Seems like the line dropped. No worries, take care! Goodbye.\"\n"
            "- Always end with \"Goodbye\". Never make up facts. English only."
        )
        return "\n\n".join(parts)

    async def _configure_session(self):
        """Configure session — g711_ulaw format, server VAD, low latency."""
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
                "input_audio_transcription": {
                    "model": "whisper-1",
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 200,
                    "silence_duration_ms": 500,
                    "create_response": True,
                },
            },
        }
        await self.openai_ws.send(json.dumps(session_config))
        mode = "ElevenLabs TTS" if self.use_elevenlabs else "built-in voice"
        logger.info(f"Session configured — {mode}, g711_ulaw, server VAD (200/300ms)")

    async def _forward_audio(self, mulaw_b64: str):
        """Forward Twilio mulaw audio directly to OpenAI."""
        try:
            await self.openai_ws.send(json.dumps({
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
            # Track what the AI is currently saying (built-in voice)
            self._current_ai_transcript += event.get("delta", "")

        elif t == "response.audio_transcript.done":
            transcript = event.get("transcript", "")
            if transcript:
                logger.info(f"AI said: {transcript[:80]}...")
                if self._contains_goodbye(transcript):
                    self._schedule_hangup(5.0)
            # Response finished naturally — clear tracker
            self._current_ai_transcript = ""

        elif t == "response.text.delta":
            # ElevenLabs mode: accumulate text, flush on sentence boundary
            if self.use_elevenlabs:
                self._response_text_buffer += event.get("delta", "")
                self._flush_sentence()
            # Also track for interruption
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
            if status == "cancelled" and self._was_interrupted:
                # Response was interrupted — inject context so AI knows to resume
                interrupted_text = self._current_ai_transcript.strip()
                if interrupted_text:
                    logger.info(f"Interrupted while saying: {interrupted_text[:80]}...")
                    await self._inject_resume_context(interrupted_text)
                self._was_interrupted = False
            self._current_ai_transcript = ""
            logger.debug(f"Response complete (status={status})")

        elif t == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            if transcript:
                logger.info(f"User said: {transcript[:80]}...")

        elif t == "input_audio_buffer.speech_started":
            logger.debug("User speaking — AI will fade out gently")
            self._cancel_hangup()
            self._was_interrupted = True
            # Gentle stop: let current word/phrase finish (~300ms) before clearing
            asyncio.create_task(self._gentle_clear(0.3))

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

    # ── Interruption resume ───────────────────────────────

    async def _inject_resume_context(self, interrupted_text: str):
        """Inject a system message so the AI knows what it was saying before interruption."""
        try:
            await self.openai_ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "system",
                    "content": [{
                        "type": "input_text",
                        "text": (
                            f"You were interrupted while saying: \"{interrupted_text}\". "
                            f"First, address what the caller just said. "
                            f"Then say \"Anyway, as I was saying...\" and CONTINUE from where you stopped. "
                            f"Do NOT repeat what you already said. Do NOT skip anything."
                        )
                    }]
                }
            }))
            logger.info("Injected resume context after interruption")
        except Exception as e:
            logger.error(f"Failed to inject resume context: {e}")

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
