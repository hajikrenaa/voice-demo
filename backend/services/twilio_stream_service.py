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
        # Interruption state (manual turn-taking — create_response=false)
        self._last_interrupted_transcript = ""
        self._interrupt_pending = False
        self._speech_start_time = 0.0
        self._first_audio_chunk = True
        self._audio_chunk_count = 0
        self._response_start_time = 0.0
        # Echo cooldown: keeps the echo gate active AFTER AI finishes generating,
        # because Twilio is still playing audio and the caller's phone echoes it back.
        # Without this, echoed audio triggers ghost responses.
        self._echo_gate_until = 0.0  # timestamp until which echo gate stays active
        self._ECHO_COOLDOWN = Config.ECHO_COOLDOWN_S

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

            mode_label = "ElevenLabs TTS" if self.use_elevenlabs else "speech-to-speech"
            has_script = "with script" if self._script else "no script"
            logger.info(f"Stream started - SID: {self.stream_sid}, Call: {self.call_sid}, Mode: {mode_label}, {has_script}")
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
            print(f"[CALL] Connecting to OpenAI Realtime ({model})...", flush=True)

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
            # NOTE: Do NOT set _connected=True yet — audio must not flow until
            # session config is confirmed (g711_ulaw format active).
            # Default is pcm16 which causes screeching on Twilio.
            print(f"[CALL] OpenAI WebSocket connected OK", flush=True)

            confirmed = await self._configure_session()
            if not confirmed:
                print("[CALL] *** SESSION CONFIG FAILED — aborting ***", flush=True)
                await self._close_twilio_on_failure("Session config failed")
                return

            # NOW it's safe to forward audio — format is confirmed g711_ulaw
            self._connected = True
            print("[CALL] _connected=True — audio forwarding enabled", flush=True)
            self._receive_task = asyncio.create_task(self._receive_openai_events())

            # Trigger the AI to speak first — both inbound and outbound calls.
            await self._trigger_initial_response()

        except asyncio.TimeoutError:
            print("[CALL] *** OpenAI connection TIMED OUT (10s) ***", flush=True)
            self._connected = False
            await self._close_twilio_on_failure("OpenAI connection timed out")
        except Exception as e:
            print(f"[CALL] *** OpenAI connection FAILED: {e} ***", flush=True)
            self._connected = False
            await self._close_twilio_on_failure(str(e))

    async def _trigger_initial_response(self):
        """Send response.create after a brief delay to let carrier messages finish."""
        try:
            # Brief pause — just enough for the stream to stabilise
            await asyncio.sleep(0.5)
            ws = self.openai_ws
            if ws:
                await ws.send(json.dumps({"type": "response.create"}))
                logger.info("Triggered initial AI response (after 0.5s delay)")
        except Exception as e:
            logger.error(f"Failed to trigger initial response: {e}")

    def _build_prompt(self) -> str:
        """Build system prompt from script config or use default."""
        if not self._script:
            return Config.SYSTEM_PROMPT

        s = self._script
        parts = []
        n_questions = len(s.get("questions", []))

        # === ROLE + CORE RULES (compact — saves tokens) ===
        parts.append(
            "You are a real person on a phone call. Sound warm, natural, confident. English only.\n"
            "Follow the script below EXACTLY. Ask every question in order. 1-2 sentences per reply max.\n"
            "NEVER skip a question. Ask ALL questions before the recap."
        )

        if s.get("behaviour"):
            parts.append(s['behaviour'])

        # === OPENING ===
        if s.get("welcome"):
            parts.append(f'SAY FIRST: "{s["welcome"]}"')

        # === QUESTIONS ===
        questions = s.get("questions", [])
        if questions:
            q_list = []
            for i, q in enumerate(questions):
                text = q.get("question", q) if isinstance(q, dict) else q
                q_list.append(f"  Q{i+1}: {text}")
            parts.append("QUESTIONS (one at a time, in order):\n" + "\n".join(q_list))

        if s.get("goal"):
            parts.append(f"GOAL: {s['goal']}")

        # === CONVERSATION RULES (all issues from testing) ===
        parts.append(
            "RULES:\n"
            "- After caller answers, react briefly ('gotcha', 'nice') then ask the NEXT question.\n"
            "- Do NOT repeat what the caller just told you. Move forward.\n"
            "- If caller asks 'why?' — answer briefly, then go to the NEXT scripted question.\n"
            "- 'yes'/'yeah'/'correct'/'okay' after you repeat a detail = CONFIRMED. Move on.\n"
            "- Only re-ask if caller says 'no' or gives a correction.\n"
            "- If a name/term sounds unclear, ask them to repeat or spell it.\n"
            "- Names: say the full name naturally ('Got it, Hajik Renaa'). Do NOT spell letter by letter.\n"
            "- Emails: say naturally ('hajik at gmail dot com'). Only spell if unsure.\n"
            "- If corrected, FORGET the old version. Use only the new one.\n"
            f"- RECAP: only after ALL {n_questions} questions are answered. Deliver it as one complete "
            "statement without pausing. Then ask 'Did I get everything right?' and end with Goodbye."
        )

        return "\n\n".join(parts)

    async def _configure_session(self) -> bool:
        """Configure session — g711_ulaw format, VAD, tuned for phone accuracy.

        Returns True if session config was confirmed, False on failure.
        Critical: g711_ulaw MUST be confirmed before any audio flows,
        otherwise OpenAI defaults to pcm16 which causes screeching on Twilio.
        """
        if self.use_elevenlabs:
            modalities = ["text"]
        else:
            modalities = ["text", "audio"]

        prompt = self._build_prompt()
        logger.info(f"System prompt ({len(prompt)} chars): {prompt[:100]}...")

        # Temperature below 0.6 can cause continuous noise/screeching from the model
        temperature = max(0.7, 0.6 if self._script else 0.8)
        max_tokens = 200 if self._script else 100

        # Try preferred VAD first, then fall back to server_vad
        vad_configs = []
        if Config.VAD_TYPE == "semantic_vad":
            vad_configs.append({
                "type": "semantic_vad",
                "eagerness": Config.SEMANTIC_VAD_EAGERNESS,
            })
        # Always include server_vad as fallback
        vad_configs.append({
            "type": "server_vad",
            "threshold": Config.VAD_THRESHOLD,
            "prefix_padding_ms": Config.VAD_PREFIX_PADDING_MS,
            "silence_duration_ms": Config.VAD_SILENCE_DURATION_MS,
        })

        for vad_config in vad_configs:
            vad_type = vad_config["type"]
            session_config = {
                "type": "session.update",
                "session": {
                    "modalities": modalities,
                    "instructions": prompt,
                    "voice": Config.REALTIME_VOICE,
                    "input_audio_format": "g711_ulaw",
                    "output_audio_format": "g711_ulaw",
                    "temperature": temperature,
                    "max_response_output_tokens": max_tokens,
                    "input_audio_transcription": {
                        "model": "whisper-1",
                        "language": "en",
                    },
                    "turn_detection": vad_config,
                    "input_audio_noise_reduction": {
                        "type": "far_field",
                    },
                },
            }
            print(f"[CALL] Sending session config — g711_ulaw, {vad_type}, temp={temperature}, noise_reduction=far_field", flush=True)
            await self.openai_ws.send(json.dumps(session_config))

            confirmed = await self._wait_for_session_confirmation(
                vad_type, temperature, max_tokens
            )
            if confirmed:
                # Clear any audio that may have buffered during setup
                try:
                    await self.openai_ws.send(json.dumps({
                        "type": "input_audio_buffer.clear",
                    }))
                except Exception:
                    pass
                return True

            if vad_type != "server_vad":
                logger.warning(f"{vad_type} failed — retrying with server_vad fallback")

        logger.error("ALL session config attempts failed — audio format is NOT g711_ulaw!")
        return False

    async def _wait_for_session_confirmation(
        self, vad_type: str, temperature: float, max_tokens: int
    ) -> bool:
        """Wait for session.updated confirmation from OpenAI.

        Returns True ONLY when the response explicitly confirms g711_ulaw
        audio format.  If the format is wrong the call would screech.
        """
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                raw = await asyncio.wait_for(self.openai_ws.recv(), timeout=remaining)
                event = json.loads(raw)
                etype = event.get("type", "")
                if etype == "session.updated":
                    # ── CRITICAL: verify the actual audio format ──
                    session = event.get("session", {})
                    in_fmt = session.get("input_audio_format", "unknown")
                    out_fmt = session.get("output_audio_format", "unknown")
                    print(
                        f"[CALL] session.updated → input={in_fmt}, output={out_fmt}",
                        flush=True,
                    )
                    if out_fmt != "g711_ulaw":
                        print(
                            f"[CALL] *** FORMAT MISMATCH! Got output={out_fmt} "
                            f"instead of g711_ulaw — this causes screeching! ***",
                            flush=True,
                        )
                        # Try one more session.update to force the format
                        await self.openai_ws.send(json.dumps({
                            "type": "session.update",
                            "session": {
                                "input_audio_format": "g711_ulaw",
                                "output_audio_format": "g711_ulaw",
                            },
                        }))
                        # Wait for the corrected confirmation
                        retry_raw = await asyncio.wait_for(
                            self.openai_ws.recv(), timeout=3.0
                        )
                        retry_event = json.loads(retry_raw)
                        if retry_event.get("type") == "session.updated":
                            retry_fmt = retry_event.get("session", {}).get(
                                "output_audio_format", "unknown"
                            )
                            if retry_fmt == "g711_ulaw":
                                print(
                                    "[CALL] Format corrected to g711_ulaw on retry",
                                    flush=True,
                                )
                            else:
                                print(
                                    f"[CALL] *** STILL WRONG FORMAT: {retry_fmt} — aborting ***",
                                    flush=True,
                                )
                                return False
                        else:
                            print(
                                "[CALL] *** Retry did not return session.updated — aborting ***",
                                flush=True,
                            )
                            return False

                    mode = "ElevenLabs TTS" if self.use_elevenlabs else "speech-to-speech"
                    script_mode = "script" if self._script else "free"
                    print(
                        f"[CALL] Session CONFIRMED — {mode}, {script_mode}, g711_ulaw, "
                        f"{vad_type}, temp={temperature}, max={max_tokens}tok",
                        flush=True,
                    )
                    return True
                elif etype == "error":
                    error = event.get("error", {})
                    print(
                        f"[CALL] *** Session REJECTED ({vad_type}): "
                        f"{error.get('message', 'unknown')} | code={error.get('code', '?')} ***",
                        flush=True,
                    )
                    return False
                else:
                    logger.debug(f"Pre-config event: {etype}")
        except asyncio.TimeoutError:
            logger.error(f"Session config timed out (5s) for {vad_type}")
        return False

    # Simple echo gate: only active while AI is speaking + short cooldown.
    # Catches obvious echo without blocking real speech.
    # OpenAI's semantic VAD handles noisy environments — we just prevent echo loops.

    def _is_echo_gate_active(self) -> bool:
        """Check if echo gate should be filtering audio right now."""
        if self._ai_is_responding or self._tts_playing:
            return True
        if time.monotonic() < self._echo_gate_until:
            return True
        return False

    async def _forward_audio(self, mulaw_b64: str):
        """Forward Twilio mulaw 8kHz directly to OpenAI (g711_ulaw — zero conversion).

        Light echo gate — only checks RMS while AI is speaking.
        Passes everything through when AI is silent (zero overhead).
        """
        try:
            if self._is_echo_gate_active():
                mulaw_bytes = base64.b64decode(mulaw_b64)
                pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
                rms = audioop.rms(pcm_8k, 2)

                if rms < Config.ECHO_GATE_RMS_HARD:
                    return  # Silence or quiet echo

                if rms < Config.ECHO_GATE_RMS_SOFT and self._ai_is_responding:
                    return  # Likely echo during active AI speech

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
                except Exception as e:
                    logger.error(f"Error handling OpenAI event: {e}", exc_info=True)
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
            self._first_audio_chunk = True
            self._audio_chunk_count = 0
            self._response_start_time = time.monotonic()

        elif t == "response.audio.delta":
            audio_b64 = event.get("delta", "")
            if audio_b64 and self._twilio_ws and self.stream_sid:
                try:
                    self._audio_chunk_count += 1

                    # Diagnostic logging for first 3 chunks
                    if self._audio_chunk_count <= 3:
                        elapsed = time.monotonic() - getattr(self, '_response_start_time', time.monotonic())
                        raw_bytes = base64.b64decode(audio_b64)
                        print(
                            f"[CALL] Audio chunk #{self._audio_chunk_count} → Twilio "
                            f"({len(audio_b64)} b64 chars, {len(raw_bytes)} bytes, "
                            f"{elapsed:.2f}s since response.created)",
                            flush=True,
                        )

                    # Natural pacing: brief pause before first audio chunk
                    if self._first_audio_chunk:
                        self._first_audio_chunk = False
                        pacing_ms = Config.RESPONSE_PACING_DELAY_MS
                        if pacing_ms > 0:
                            await asyncio.sleep(pacing_ms / 1000.0)

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
                # User interrupted — save transcript for recovery, then clear
                self._response_text_buffer = ""
                saved_transcript = self._current_ai_transcript.strip()
                self._current_ai_transcript = ""
                self._drain_tts_queue()
                logger.info(f"Response cancelled — saved: '{saved_transcript[:60]}'")
                # Inject interrupted context so AI can resume naturally
                if saved_transcript and len(saved_transcript) > 20:
                    try:
                        ws = self.openai_ws
                        if ws:
                            await ws.send(json.dumps({
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "message",
                                    "role": "system",
                                    "content": [{
                                        "type": "input_text",
                                        "text": (
                                            f'[You were interrupted while saying: "{saved_transcript}" '
                                            f"If relevant, naturally continue after addressing the caller's point.]"
                                        ),
                                    }],
                                },
                            }))
                            logger.info("Injected interruption context for AI recovery")
                    except Exception as e:
                        logger.debug(f"Could not inject interruption context: {e}")
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
            if self._ai_is_responding or self._tts_playing:
                # User is interrupting — clear Twilio audio buffer
                logger.info("User interrupted — clearing audio")
                self._drain_tts_queue()
                self._response_text_buffer = ""
                asyncio.create_task(self._gentle_clear(Config.GENTLE_CLEAR_DELAY_MS / 1000.0))

        elif t == "input_audio_buffer.speech_stopped":
            pass  # OpenAI handles turn-taking (create_response=true)

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

    # ── Failure recovery ──────────────────────────────────────

    async def _close_twilio_on_failure(self, reason: str):
        """Close Twilio stream so TwiML falls through to <Say> fallback."""
        logger.warning(f"Closing Twilio stream due to failure: {reason}")
        try:
            if self._twilio_ws:
                await self._twilio_ws.close()
        except Exception:
            pass

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
