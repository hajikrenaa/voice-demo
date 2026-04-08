"""
Twilio Bidirectional Media Stream -- OpenAI Realtime API

Ultra-low latency voice agent.

Mode 1 (ElevenLabs OFF -- default):
  Twilio mulaw 8kHz ──► g711_ulaw ──► OpenAI Realtime (speech-to-speech)
  Twilio mulaw 8kHz ◄── g711_ulaw ◄──┘
  Zero audio conversion on both input AND output. ~300ms latency.

Mode 2 (ElevenLabs ON):
  Twilio mulaw 8kHz ──► g711_ulaw ──► OpenAI Realtime (text-only response)
                                            │ text (sentence-streamed)
  Twilio mulaw 8kHz ◄── ulaw_8000 (native) ◄── ElevenLabs streaming
  Twilio mulaw 8kHz ◄── PCM->mulaw          ◄── OpenAI TTS fallback
"""

import asyncio
import audioop
import base64
import json
import logging
import time
from typing import Optional

import websockets
from openai import AsyncOpenAI

from config import Config
from services.elevenlabs_tts_service import ElevenLabsTTSService
from utils.audio_processing import downsample_24k_to_8k

logger = logging.getLogger(__name__)


class TwilioRealtimeHandler:
    """
    Bridges Twilio <-> OpenAI Realtime API.

    When use_elevenlabs=False:
      - g711_ulaw in/out -- zero conversion, ~300ms latency.
    When use_elevenlabs=True:
      - g711_ulaw input, text output from OpenAI
      - Sentence-by-sentence -> TTS streaming -> mulaw -> Twilio
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
        # Interruption state (manual turn-taking -- create_response=false)
        self._last_interrupted_transcript = ""
        self._interrupt_pending = False
        self._speech_start_time = 0.0
        self._first_audio_chunk = True
        self._audio_chunk_count = 0
        self._response_start_time = 0.0
        self._speech_stopped_time = 0.0  # For end-to-end latency tracking
        # PCM16 buffer for accumulating audio chunks before conversion
        self._pcm_buffer = b""
        # Pre-warmed OpenAI WebSocket (set externally before start event)
        self._prewarm_task: Optional[asyncio.Task] = None
        self._session_preconfigured = False  # True if pre-warm already sent session.update
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
        """Connect to OpenAI Realtime API with a connection timeout.

        If a pre-warmed WebSocket was started at TwiML time, await it
        instead of opening a cold connection — saves ~1s of handshake.
        """
        try:
            if self._prewarm_task:
                # Use the connection that was started at TwiML webhook time
                print("[CALL] Awaiting pre-warmed OpenAI connection...", flush=True)
                try:
                    self.openai_ws = await asyncio.wait_for(
                        self._prewarm_task, timeout=10.0
                    )
                except Exception as e:
                    print(f"[CALL] Pre-warm failed ({e}), cold-connecting...", flush=True)
                    self.openai_ws = None
                finally:
                    self._prewarm_task = None

            if not self.openai_ws:
                # Pre-warm failed or returned dead WebSocket — need fresh connection
                self._session_preconfigured = False
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

            # NOTE: Do NOT set _connected=True yet -- audio must not flow until
            # session config is confirmed (g711_ulaw format active).
            print(f"[CALL] OpenAI WebSocket connected OK", flush=True)

            confirmed = await self._configure_session()
            if not confirmed:
                print("[CALL] *** SESSION CONFIG FAILED -- aborting ***", flush=True)
                await self._close_twilio_on_failure("Session config failed")
                return

            # NOW it's safe to forward audio -- format is confirmed g711_ulaw
            self._connected = True
            print("[CALL] _connected=True -- audio forwarding enabled", flush=True)
            self._receive_task = asyncio.create_task(self._receive_openai_events())

            # Don't trigger AI to speak first — wait for caller to say hello.
            # OpenAI VAD will detect their speech and auto-create a response.

        except asyncio.TimeoutError:
            print("[CALL] *** OpenAI connection TIMED OUT (10s) ***", flush=True)
            self._connected = False
            await self._close_twilio_on_failure("OpenAI connection timed out")
        except Exception as e:
            print(f"[CALL] *** OpenAI connection FAILED: {e} ***", flush=True)
            self._connected = False
            await self._close_twilio_on_failure(str(e))

    async def _trigger_initial_response(self):
        """Send response.create immediately to start the welcome message."""
        try:
            ws = self.openai_ws
            if ws:
                await ws.send(json.dumps({"type": "response.create"}))
                logger.info("Triggered initial AI response")
        except Exception as e:
            logger.error(f"Failed to trigger initial response: {e}")

    def _build_prompt(self) -> str:
        """Build system prompt from script config or use default.

        This prompt is for OpenAI Realtime speech-to-speech mode:
        the model receives raw audio and outputs audio directly.
        There is no separate STT/TTS — the model hears and speaks.
        """
        if not self._script:
            return Config.SYSTEM_PROMPT

        s = self._script
        parts = []
        n = len(s.get("questions", []))

        parts.append(
            "You are on a live phone call. You hear the caller's voice directly. "
            "Be warm, professional, English only. Keep responses to 1-2 sentences. "
            "Speak slowly and clearly — the caller may not understand fast speech."
        )

        if s.get("behaviour"):
            parts.append(s['behaviour'])

        if s.get("welcome"):
            parts.append(
                "Wait for the caller to speak first (they will say hello or greet you). "
                f'Then respond with: "{s["welcome"]}"'
            )

        questions = s.get("questions", [])
        if questions:
            q_list = [f"Q{i+1}: {q.get('question', q) if isinstance(q, dict) else q}"
                      for i, q in enumerate(questions)]
            parts.append(
                f"QUESTIONS — ask ALL {n} in order (Q1, Q2, Q3... do NOT skip any):\n"
                + "\n".join(q_list)
            )

        if s.get("goal"):
            parts.append(f"GOAL: {s['goal']}")

        parts.append(
            "RULES YOU MUST FOLLOW:\n"
            "1. Answer the caller's questions first. If they ask 'what is this call about?' — re-explain.\n"
            "2. Only move to the next question AFTER the caller answers the current one. "
            "Questions, complaints, 'I can't hear you' are NOT answers — address them, then re-ask.\n"
            "3. When caller says 'no', 'wrong', 'that's not right', 'that's wrong', or ANY disagreement — STOP. "
            "You made a mistake. Apologize and ask them to provide it again. "
            "Do NOT repeat your previous wrong answer. Do NOT move to the next question.\n"
            "4. EXACT REPETITION: When confirming what the caller said, use their EXACT words. "
            "Do NOT change numbers (8 stays 8, not 7), units (weeks stays weeks, not months), "
            "tool names (n8n stays n8n, not Jenkins), amounts (1 crore stays 1 crore). Zero changes.\n"
            "5. If caller is frustrated — acknowledge, apologize, ask if they want to continue.\n"
            "6. If audio is bad or they ask to repeat — apologize and repeat your last question.\n"
            "7. If caller asks to start over — say 'Sure, let me start fresh' and wait.\n"
            "8. If caller says they don't want the job, wants to end the call, or says 'bye' — "
            "politely say 'Thank you for your time. Have a great day! Goodbye.' and stop. "
            "Do NOT continue with questions or force a recap.\n"
            "\n"
            "NAMES AND EMAILS:\n"
            "Phone audio can be unclear. Listen carefully.\n"
            "- When confirming a name, say it as a whole word: 'Got it, [name]. Is that correct?'\n"
            "- NEVER say individual letters with dashes (no A-B-C-D). Just say the whole word.\n"
            "- ALWAYS ask 'Is that correct?' and wait for 'yes'.\n"
            "- If they say 'no' or 'wrong' — ask them to spell it again. "
            "Do NOT repeat your wrong version. Do NOT guess a different name.\n"
            "- EMAILS: Only confirm an email the caller has ACTUALLY given you. "
            "NEVER make up or guess an email address. If you don't have it yet, ASK for it.\n"
            "- Do NOT end the call until you have collected ALL required information "
            "or the caller explicitly wants to leave.\n"
            "\n"
            "CONVERSATION FLOW:\n"
            "- After a confirmed answer, react briefly, then ask the next question.\n"
            "- Use exact tool names, framework names — never substitute.\n"
            f"- After ALL {n} questions are answered and confirmed: give a complete recap of "
            "everything collected, then ask 'Did I get everything right?', then say Goodbye.\n"
            "- ENDING: When you say goodbye, include EVERYTHING in ONE response "
            "(thank you + next steps + goodbye). After saying goodbye, STOP completely. "
            "Do NOT respond to the caller's 'bye' or 'thank you' — the call is over."
        )

        return "\n\n".join(parts)

    async def _configure_session(self) -> bool:
        """Configure session -- g711_ulaw format, VAD, tuned for phone accuracy.

        Returns True if session config was confirmed, False on failure.
        Critical: g711_ulaw MUST be confirmed before any audio flows,
        otherwise OpenAI defaults to pcm16 which causes screeching on Twilio.
        """
        # Fast path: session was already configured during pre-warm
        if self._session_preconfigured and not self.use_elevenlabs:
            logger.info("Session pre-configured during pre-warm, waiting for confirmation...")
            temperature = 0.7 if self._script else 0.8
            max_tokens = 400 if self._script else 150
            confirmed = await self._wait_for_session_confirmation(
                Config.VAD_TYPE, temperature, max_tokens
            )
            if confirmed:
                try:
                    await self.openai_ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
                except Exception:
                    pass
                return True
            # Pre-warm config not confirmed — fall through to normal path
            logger.warning("Pre-warm session config not confirmed, re-configuring...")

        if self.use_elevenlabs:
            modalities = ["text"]
        else:
            modalities = ["text", "audio"]

        prompt = self._build_prompt()
        logger.info(f"System prompt ({len(prompt)} chars): {prompt[:100]}...")

        # With g711_ulaw native output, 0.6 is safe (screeching was a pcm16 issue)
        temperature = 0.7 if self._script else 0.8
        max_tokens = 400 if self._script else 150

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
            # For speech-to-speech (no ElevenLabs): use g711_ulaw output -- zero conversion, no screeching
            # For ElevenLabs mode: use pcm16 output (text-only modality, TTS handled separately)
            output_format = "pcm16" if self.use_elevenlabs else "g711_ulaw"
            session_config = {
                "type": "session.update",
                "session": {
                    "modalities": modalities,
                    "instructions": prompt,
                    "voice": Config.REALTIME_VOICE,
                    "input_audio_format": "g711_ulaw",
                    "output_audio_format": output_format,
                    "temperature": temperature,
                    "max_response_output_tokens": max_tokens,
                    "input_audio_transcription": {
                        "model": "whisper-1",
                        "language": "en",
                    },
                    "turn_detection": vad_config,
                    "input_audio_noise_reduction": {
                        "type": "near_field",
                    },
                },
            }
            print(f"[CALL] Sending session config -- input=g711_ulaw, output={output_format}, {vad_type}, temp={temperature}", flush=True)
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
                logger.warning(f"{vad_type} failed -- retrying with server_vad fallback")

        logger.error("ALL session config attempts failed -- audio format is NOT g711_ulaw!")
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
                    expected_out = "pcm16" if self.use_elevenlabs else "g711_ulaw"
                    print(
                        f"[CALL] session.updated -> input={in_fmt}, output={out_fmt} (expected={expected_out})",
                        flush=True,
                    )
                    if out_fmt != expected_out:
                        print(
                            f"[CALL] *** FORMAT MISMATCH! Got output={out_fmt} "
                            f"instead of {expected_out} -- retrying ***",
                            flush=True,
                        )
                        # Try one more session.update to force the format
                        await self.openai_ws.send(json.dumps({
                            "type": "session.update",
                            "session": {
                                "input_audio_format": "g711_ulaw",
                                "output_audio_format": expected_out,
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
                            if retry_fmt == expected_out:
                                print(
                                    f"[CALL] Format corrected to {expected_out} on retry",
                                    flush=True,
                                )
                            else:
                                print(
                                    f"[CALL] *** STILL WRONG FORMAT: {retry_fmt} -- aborting ***",
                                    flush=True,
                                )
                                return False
                        else:
                            print(
                                "[CALL] *** Retry did not return session.updated -- aborting ***",
                                flush=True,
                            )
                            return False

                    mode = "ElevenLabs TTS" if self.use_elevenlabs else "speech-to-speech"
                    script_mode = "script" if self._script else "free"
                    print(
                        f"[CALL] Session CONFIRMED -- {mode}, {script_mode}, in={in_fmt}, out={out_fmt}, "
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
    # OpenAI's semantic VAD handles noisy environments -- we just prevent echo loops.

    def _is_echo_gate_active(self) -> bool:
        """Check if echo gate should be filtering audio right now."""
        if self._ai_is_responding or self._tts_playing:
            return True
        if time.monotonic() < self._echo_gate_until:
            return True
        return False

    async def _forward_audio(self, mulaw_b64: str):
        """Forward Twilio mulaw 8kHz directly to OpenAI (g711_ulaw -- zero conversion).

        Light echo gate -- only checks RMS while AI is speaking.
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
            # Reset PCM buffer for each new response
            self._pcm_buffer = b""

        elif t == "response.audio.delta":
            # Audio from OpenAI -> forward to Twilio
            # g711_ulaw mode: already mulaw, forward directly (zero conversion!)
            # pcm16 mode (ElevenLabs): convert PCM16 24kHz -> mulaw 8kHz
            audio_b64 = event.get("delta", "")
            if audio_b64 and self._twilio_ws and self.stream_sid:
                try:
                    self._audio_chunk_count += 1

                    # Diagnostic logging for first 3 chunks
                    if self._audio_chunk_count <= 3:
                        elapsed = time.monotonic() - self._response_start_time
                        raw_data = base64.b64decode(audio_b64)
                        fmt = "mulaw" if not self.use_elevenlabs else "PCM16"
                        print(
                            f"[CALL] Audio chunk #{self._audio_chunk_count} "
                            f"({len(raw_data)} {fmt} bytes, "
                            f"{elapsed:.2f}s since response.created)",
                            flush=True,
                        )

                    # Log end-to-end latency on first audio chunk
                    if self._first_audio_chunk:
                        self._first_audio_chunk = False
                        if self._speech_stopped_time > 0:
                            e2e_ms = (time.monotonic() - self._speech_stopped_time) * 1000
                            print(
                                f"[CALL] E2E latency: {e2e_ms:.0f}ms "
                                f"(speech_stopped -> first audio to Twilio)",
                                flush=True,
                            )
                        pacing_ms = Config.RESPONSE_PACING_DELAY_MS
                        if pacing_ms > 0:
                            await asyncio.sleep(pacing_ms / 1000.0)

                    if not self.use_elevenlabs:
                        # g711_ulaw output -> forward directly to Twilio (ZERO conversion)
                        await self._twilio_ws.send_json({
                            "event": "media",
                            "streamSid": self.stream_sid,
                            "media": {"payload": audio_b64},
                        })
                    else:
                        # pcm16 output -> buffer and convert to mulaw for Twilio
                        pcm_data = base64.b64decode(audio_b64)
                        self._pcm_buffer += pcm_data
                        PCM_CHUNK = 4800  # 100ms at 24kHz 16-bit mono

                        while len(self._pcm_buffer) >= PCM_CHUNK:
                            chunk = self._pcm_buffer[:PCM_CHUNK]
                            self._pcm_buffer = self._pcm_buffer[PCM_CHUNK:]

                            # Anti-aliased downsample 24kHz -> 8kHz, then PCM16 -> mulaw
                            resampled = downsample_24k_to_8k(chunk)
                            mulaw = audioop.lin2ulaw(resampled, 2)
                            mulaw_b64_chunk = base64.b64encode(mulaw).decode("utf-8")

                            await self._twilio_ws.send_json({
                                "event": "media",
                                "streamSid": self.stream_sid,
                                "media": {"payload": mulaw_b64_chunk},
                            })

                    # Refresh echo cooldown
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
                    self._schedule_hangup(3.0)
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
                self._schedule_hangup(3.0)
            self._current_ai_transcript = ""

        elif t == "response.done":
            status = event.get("response", {}).get("status", "")
            # Safety: if interrupt was pending but speech_stopped never came, resolve it
            if self._interrupt_pending:
                self._interrupt_pending = False
                self._drain_tts_queue()
                self._response_text_buffer = ""
                logger.info("Interrupt pending resolved by response.done (speech_stopped missed)")
            # Flush any remaining PCM audio buffer (only relevant in pcm16/ElevenLabs mode)
            if self.use_elevenlabs:
                await self._flush_pcm_remainder()
            self._ai_is_responding = False
            if status == "cancelled":
                # User interrupted -- save transcript for recovery, then clear
                self._response_text_buffer = ""
                saved_transcript = self._current_ai_transcript.strip()
                self._current_ai_transcript = ""
                self._drain_tts_queue()
                logger.info(f"Response cancelled -- saved: '{saved_transcript[:60]}'")
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
                # Detect restart intent — triggers full session reset
                lower = transcript.lower().strip()
                restart_phrases = ["start over", "start fresh", "restart",
                                   "erase everything", "begin again",
                                   "start from scratch", "reset"]
                if any(phrase in lower for phrase in restart_phrases):
                    logger.info(f"RESTART: Intent detected in '{transcript}'")
                    await self._restart_session()
                    return

                # Only inject transcription when it contains letter-by-letter
                # spelling (e.g. "H-A-J-I-K") — those are more reliable than
                # Whisper's general transcription for accented speech.
                import re
                has_spelling = bool(re.search(r'[A-Z]-[A-Z]-[A-Z]', transcript))
                if has_spelling:
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
                                            f'[Spelled letters detected in caller speech: "{transcript}". '
                                            f"Assemble these letters exactly for the name/email.]"
                                        ),
                                    }],
                                },
                            }))
                            logger.debug(f"Injected spelling hint: {transcript[:60]}")
                    except Exception as e:
                        logger.debug(f"Could not inject spelling hint: {e}")
            else:
                logger.warning("User audio transcription was EMPTY -- likely garbled or too quiet")

        elif t == "conversation.item.input_audio_transcription.failed":
            error = event.get("error", {})
            logger.error(f"Transcription FAILED: {error.get('message', 'unknown')}")

        elif t == "input_audio_buffer.speech_started":
            self._cancel_hangup()
            self._speech_start_time = time.monotonic()
            if self._ai_is_responding or self._tts_playing:
                # Stage 1: Clear Twilio buffer immediately (user hears silence),
                # but DON'T drain TTS queue yet — wait for speech_stopped to
                # distinguish backchannel ("uh-huh") from real interruption.
                logger.info("User speech during AI response -- clearing Twilio buffer (pending eval)")
                self._interrupt_pending = True
                asyncio.create_task(self._gentle_clear(Config.GENTLE_CLEAR_DELAY_MS / 1000.0))

        elif t == "input_audio_buffer.speech_stopped":
            self._speech_stopped_time = time.monotonic()
            if self._interrupt_pending:
                # Stage 2: Evaluate speech duration to classify interruption
                speech_duration_ms = (time.monotonic() - self._speech_start_time) * 1000
                self._interrupt_pending = False

                if speech_duration_ms < Config.BACKCHANNEL_MAX_DURATION_MS:
                    # Short utterance = backchannel ("uh-huh", "yeah", "okay")
                    # Don't drain TTS queue — AI resumes via interrupted context injection
                    logger.info(
                        f"Backchannel ({speech_duration_ms:.0f}ms < "
                        f"{Config.BACKCHANNEL_MAX_DURATION_MS}ms) -- not draining"
                    )
                else:
                    # Real interruption — drain pending TTS and clear buffer
                    logger.info(
                        f"Real interruption ({speech_duration_ms:.0f}ms) -- draining TTS queue"
                    )
                    self._drain_tts_queue()
                    self._response_text_buffer = ""

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
        """Process TTS queue sequentially -- sentences play in order."""
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
        """Try ElevenLabs streaming first, fall back to streaming OpenAI TTS."""
        if not self._twilio_ws or not self.stream_sid:
            return

        if self._elevenlabs_tts and self._elevenlabs_available:
            try:
                await self._stream_elevenlabs_tts(text)
                return
            except Exception as e:
                logger.warning(f"ElevenLabs failed -- switching to OpenAI TTS: {e}")
                self._elevenlabs_available = False

        await self._stream_openai_tts(text)

    # ── ElevenLabs streaming TTS (native ulaw) ─────────────

    async def _stream_elevenlabs_tts(self, text: str):
        """Stream ElevenLabs TTS -> ulaw_8000 (native) -> Twilio.

        ElevenLabs returns raw mu-law bytes at 8 kHz -- no conversion needed.
        Chunks are batched into ~400 ms packets for smooth playback.
        """
        mulaw_buffer = b""
        CHUNK_SIZE = 3200  # ~400ms at 8kHz mulaw
        chunks_sent = 0

        async for chunk in self._elevenlabs_tts.synthesize_stream(text):
            mulaw_buffer += chunk

            while len(mulaw_buffer) >= CHUNK_SIZE:
                send_chunk = mulaw_buffer[:CHUNK_SIZE]
                mulaw_buffer = mulaw_buffer[CHUNK_SIZE:]
                chunk_b64 = base64.b64encode(send_chunk).decode("utf-8")

                if self._twilio_ws and self.stream_sid:
                    await self._twilio_ws.send_json({
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {"payload": chunk_b64},
                    })
                    self._echo_gate_until = time.monotonic() + self._ECHO_COOLDOWN
                    chunks_sent += 1

        # Flush remaining data
        if mulaw_buffer and self._twilio_ws and self.stream_sid:
            chunk_b64 = base64.b64encode(mulaw_buffer).decode("utf-8")
            await self._twilio_ws.send_json({
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": chunk_b64},
            })
            self._echo_gate_until = time.monotonic() + self._ECHO_COOLDOWN
            chunks_sent += 1

        logger.info(f"ElevenLabs streamed ({chunks_sent} chunks): {text[:50]}")

    # ── OpenAI streaming TTS (fallback) ────────────────────

    async def _stream_openai_tts(self, text: str):
        """Stream OpenAI TTS -> PCM 24kHz -> resample 8kHz -> mulaw -> Twilio."""
        try:
            if not self._openai_tts_client:
                self._openai_tts_client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)

            async with self._openai_tts_client.audio.speech.with_streaming_response.create(
                model=Config.TTS_MODEL,
                voice=Config.REALTIME_VOICE,
                input=text,
                response_format="pcm",
            ) as response:
                pcm_buffer = b""
                chunks_sent = 0
                PCM_CHUNK = 4800  # 100ms at 24kHz 16-bit mono

                async for data in response.iter_bytes(chunk_size=PCM_CHUNK):
                    pcm_buffer += data

                    while len(pcm_buffer) >= PCM_CHUNK:
                        pcm_chunk = pcm_buffer[:PCM_CHUNK]
                        pcm_buffer = pcm_buffer[PCM_CHUNK:]

                        resampled = downsample_24k_to_8k(pcm_chunk)
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
                        resampled = downsample_24k_to_8k(pcm_buffer[:usable])
                        if resampled:
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

    # ── Flush remaining PCM buffer ──────────────────────────

    async def _flush_pcm_remainder(self):
        """Flush any remaining PCM data in the buffer (ElevenLabs/pcm16 mode only)."""
        if not self._pcm_buffer or not self._twilio_ws or not self.stream_sid:
            self._pcm_buffer = b""
            return
        try:
            usable = len(self._pcm_buffer) - (len(self._pcm_buffer) % 2)
            if usable > 0:
                resampled = downsample_24k_to_8k(self._pcm_buffer[:usable])
                if resampled:
                    mulaw = audioop.lin2ulaw(resampled, 2)
                    mulaw_b64 = base64.b64encode(mulaw).decode("utf-8")
                    await self._twilio_ws.send_json({
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {"payload": mulaw_b64},
                    })
                    self._echo_gate_until = time.monotonic() + self._ECHO_COOLDOWN
        except Exception as e:
            logger.error(f"Error flushing PCM remainder: {e}")
        finally:
            self._pcm_buffer = b""

    # ── Session restart (user asked to start over) ──────────

    async def _restart_session(self):
        """Restart OpenAI session with clean context — user asked to start over.

        Closes the WebSocket (drops all conversation history), resets state,
        and reconnects fresh with the initial greeting.
        """
        logger.info("RESTART: Closing session and reconnecting fresh")

        # Cancel receive task
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except (asyncio.CancelledError, Exception):
                pass

        # Close old WebSocket (drops all context)
        if self.openai_ws:
            try:
                await self.openai_ws.close()
            except Exception:
                pass
            self.openai_ws = None

        # Reset all state
        self._connected = False
        self._ai_is_responding = False
        self._tts_playing = False
        self._current_ai_transcript = ""
        self._response_text_buffer = ""
        self._interrupt_pending = False
        self._pcm_buffer = b""
        self._session_preconfigured = False
        self._goodbye_detected = False
        self._cancel_hangup()
        self._drain_tts_queue()

        # Clear Twilio audio buffer
        try:
            if self._twilio_ws and self.stream_sid:
                await self._twilio_ws.send_json({
                    "event": "clear",
                    "streamSid": self.stream_sid,
                })
        except Exception:
            pass

        # Reconnect with fresh context + greeting
        await self._connect_openai()
        logger.info("RESTART: Fresh session started")

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
        """Schedule a hangup -- cancellable if user speaks again."""
        self._cancel_hangup()
        self._goodbye_detected = True
        self._hangup_task = asyncio.create_task(self._hangup_after_delay(delay))

    def _cancel_hangup(self):
        """Cancel pending hangup because user is still talking."""
        if self._hangup_task and not self._hangup_task.done():
            self._hangup_task.cancel()
            self._goodbye_detected = False
            logger.info("Hangup cancelled -- user is still speaking")

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
            logger.info("Hangup was cancelled -- conversation continues")
        except Exception as e:
            logger.error(f"Auto-hangup failed: {e}")

    @staticmethod
    def _sync_hangup_call(call_sid: str):
        """Synchronous Twilio hangup -- runs in a thread."""
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
