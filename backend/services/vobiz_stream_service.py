"""
Vobiz Bidirectional Media Stream — OpenAI Realtime API

Ultra-low latency voice agent bridging Vobiz ↔ OpenAI Realtime.

Vobiz WebSocket protocol:
  Incoming (Vobiz → us):
    {"event":"start",  "start":{"callId":"...","streamId":"...","mediaFormat":{...}}}
    {"event":"media",  "media":{"payload":"<base64-mulaw>","contentType":"audio/x-mulaw","sampleRate":8000}}
    {"event":"stop",   "reason":"call_ended"}

  Outgoing (us → Vobiz):
    {"event":"playAudio","media":{"contentType":"audio/x-mulaw","sampleRate":8000,"payload":"<base64-mulaw>"}}
    {"event":"clearAudio"}   — clears queued audio (interruption)

Mode 1 (ElevenLabs OFF — default):
  Vobiz mulaw 8kHz ──► g711_ulaw ──► OpenAI Realtime (speech-to-speech)
  Vobiz mulaw 8kHz ◄── g711_ulaw ◄──┘
  Zero audio conversion on both input AND output. ~300ms latency.

Mode 2 (ElevenLabs ON):
  Vobiz mulaw 8kHz ──► g711_ulaw ──► OpenAI Realtime (text-only response)
                                          │ text (sentence-streamed)
  Vobiz mulaw 8kHz ◄── ulaw_8000 (native) ◄── ElevenLabs streaming
  Vobiz mulaw 8kHz ◄── PCM->mulaw          ◄── OpenAI TTS fallback
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

# Vobiz REST API base
VOBIZ_API_BASE = "https://api.vobiz.ai/api/v1"


class VobizRealtimeHandler:
    """
    Bridges Vobiz ↔ OpenAI Realtime API.

    Audio format: audio/x-mulaw 8kHz (same as Twilio g711_ulaw — zero conversion!)

    When use_elevenlabs=False:
      - g711_ulaw in/out — zero conversion, ~300ms latency.
    When use_elevenlabs=True:
      - g711_ulaw input, text output from OpenAI
      - Sentence-by-sentence → TTS streaming → mulaw → Vobiz
      - Uses ordered queue so sentences never overlap or reorder.
    """

    OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"

    def __init__(self, use_elevenlabs: bool = False, active_script: dict = None):
        self.stream_id: Optional[str] = None
        self.call_id: Optional[str] = None
        self.openai_ws = None
        self._connected = False
        self._vobiz_ws = None
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
        self._ai_is_responding = False   # True while OpenAI is generating audio
        self._tts_playing = False        # True while TTS audio is being sent to Vobiz
        # Interruption state
        self._last_interrupted_transcript = ""
        self._interrupt_pending = False
        self._speech_start_time = 0.0
        self._first_audio_chunk = True
        self._audio_chunk_count = 0
        self._response_start_time = 0.0
        self._speech_stopped_time = 0.0
        # PCM16 buffer for ElevenLabs mode
        self._pcm_buffer = b""
        # Pre-warmed OpenAI WebSocket
        self._prewarm_task: Optional[asyncio.Task] = None
        self._session_preconfigured = False
        # Echo gate
        self._echo_gate_until = 0.0
        self._ECHO_COOLDOWN = Config.ECHO_COOLDOWN_S
        # Estimated time when Vobiz finishes playing all queued TTS audio
        self._estimated_playback_end = 0.0

    # ── Vobiz message handling ──────────────────────────────────────────────

    async def handle_vobiz_message(self, vobiz_ws, message: dict):
        """Process a JSON message from Vobiz's media stream."""
        event = message.get("event")

        if event == "start":
            start_data = message.get("start", {})
            self.stream_id = start_data.get("streamId")
            self.call_id = start_data.get("callId")
            self._vobiz_ws = vobiz_ws

            # Check extra_headers for elevenlabs flag
            # Vobiz sends extra_headers as "key=value,key2=value2" (NOT JSON)
            extra_headers_raw = message.get("extra_headers", "")
            extra_headers = {}
            if isinstance(extra_headers_raw, dict):
                extra_headers = extra_headers_raw
            elif isinstance(extra_headers_raw, str) and extra_headers_raw:
                try:
                    extra_headers = json.loads(extra_headers_raw)
                except (json.JSONDecodeError, ValueError):
                    for pair in extra_headers_raw.split(","):
                        if "=" in pair:
                            k, v = pair.split("=", 1)
                            extra_headers[k.strip()] = v.strip()

            elevenlabs_param = extra_headers.get("elevenlabs", "false")
            if str(elevenlabs_param).lower() == "true":
                self.use_elevenlabs = True

            # Ensure TTS worker is running if ElevenLabs mode (from constructor or extra_headers)
            if self.use_elevenlabs and not self._tts_worker_task:
                if not self._elevenlabs_tts:
                    self._elevenlabs_tts = ElevenLabsTTSService()
                self._elevenlabs_available = True
                self._tts_worker_task = asyncio.create_task(self._tts_worker())

            mode_label = "ElevenLabs TTS" if self.use_elevenlabs else "speech-to-speech"
            has_script = "with script" if self._script else "no script"
            logger.info(
                f"Vobiz stream started — streamId={self.stream_id}, "
                f"callId={self.call_id}, mode={mode_label}, {has_script}"
            )
            await self._connect_openai()

        elif event == "media":
            payload = message.get("media", {}).get("payload", "")
            if payload and self._connected:
                await self._forward_audio(payload)

        elif event == "stop":
            reason = message.get("reason", "unknown")
            logger.info(f"Vobiz stream stopped — callId={self.call_id}, reason={reason}")
            await self._full_cleanup()

    # ── OpenAI Realtime connection ──────────────────────────────────────────

    async def _connect_openai(self):
        """Connect to OpenAI Realtime API (with pre-warm support)."""
        try:
            if self._prewarm_task:
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

            print("[CALL] OpenAI WebSocket connected OK", flush=True)
            confirmed = await self._configure_session()
            if not confirmed:
                print("[CALL] *** SESSION CONFIG FAILED — aborting ***", flush=True)
                await self._close_vobiz_on_failure("Session config failed")
                return

            self._connected = True
            print("[CALL] _connected=True — audio forwarding enabled", flush=True)
            self._receive_task = asyncio.create_task(self._receive_openai_events())
            await self._trigger_initial_response()

        except asyncio.TimeoutError:
            print("[CALL] *** OpenAI connection TIMED OUT (10s) ***", flush=True)
            self._connected = False
            await self._close_vobiz_on_failure("OpenAI connection timed out")
        except Exception as e:
            print(f"[CALL] *** OpenAI connection FAILED: {e} ***", flush=True)
            self._connected = False
            await self._close_vobiz_on_failure(str(e))

    async def _trigger_initial_response(self):
        """Send response.create to start the welcome message."""
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
                f'Start the conversation immediately by saying: "{s["welcome"]}" '
                "Do NOT wait for the caller to speak first — you are making an outbound call, "
                "so introduce yourself right away."
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
            "NAMES AND EMAILS — CRITICAL ACCURACY RULES:\n"
            "Phone audio is LOW quality. Single letters sound identical (B/D/P/T, M/N, S/F). "
            "You WILL mishear names if they just spell letters. Use word-spelling to fix this.\n"
            "\n"
            "1. NEVER guess or assume a name. NEVER substitute what you heard with a common name. "
            "If it sounds like 'Hajik', say 'Hajik' — do NOT change it to Rajiv, Sajith, or Ranjith.\n"
            "2. After the caller says their name, ALWAYS ask them to spell using words: "
            "'Could you spell that using words? For example, A as in Apple, B as in Boy.'\n"
            "3. When they say words like 'H as in Hotel, A as in Apple, J as in Japan, I as in India, K as in King', "
            "take the FIRST LETTER of each word: H-A-J-I-K → Hajik. Write EXACTLY those letters.\n"
            "4. Do NOT 'correct' the result to a name you recognize. "
            "The spelled letters are the truth — they override whatever you thought you heard.\n"
            "5. When confirming back, just say the name naturally: 'Got it, Hajik. Is that correct?' "
            "Do NOT spell it back with words. Do NOT say letters with dashes. Just say the name.\n"
            "6. If they say 'no' or 'wrong': apologize, FORGET your previous attempt completely, "
            "and ask them to spell again using words. Do NOT repeat your wrong version.\n"
            "7. EMAILS: NEVER guess or auto-complete. Ask them to spell the part before @ using words too. "
            "If they say 'H as in Hotel, A as in Apple, J as in Japan, I as in India, K as in King, "
            "R as in Red, E as in Echo, N as in November, A as in Apple, A as in Apple at gmail.com', "
            "the email is hajikrenaa@gmail.com. Write EXACTLY those letters. Do NOT rearrange.\n"
            "8. Do NOT end the call until you have collected ALL required information "
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
        """Configure OpenAI session — g711_ulaw format (zero-conversion for Vobiz mulaw).

        Returns True if session config was confirmed, False on failure.
        """
        # Fast path: session was already configured during pre-warm
        if self._session_preconfigured and not self.use_elevenlabs:
            logger.info("Session pre-configured during pre-warm, waiting for confirmation...")
            temperature = 0.6 if self._script else 0.8
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
            logger.warning("Pre-warm session config not confirmed, re-configuring...")

        if self.use_elevenlabs:
            modalities = ["text"]
        else:
            modalities = ["text", "audio"]

        prompt = self._build_prompt()
        logger.info(f"System prompt ({len(prompt)} chars): {prompt[:100]}...")

        temperature = 0.6 if self._script else 0.8
        max_tokens = 400 if self._script else 150

        # Try preferred VAD first, then fall back to server_vad
        vad_configs = []
        if Config.VAD_TYPE == "semantic_vad":
            vad_configs.append({
                "type": "semantic_vad",
                "eagerness": Config.SEMANTIC_VAD_EAGERNESS,
            })
        vad_configs.append({
            "type": "server_vad",
            "threshold": Config.VAD_THRESHOLD,
            "prefix_padding_ms": Config.VAD_PREFIX_PADDING_MS,
            "silence_duration_ms": Config.VAD_SILENCE_DURATION_MS,
        })

        for vad_config in vad_configs:
            vad_type = vad_config["type"]
            # g711_ulaw matches Vobiz audio/x-mulaw — ZERO conversion needed
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
            print(
                f"[CALL] Sending session config — input=g711_ulaw, output={output_format}, "
                f"{vad_type}, temp={temperature}",
                flush=True,
            )
            await self.openai_ws.send(json.dumps(session_config))

            confirmed = await self._wait_for_session_confirmation(
                vad_type, temperature, max_tokens
            )
            if confirmed:
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
        """Wait for session.updated confirmation from OpenAI."""
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                raw = await asyncio.wait_for(self.openai_ws.recv(), timeout=remaining)
                event = json.loads(raw)
                etype = event.get("type", "")
                if etype == "session.updated":
                    session = event.get("session", {})
                    in_fmt = session.get("input_audio_format", "unknown")
                    out_fmt = session.get("output_audio_format", "unknown")
                    expected_out = "pcm16" if self.use_elevenlabs else "g711_ulaw"
                    print(
                        f"[CALL] session.updated -> input={in_fmt}, output={out_fmt} "
                        f"(expected={expected_out})",
                        flush=True,
                    )
                    if out_fmt != expected_out:
                        print(
                            f"[CALL] *** FORMAT MISMATCH! Got output={out_fmt} "
                            f"instead of {expected_out} — retrying ***",
                            flush=True,
                        )
                        expected_modalities = ["text"] if self.use_elevenlabs else ["text", "audio"]
                        await self.openai_ws.send(json.dumps({
                            "type": "session.update",
                            "session": {
                                "modalities": expected_modalities,
                                "input_audio_format": "g711_ulaw",
                                "output_audio_format": expected_out,
                            },
                        }))
                        retry_raw = await asyncio.wait_for(
                            self.openai_ws.recv(), timeout=3.0
                        )
                        retry_event = json.loads(retry_raw)
                        if retry_event.get("type") == "session.updated":
                            retry_fmt = retry_event.get("session", {}).get(
                                "output_audio_format", "unknown"
                            )
                            if retry_fmt == expected_out:
                                print(f"[CALL] Format corrected to {expected_out} on retry", flush=True)
                            else:
                                print(f"[CALL] *** STILL WRONG FORMAT: {retry_fmt} — aborting ***", flush=True)
                                return False
                        else:
                            print("[CALL] *** Retry did not return session.updated — aborting ***", flush=True)
                            return False

                    mode = "ElevenLabs TTS" if self.use_elevenlabs else "speech-to-speech"
                    script_mode = "script" if self._script else "free"
                    print(
                        f"[CALL] Session CONFIRMED — {mode}, {script_mode}, "
                        f"in={in_fmt}, out={out_fmt}, {vad_type}, "
                        f"temp={temperature}, max={max_tokens}tok",
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

    # ── Echo gate ───────────────────────────────────────────────────────────

    def _is_echo_gate_active(self) -> bool:
        if self._ai_is_responding or self._tts_playing:
            return True
        if time.monotonic() < self._estimated_playback_end + self._ECHO_COOLDOWN:
            return True
        if time.monotonic() < self._echo_gate_until:
            return True
        return False

    async def _forward_audio(self, mulaw_b64: str):
        """Forward Vobiz mulaw 8kHz directly to OpenAI (g711_ulaw — zero conversion)."""
        try:
            if self._is_echo_gate_active():
                mulaw_bytes = base64.b64decode(mulaw_b64)
                pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
                rms = audioop.rms(pcm_8k, 2)

                if rms < Config.ECHO_GATE_RMS_HARD:
                    return   # Silence or quiet echo

                playback_active = self._ai_is_responding or time.monotonic() < self._estimated_playback_end
                if rms < Config.ECHO_GATE_RMS_SOFT and playback_active:
                    return   # Likely echo during active AI/TTS playback

            ws = self.openai_ws
            if not ws:
                return
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": mulaw_b64,
            }))
        except Exception as e:
            logger.error(f"Error forwarding audio: {e}")

    # ── OpenAI event processing ─────────────────────────────────────────────

    async def _receive_openai_events(self):
        """Receive events from OpenAI and forward audio to Vobiz."""
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

    async def _send_audio_to_vobiz(self, mulaw_b64: str):
        """Send base64-encoded mulaw audio to Vobiz via playAudio event."""
        ws = self._vobiz_ws
        if not ws or not self.stream_id:
            return
        await ws.send_json({
            "event": "playAudio",
            "media": {
                "contentType": "audio/x-mulaw",
                "sampleRate": 8000,
                "payload": mulaw_b64,
            },
        })

    async def _clear_vobiz_audio(self):
        """Send clearAudio event to Vobiz to stop any queued audio playback."""
        self._estimated_playback_end = 0.0
        ws = self._vobiz_ws
        if not ws:
            return
        try:
            await ws.send_json({"event": "clearAudio"})
        except Exception as e:
            logger.debug(f"clearAudio failed: {e}")

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
            self._pcm_buffer = b""

        elif t == "response.audio.delta":
            # Audio from OpenAI → forward to Vobiz
            # g711_ulaw mode: already mulaw, forward directly (ZERO conversion!)
            # pcm16 mode (ElevenLabs): convert PCM16 24kHz → mulaw 8kHz
            audio_b64 = event.get("delta", "")
            if audio_b64 and self._vobiz_ws and self.stream_id:
                if self._interrupt_pending:
                    return
                try:
                    self._audio_chunk_count += 1

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

                    if self._first_audio_chunk:
                        self._first_audio_chunk = False
                        if self._speech_stopped_time > 0:
                            e2e_ms = (time.monotonic() - self._speech_stopped_time) * 1000
                            print(
                                f"[CALL] E2E latency: {e2e_ms:.0f}ms "
                                f"(speech_stopped -> first audio to Vobiz)",
                                flush=True,
                            )
                        pacing_ms = Config.RESPONSE_PACING_DELAY_MS
                        if pacing_ms > 0:
                            await asyncio.sleep(pacing_ms / 1000.0)

                    if not self.use_elevenlabs:
                        # g711_ulaw output → send directly to Vobiz as mulaw (ZERO conversion)
                        await self._send_audio_to_vobiz(audio_b64)
                    else:
                        # pcm16 output → buffer and convert to mulaw for Vobiz
                        pcm_data = base64.b64decode(audio_b64)
                        self._pcm_buffer += pcm_data
                        PCM_CHUNK = 4800  # 100ms at 24kHz 16-bit mono

                        while len(self._pcm_buffer) >= PCM_CHUNK:
                            chunk = self._pcm_buffer[:PCM_CHUNK]
                            self._pcm_buffer = self._pcm_buffer[PCM_CHUNK:]
                            resampled = downsample_24k_to_8k(chunk)
                            mulaw = audioop.lin2ulaw(resampled, 2)
                            mulaw_b64_chunk = base64.b64encode(mulaw).decode("utf-8")
                            await self._send_audio_to_vobiz(mulaw_b64_chunk)

                    # Refresh echo cooldown
                    self._echo_gate_until = time.monotonic() + self._ECHO_COOLDOWN
                except Exception as e:
                    logger.error(f"Error sending audio to Vobiz: {e}")

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
            if self._interrupt_pending:
                self._interrupt_pending = False
                self._drain_tts_queue()
                self._response_text_buffer = ""
                logger.info("Interrupt pending resolved by response.done")
            if self.use_elevenlabs:
                await self._flush_pcm_remainder()
            self._ai_is_responding = False
            if status == "cancelled":
                self._response_text_buffer = ""
                saved_transcript = self._current_ai_transcript.strip()
                self._current_ai_transcript = ""
                self._drain_tts_queue()
                logger.info(f"Response cancelled — saved: '{saved_transcript[:60]}'")
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
                                            "[You were interrupted by the caller. "
                                            "Do NOT repeat or finish what you were saying. "
                                            "Focus entirely on what the caller just said. "
                                            "If they answered your question, acknowledge briefly and move to the next question. "
                                            "If they asked something, answer it then continue forward. "
                                            "Never go back to repeat interrupted content.]"
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
                lower = transcript.lower().strip()
                restart_phrases = ["start over", "start fresh", "restart",
                                   "erase everything", "begin again",
                                   "start from scratch", "reset"]
                if any(phrase in lower for phrase in restart_phrases):
                    logger.info(f"RESTART: Intent detected in '{transcript}'")
                    await self._restart_session()
                    return

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
                logger.warning("User audio transcription was EMPTY")

        elif t == "conversation.item.input_audio_transcription.failed":
            error = event.get("error", {})
            logger.error(f"Transcription FAILED: {error.get('message', 'unknown')}")

        elif t == "input_audio_buffer.speech_started":
            self._cancel_hangup()
            self._speech_start_time = time.monotonic()
            playback_active = time.monotonic() < self._estimated_playback_end
            if self._ai_is_responding:
                logger.info("User speech during AI response — clearing Vobiz buffer (pending eval)")
                self._interrupt_pending = True
                asyncio.create_task(self._gentle_clear(Config.GENTLE_CLEAR_DELAY_MS / 1000.0))
            elif playback_active:
                logger.info("Speech detected during TTS playback — likely echo, not interrupting")

        elif t == "input_audio_buffer.speech_stopped":
            self._speech_stopped_time = time.monotonic()
            if self._interrupt_pending:
                speech_duration_ms = (time.monotonic() - self._speech_start_time) * 1000
                self._interrupt_pending = False

                if speech_duration_ms < Config.BACKCHANNEL_MAX_DURATION_MS:
                    logger.info(
                        f"Backchannel ({speech_duration_ms:.0f}ms < "
                        f"{Config.BACKCHANNEL_MAX_DURATION_MS}ms) — not draining"
                    )
                else:
                    logger.info(
                        f"Real interruption ({speech_duration_ms:.0f}ms) — cancelling response"
                    )
                    self._drain_tts_queue()
                    self._response_text_buffer = ""
                    if self._ai_is_responding:
                        try:
                            ws = self.openai_ws
                            if ws:
                                await ws.send(json.dumps({"type": "response.cancel"}))
                        except Exception as e:
                            logger.debug(f"Could not cancel response: {e}")

        elif t == "error":
            error = event.get("error", {})
            logger.error(
                f"OpenAI error: {error.get('message', 'unknown')} "
                f"(code={error.get('code', '?')})"
            )

    # ── Interruption helpers ────────────────────────────────────────────────

    async def _gentle_clear(self, delay_s: float):
        """After a brief delay, send clearAudio to Vobiz (stops AI audio playback)."""
        await asyncio.sleep(delay_s)
        if self._interrupt_pending or self._ai_is_responding:
            await self._clear_vobiz_audio()

    def _drain_tts_queue(self):
        """Empty the TTS queue (used on real interruptions to stop ElevenLabs pipeline)."""
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
                self._tts_queue.task_done()
            except asyncio.QueueEmpty:
                break

    # ── ElevenLabs TTS pipeline ─────────────────────────────────────────────

    def _flush_sentences(self):
        """Extract complete sentences from the text buffer and enqueue for TTS."""
        import re
        sentence_end = re.compile(r'(?<=[.!?])\s+')
        parts = sentence_end.split(self._response_text_buffer)
        if len(parts) > 1:
            for sentence in parts[:-1]:
                sentence = sentence.strip()
                if sentence:
                    asyncio.create_task(self._enqueue_tts(sentence))
            self._response_text_buffer = parts[-1]

    async def _enqueue_tts(self, text: str):
        """Enqueue text for TTS processing."""
        try:
            await asyncio.wait_for(self._tts_queue.put(text), timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning(f"TTS queue full, dropping: {text[:40]}")

    async def _tts_worker(self):
        """Worker that processes TTS queue and sends audio to Vobiz."""
        logger.info("TTS worker started")
        try:
            while True:
                self._tts_playing = False
                text = await self._tts_queue.get()
                if text is None:
                    break
                self._tts_playing = True
                try:
                    await self._synthesize_and_send(text)
                except Exception as e:
                    logger.error(f"TTS worker error: {e}")
                finally:
                    self._tts_queue.task_done()
        finally:
            self._tts_playing = False
            logger.info("TTS worker stopped")

    async def _synthesize_and_send(self, text: str):
        """Synthesize text to speech and stream audio to Vobiz."""
        if not self._vobiz_ws or not self.stream_id:
            return
        try:
            if self._elevenlabs_tts and self._elevenlabs_available:
                audio_bytes = await self._elevenlabs_tts.synthesize(text)
                if audio_bytes:
                    CHUNK_SIZE = 4000
                    for i in range(0, len(audio_bytes), CHUNK_SIZE):
                        chunk = audio_bytes[i:i + CHUNK_SIZE]
                        mulaw_b64 = base64.b64encode(chunk).decode("utf-8")
                        await self._send_audio_to_vobiz(mulaw_b64)
                    # Estimate when Vobiz will finish playing this audio
                    playback_secs = len(audio_bytes) / 8000.0
                    now = time.monotonic()
                    play_start = max(now, self._estimated_playback_end)
                    self._estimated_playback_end = play_start + playback_secs
                    self._echo_gate_until = self._estimated_playback_end + self._ECHO_COOLDOWN
                    logger.info(f"TTS sent {len(audio_bytes)} bytes (~{playback_secs:.1f}s) for: {text[:50]}...")
                return
        except Exception as e:
            logger.warning(f"ElevenLabs TTS failed ({e}), falling back to OpenAI TTS")
            self._elevenlabs_available = False

        # OpenAI TTS fallback
        try:
            if not self._openai_tts_client:
                self._openai_tts_client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
            async with self._openai_tts_client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="alloy",
                input=text,
                response_format="pcm",
            ) as response:
                pcm_buffer = b""
                async for chunk in response.iter_bytes(chunk_size=4800):
                    pcm_buffer += chunk
                    PCM_CHUNK = 4800
                    while len(pcm_buffer) >= PCM_CHUNK:
                        seg = pcm_buffer[:PCM_CHUNK]
                        pcm_buffer = pcm_buffer[PCM_CHUNK:]
                        resampled = downsample_24k_to_8k(seg)
                        mulaw = audioop.lin2ulaw(resampled, 2)
                        mulaw_b64 = base64.b64encode(mulaw).decode("utf-8")
                        await self._send_audio_to_vobiz(mulaw_b64)
                        self._echo_gate_until = time.monotonic() + self._ECHO_COOLDOWN
                # Flush remainder
                if pcm_buffer:
                    resampled = downsample_24k_to_8k(pcm_buffer)
                    mulaw = audioop.lin2ulaw(resampled, 2)
                    mulaw_b64 = base64.b64encode(mulaw).decode("utf-8")
                    await self._send_audio_to_vobiz(mulaw_b64)
        except Exception as e:
            logger.error(f"OpenAI TTS fallback also failed: {e}")

    async def _flush_pcm_remainder(self):
        """Flush any leftover PCM buffer at end of response (ElevenLabs mode)."""
        if self._pcm_buffer:
            try:
                resampled = downsample_24k_to_8k(self._pcm_buffer)
                mulaw = audioop.lin2ulaw(resampled, 2)
                mulaw_b64 = base64.b64encode(mulaw).decode("utf-8")
                await self._send_audio_to_vobiz(mulaw_b64)
            except Exception as e:
                logger.error(f"Error flushing PCM remainder: {e}")
            self._pcm_buffer = b""

    # ── Goodbye / hangup ────────────────────────────────────────────────────

    def _contains_goodbye(self, text: str) -> bool:
        lower = text.lower()
        goodbye_phrases = ["goodbye", "bye bye", "have a great day", "take care", "farewell"]
        return any(phrase in lower for phrase in goodbye_phrases)

    def _schedule_hangup(self, delay_s: float):
        """Schedule a call hangup after delay_s seconds."""
        if not self._goodbye_detected:
            self._goodbye_detected = True
            self._hangup_task = asyncio.create_task(self._delayed_hangup(delay_s))
            logger.info(f"Goodbye detected — scheduling hangup in {delay_s}s")

    def _cancel_hangup(self):
        if self._hangup_task and not self._hangup_task.done():
            self._hangup_task.cancel()
            self._goodbye_detected = False
            logger.info("Hangup cancelled (caller spoke again)")

    async def _delayed_hangup(self, delay_s: float):
        await asyncio.sleep(delay_s)
        try:
            if self._vobiz_ws:
                await self._vobiz_ws.close()
        except Exception:
            pass

    # ── Session restart (on "start over" command) ───────────────────────────

    async def _restart_session(self):
        """Fully reset the OpenAI session to clear conversation history."""
        logger.info("Restarting OpenAI session (caller requested start over)...")
        try:
            if self.openai_ws:
                await self.openai_ws.close()
        except Exception:
            pass
        self.openai_ws = None
        self._connected = False
        self._current_ai_transcript = ""
        self._response_text_buffer = ""
        self._goodbye_detected = False
        self._pcm_buffer = b""
        await self._connect_openai()

    # ── Vobiz API: hangup call via REST ────────────────────────────────────

    async def hangup_via_api(self):
        """Hang up the current call using the Vobiz REST API."""
        if not self.call_id:
            logger.warning("No call_id to hang up")
            return
        try:
            import httpx
            url = (
                f"https://api.vobiz.ai/api/v1/Account/{Config.VOBIZ_AUTH_ID}"
                f"/Call/{self.call_id}/"
            )
            headers = {
                "X-Auth-ID": Config.VOBIZ_AUTH_ID,
                "X-Auth-Token": Config.VOBIZ_AUTH_TOKEN,
            }
            async with httpx.AsyncClient() as client:
                resp = await client.delete(url, headers=headers, timeout=10.0)
            if resp.status_code in (200, 204):
                logger.info(f"Hung up call {self.call_id} via Vobiz API")
            else:
                logger.error(f"Hang up call {self.call_id} failed: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.error(f"Hang up via Vobiz API failed: {e}")

    # ── Cleanup ─────────────────────────────────────────────────────────────

    async def _close_vobiz_on_failure(self, reason: str):
        """Close the Vobiz WebSocket when OpenAI connection fails."""
        logger.error(f"Closing Vobiz stream due to: {reason}")
        try:
            if self._vobiz_ws:
                await self._vobiz_ws.close()
        except Exception:
            pass

    async def _full_cleanup(self):
        """Clean up all connections and tasks."""
        # Cancel tasks
        for task in [self._receive_task, self._tts_worker_task, self._hangup_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        # Stop TTS worker
        if self._tts_worker_task and not self._tts_worker_task.done():
            try:
                await self._tts_queue.put(None)
            except Exception:
                pass

        # Close OpenAI connection
        if self.openai_ws:
            try:
                await self.openai_ws.close()
            except Exception:
                pass
            self.openai_ws = None

        self._connected = False
        logger.info(f"VobizRealtimeHandler cleanup complete — callId={self.call_id}")
