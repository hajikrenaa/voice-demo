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

    def __init__(self, use_elevenlabs: bool = False):
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
        # Ordered TTS queue — sentences synthesized & sent one at a time
        self._tts_queue: asyncio.Queue = asyncio.Queue()
        self._tts_worker_task = None

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
            self.use_elevenlabs = elevenlabs_param.lower() == "true"
            if self.use_elevenlabs:
                self._elevenlabs_tts = ElevenLabsTTSService()
                self._elevenlabs_available = True
                # Start TTS worker for ordered sentence processing
                self._tts_worker_task = asyncio.create_task(self._tts_worker())

            mode = "ElevenLabs" if self.use_elevenlabs else "built-in"
            logger.info(f"Stream started - SID: {self.stream_sid}, Call: {self.call_sid}, Voice: {mode}")
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

        except Exception as e:
            logger.error(f"Failed to connect to OpenAI Realtime: {e}")
            self._connected = False

    async def _configure_session(self):
        """Configure session — g711_ulaw format, server VAD."""
        if self.use_elevenlabs:
            modalities = ["text"]
        else:
            modalities = ["text", "audio"]

        session_config = {
            "type": "session.update",
            "session": {
                "modalities": modalities,
                "instructions": Config.SYSTEM_PROMPT,
                "voice": Config.REALTIME_VOICE,
                "input_audio_format": "g711_ulaw",
                "output_audio_format": "g711_ulaw",
                "input_audio_transcription": {
                    "model": "whisper-1",
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 400,
                },
            },
        }
        await self.openai_ws.send(json.dumps(session_config))
        mode = "ElevenLabs TTS" if self.use_elevenlabs else "built-in voice"
        logger.info(f"Session configured — {mode}, g711_ulaw, server VAD")

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
            pass

        elif t == "response.audio_transcript.done":
            transcript = event.get("transcript", "")
            if transcript:
                logger.info(f"AI said: {transcript[:80]}...")

        elif t == "response.text.delta":
            # ElevenLabs mode: accumulate text, flush on sentence boundary
            if self.use_elevenlabs:
                self._response_text_buffer += event.get("delta", "")
                self._flush_sentence()

        elif t == "response.text.done":
            # Flush any remaining text in the buffer
            if self.use_elevenlabs and self._response_text_buffer.strip():
                text = self._response_text_buffer.strip()
                self._response_text_buffer = ""
                logger.info(f"[flush-final] {text[:60]}")
                self._tts_queue.put_nowait(text)

        elif t == "response.done":
            logger.debug("Response complete")

        elif t == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            if transcript:
                logger.info(f"User said: {transcript[:80]}...")

        elif t == "input_audio_buffer.speech_started":
            logger.debug("User speaking — clearing Twilio queue")
            if self._twilio_ws and self.stream_sid:
                try:
                    await self._twilio_ws.send_json({
                        "event": "clear",
                        "streamSid": self.stream_sid,
                    })
                except Exception:
                    pass

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
