import asyncio
import audioop
import base64
import io
import json
import logging
import time
import wave

import httpx
import websockets

from config import Config

logger = logging.getLogger(__name__)


class SarvamTTSService:
    """
    Sarvam AI Text-to-Speech service (bulbul) — the 3rd voice option.

    TTS-only: STT and the LLM stay on OpenAI Realtime (text-output mode), same
    as the ElevenLabs path.

    Primary transport is Sarvam's WebSocket streaming API with native
    mulaw/8000 output (zero conversion for Vobiz). Measured on 2026-07-09:
    ~300-420ms to first audio chunk, while the REST endpoint on the same
    account swung 0.6s-14s+ with timeouts under load — REST is kept only as a
    per-utterance fallback (tight timeout), and the call bridge falls back to
    OpenAI TTS if both fail.

    One persistent WS connection per call (this service is per-call). Sarvam
    drops idle connections after ~60s, so a keepalive ping runs while
    connected. synthesize_stream() yields mulaw chunks as they arrive so the
    caller hears the first chunk while the rest is still synthesizing.
    """

    REST_URL = "https://api.sarvam.ai/text-to-speech"
    WS_URL = "wss://api.sarvam.ai/text-to-speech/ws"

    # Sarvam target_language_code is BCP-47; map the app's per-call language.
    LANGUAGE_CODES = {"en": "en-IN", "ta": "ta-IN"}

    # REST fallback must not stall a live call the way the default 30s client
    # timeout would; if Sarvam is this slow the bridge's OpenAI fallback is better.
    REST_TIMEOUT_S = 8.0
    WS_CONNECT_TIMEOUT_S = 5.0
    WS_CHUNK_TIMEOUT_S = 10.0
    # Sarvam idle-drops at ~60s. 30s + the last-activity skip gave a worst-case
    # ~60s ping gap — a long caller monologue could silently kill the socket and
    # cost ~1s reconnect on the next reply. 20s keeps worst-case under 40s.
    KEEPALIVE_INTERVAL_S = 20.0

    def __init__(self, language: str = "en"):
        self.api_key = Config.SARVAM_API_KEY
        self.language = language
        self.speaker = (
            Config.SARVAM_SPEAKER_TA if language == "ta" else Config.SARVAM_SPEAKER
        )
        self.model = Config.SARVAM_TTS_MODEL
        self.target_language_code = self.LANGUAGE_CODES.get(language, "en-IN")
        self._client: httpx.AsyncClient | None = None
        self._ws = None
        self._connect_lock = asyncio.Lock()
        self._keepalive_task: asyncio.Task | None = None
        self._prewarm_task: asyncio.Task | None = None
        self._last_activity = 0.0
        # Set by close(); blocks any in-flight/late connect from resurrecting
        # the connection (a prewarm completing after hangup leaked a WS + an
        # immortal keepalive task per call).
        self._closed = False

    # ── WebSocket streaming (primary) ───────────────────────────────────────

    async def _ensure_ws(self):
        """Return the live streaming connection, opening one if needed.

        Locked so a background pre-warm and the first utterance can't open two
        connections at once.
        """
        if self._closed:
            raise RuntimeError("Sarvam TTS service is closed")
        async with self._connect_lock:
            if self._closed:
                raise RuntimeError("Sarvam TTS service is closed")
            if self._ws is None:
                await self._connect_ws()
            return self._ws

    def prewarm(self) -> asyncio.Task:
        """Open the streaming connection in the background.

        Called when a call starts, so the WS+config handshake (~0.7-1s) overlaps
        the OpenAI session setup instead of delaying the greeting's first audio.
        The task is tracked so close() can cancel a handshake still in flight.
        """
        async def _connect():
            try:
                await self._ensure_ws()
            except Exception as e:
                logger.debug(f"Sarvam WS pre-connect failed (will retry on use): {e}")

        self._prewarm_task = asyncio.create_task(_connect())
        return self._prewarm_task

    async def _connect_ws(self):
        """Open + configure a streaming connection (mulaw 8kHz output)."""
        url = f"{self.WS_URL}?model={self.model}&send_completion_event=true"
        ws = await asyncio.wait_for(
            websockets.connect(
                url, additional_headers={"Api-Subscription-Key": self.api_key}
            ),
            timeout=self.WS_CONNECT_TIMEOUT_S,
        )
        if self._closed:
            # close() ran while the handshake was in flight — don't resurrect.
            await ws.close()
            raise RuntimeError("Sarvam TTS service is closed")
        config_data = {
            "target_language_code": self.target_language_code,
            "speaker": self.speaker,
            "model": self.model,
            "speech_sample_rate": "8000",
            "output_audio_codec": "mulaw",
            "min_buffer_size": Config.SARVAM_MIN_BUFFER_SIZE,
            "max_chunk_length": Config.SARVAM_MAX_CHUNK_LENGTH,
            "pace": Config.SARVAM_TTS_PACE,
        }
        # temperature is a bulbul:v3-only knob (v2 uses pitch/loudness instead).
        if self.model.startswith("bulbul:v3"):
            config_data["temperature"] = Config.SARVAM_TTS_TEMPERATURE
        await ws.send(json.dumps({"type": "config", "data": config_data}))
        self._ws = ws
        self._last_activity = time.monotonic()
        if self._keepalive_task is None or self._keepalive_task.done():
            self._keepalive_task = asyncio.create_task(self._keepalive())
        logger.info(
            f"Sarvam WS connected ({self.model}/{self.speaker}, "
            f"{self.target_language_code})"
        )
        return ws

    async def _keepalive(self):
        """Sarvam closes idle streaming connections (~60s); ping while open."""
        try:
            while not self._closed and self._ws is not None:
                await asyncio.sleep(self.KEEPALIVE_INTERVAL_S)
                ws = self._ws
                if ws is None:
                    return
                if time.monotonic() - self._last_activity < self.KEEPALIVE_INTERVAL_S:
                    continue
                try:
                    await ws.send(json.dumps({"type": "ping"}))
                    self._last_activity = time.monotonic()
                except Exception:
                    # Ping failed — the connection is dead. Drop it now so the
                    # next utterance reconnects immediately instead of paying a
                    # failed send + retry first. If a reconnect already swapped
                    # in a fresh ws, keep looping and adopt it.
                    if self._ws is ws:
                        self._ws = None
                        try:
                            await ws.close()
                        except Exception:
                            pass
                    continue
        except asyncio.CancelledError:
            pass

    async def _discard_ws(self):
        """Drop the streaming connection (it reopens on next use).

        close() is time-bounded: an unbounded close handshake inside a
        generator's cleanup wedged the TTS worker silently on a live call
        (2026-07-11 — agent went mute for 40s with zero errors logged).
        """
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await asyncio.wait_for(ws.close(), timeout=3.0)
            except Exception:
                pass

    async def synthesize_stream(self, text: str):
        """Yield mulaw 8kHz chunks for `text` as Sarvam produces them.

        Retries once on a dead/stale connection *before* any audio is yielded
        (a mid-utterance retry would duplicate speech). If the stream is
        abandoned mid-utterance (caller barge-in) or dies after first audio,
        the connection is discarded so leftover chunks of this utterance can
        never bleed into the next one, and a background reconnect starts so
        the NEXT reply doesn't pay the ~1s handshake.

        Deliberately a SINGLE generator: this used to delegate to a nested
        `_stream_utterance` generator, and closing the outer one at a yield
        left the inner one to a GC-scheduled aclose that raced live use
        ("aclose(): asynchronous generator is already running" — the silent
        TTS-worker wedge on the 2026-07-11 English call).
        """
        if not text or not text.strip():
            return

        yielded = False
        completed = False
        try:
            for attempt in (0, 1):
                # A fresh-connect failure propagates immediately (retrying the
                # same connect back-to-back rarely helps and just adds dead air
                # before the REST/OpenAI fallbacks). The retry below exists for
                # a STALE kept-alive connection dying on first use.
                ws = await self._ensure_ws()
                try:
                    await ws.send(json.dumps({"type": "text", "data": {"text": text}}))
                    await ws.send(json.dumps({"type": "flush", "data": {}}))
                    self._last_activity = time.monotonic()
                    while True:
                        raw = await asyncio.wait_for(
                            ws.recv(), timeout=self.WS_CHUNK_TIMEOUT_S
                        )
                        self._last_activity = time.monotonic()
                        msg = json.loads(raw)
                        mtype = msg.get("type")
                        if mtype == "audio":
                            chunk = base64.b64decode(msg["data"]["audio"])
                            if chunk:
                                yielded = True
                                yield chunk
                        elif mtype == "event":
                            if msg.get("data", {}).get("event_type") == "final":
                                completed = True
                                return
                        elif mtype == "error":
                            raise RuntimeError(
                                f"Sarvam WS error: {str(msg.get('data'))[:200]}"
                            )
                except Exception as e:
                    # GeneratorExit is BaseException — a barge-in close skips
                    # this handler and unwinds straight to the finally below.
                    await self._discard_ws()
                    if yielded or attempt == 1:
                        raise
                    logger.info(f"Sarvam WS retry after: {type(e).__name__}: {e}")
        finally:
            if not completed:
                await self._discard_ws()
                if not self._closed:
                    # Barge-ins land here — reconnect in the background so the
                    # caller's next reply gets a warm socket (a live call paid
                    # a 2.3s first-chunk on the cold reconnect).
                    self.prewarm()

    async def synthesize(self, text: str) -> bytes:
        """Convert text to speech; returns complete mulaw 8kHz bytes.

        Tries the WS streaming path first, then one REST attempt. Raising on
        total failure lets the call bridge fall back to OpenAI TTS.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            audio = b""
            async for chunk in self.synthesize_stream(text):
                audio += chunk
            if audio:
                logger.info(
                    f"Sarvam TTS (ws): {len(audio)} bytes "
                    f"(~{len(audio) / 8000.0:.1f}s) for: {text[:50]}..."
                )
                return audio
            raise RuntimeError("Sarvam WS returned no audio")
        except Exception as e:
            logger.warning(f"Sarvam WS failed ({e}); trying REST fallback")

        return await self.synthesize_rest(text)

    # ── REST fallback ───────────────────────────────────────────────────────

    def _get_client(self) -> httpx.AsyncClient:
        """Lazily create and reuse a shared httpx client for connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.REST_TIMEOUT_S)
        return self._client

    @staticmethod
    def _wav_to_ulaw8k(wav_bytes: bytes) -> bytes:
        """Decode a WAV container to mulaw 8kHz mono bytes for Vobiz.

        Sarvam returns PCM16 mono at the requested 8kHz, but each property is
        still verified (and converted if a future API change drifts) so a format
        change degrades to a resample instead of garbled audio on a live call.
        """
        with wave.open(io.BytesIO(wav_bytes)) as w:
            channels = w.getnchannels()
            sampwidth = w.getsampwidth()
            rate = w.getframerate()
            pcm = w.readframes(w.getnframes())

        if channels == 2:
            pcm = audioop.tomono(pcm, sampwidth, 0.5, 0.5)
        if sampwidth != 2:
            pcm = audioop.lin2lin(pcm, sampwidth, 2)
        if rate != 8000:
            pcm, _ = audioop.ratecv(pcm, 2, 1, rate, 8000, None)
        return audioop.lin2ulaw(pcm, 2)

    async def synthesize_rest(self, text: str) -> bytes:
        """One-shot REST synthesis; returns mulaw 8kHz bytes."""
        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "target_language_code": self.target_language_code,
            "speaker": self.speaker,
            "model": self.model,
            "speech_sample_rate": 8000,
            "enable_preprocessing": True,
        }

        try:
            client = self._get_client()
            response = await client.post(self.REST_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            audios = data.get("audios") or []
            if not audios:
                raise ValueError(f"Sarvam TTS returned no audio: {str(data)[:200]}")

            wav_bytes = base64.b64decode(audios[0])
            ulaw = self._wav_to_ulaw8k(wav_bytes)
            logger.info(
                f"Sarvam TTS (rest): {len(ulaw)} bytes "
                f"(~{len(ulaw) / 8000.0:.1f}s) for: {text[:50]}..."
            )
            return ulaw

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Sarvam API error: {e.response.status_code} {e.response.text[:200]}"
            )
            raise
        except Exception as e:
            logger.error(f"Sarvam TTS error: {e}")
            raise

    async def close(self):
        """Close the WS connection, keepalive/prewarm tasks and HTTP client."""
        self._closed = True
        for task in (self._prewarm_task, self._keepalive_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._prewarm_task = None
        self._keepalive_task = None
        await self._discard_ws()
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
