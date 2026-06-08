import httpx
import logging
from config import Config

logger = logging.getLogger(__name__)


class ElevenLabsTTSService:
    """
    ElevenLabs Text-to-Speech service for high-quality voice synthesis.
    Returns raw audio bytes (mp3 by default) suitable for conversion to mulaw for Twilio.

    Reuses a single httpx.AsyncClient across requests for connection pooling.
    """

    API_URL = "https://api.elevenlabs.io/v1/text-to-speech"

    def __init__(self, voice_id: str = None):
        self.api_key = Config.ELEVENLABS_API_KEY
        self.voice_id = voice_id or Config.ELEVENLABS_VOICE_ID
        self.model_id = Config.ELEVENLABS_MODEL_ID
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Lazily create and reuse a shared httpx client for connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def synthesize(self, text: str) -> bytes:
        """Convert text to speech using ElevenLabs API.

        Includes an anomaly guard: flash/turbo intermittently return 15-22s of
        garbage audio for a short phrase (active babble, not trailing silence, so it
        can't be trimmed). If a synthesis is far longer than the text warrants, we
        re-synthesize (the glitch is non-deterministic) and keep the shortest result,
        hard-capping it if every attempt is bad. Tunable via Config.TTS_* knobs.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        url = f"{self.API_URL}/{self.voice_id}?output_format=ulaw_8000"

        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/basic",
        }

        payload = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True,
            },
        }

        # ulaw_8000 is 8000 bytes/s and normal speech is ~420 bytes/char; anything
        # past this limit is the intermittent flash/turbo runaway glitch.
        max_ok = Config.TTS_SANE_FLOOR_BYTES + len(text) * Config.TTS_BYTES_PER_CHAR
        attempts = Config.TTS_ANOMALY_RETRIES + 1
        best = None

        try:
            client = self._get_client()
            for attempt in range(attempts):
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                audio_bytes = response.content

                if best is None or len(audio_bytes) < len(best):
                    best = audio_bytes

                if len(audio_bytes) <= max_ok:
                    if attempt > 0:
                        logger.info(
                            f"ElevenLabs TTS: clean on retry {attempt} "
                            f"({len(audio_bytes)}b) for: {text[:40]}"
                        )
                    logger.info(
                        f"ElevenLabs TTS: synthesized {len(audio_bytes)} bytes for: {text[:50]}..."
                    )
                    return audio_bytes

                logger.warning(
                    f"ElevenLabs TTS anomaly: {len(audio_bytes)}b > {max_ok}b cap "
                    f"for {len(text)}-char text (attempt {attempt + 1}/{attempts}): {text[:40]}"
                )

            # Every attempt was anomalous — cap the damage to the shortest, truncated.
            capped = best[:max_ok]
            logger.error(
                f"ElevenLabs TTS still anomalous after {attempts} attempts; "
                f"capped {len(best)}b -> {len(capped)}b for: {text[:40]}"
            )
            return capped

        except httpx.HTTPStatusError as e:
            logger.error(f"ElevenLabs API error: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"ElevenLabs TTS error: {e}")
            raise

    async def synthesize_stream(self, text: str):
        """Stream audio from ElevenLabs API chunk by chunk."""
        if not text or not text.strip():
            return

        url = f"{self.API_URL}/{self.voice_id}/stream?output_format=ulaw_8000"

        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/basic",
        }

        payload = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }

        try:
            client = self._get_client()
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    if chunk:
                        yield chunk

        except Exception as e:
            logger.error(f"ElevenLabs streaming error: {e}")
            raise

    async def close(self):
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
