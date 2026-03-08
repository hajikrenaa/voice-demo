import httpx
import asyncio
import logging
from config import Config

logger = logging.getLogger(__name__)


class ElevenLabsTTSService:
    """
    ElevenLabs Text-to-Speech service for high-quality voice synthesis.
    Returns raw audio bytes (mp3 by default) suitable for conversion to mulaw for Twilio.
    """

    API_URL = "https://api.elevenlabs.io/v1/text-to-speech"

    def __init__(self, voice_id: str = None):
        self.api_key = Config.ELEVENLABS_API_KEY
        self.voice_id = voice_id or Config.ELEVENLABS_VOICE_ID
        self.model_id = Config.ELEVENLABS_MODEL_ID

    async def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech using ElevenLabs API.

        Args:
            text: Text to convert to speech

        Returns:
            bytes: Audio data in mp3 format
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        url = f"{self.API_URL}/{self.voice_id}"

        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
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

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()

                audio_bytes = response.content
                logger.info(
                    f"ElevenLabs TTS: synthesized {len(audio_bytes)} bytes for: {text[:50]}..."
                )
                return audio_bytes

        except httpx.HTTPStatusError as e:
            logger.error(f"ElevenLabs API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"ElevenLabs TTS error: {e}")
            raise

    async def synthesize_stream(self, text: str):
        """
        Stream audio from ElevenLabs API chunk by chunk.

        Args:
            text: Text to convert to speech

        Yields:
            bytes: Audio chunks in mp3 format
        """
        if not text or not text.strip():
            return

        url = f"{self.API_URL}/{self.voice_id}/stream"

        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
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
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes(chunk_size=4096):
                        if chunk:
                            yield chunk

        except Exception as e:
            logger.error(f"ElevenLabs streaming error: {e}")
            raise
