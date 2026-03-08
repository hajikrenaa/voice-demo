from openai import AsyncOpenAI, OpenAIError
from config import Config
import logging
import asyncio

logger = logging.getLogger(__name__)


class TTSService:
    """
    OpenAI Text-to-Speech service
    """

    def __init__(self, voice: str = None):
        """
        Initialize TTS service

        Args:
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
                  Defaults to Config.TTS_VOICE
        """
        self.client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = Config.TTS_MODEL
        self.voice = voice or Config.TTS_VOICE
        self.speed = Config.TTS_SPEED

        # Simple cache for common phrases
        self.cache = {}

    async def synthesize(
        self,
        text: str,
        voice: str = None,
        speed: float = None
    ) -> bytes:
        """
        Convert text to speech using OpenAI TTS

        Args:
            text: Text to convert to speech
            voice: Optional voice override
            speed: Optional speed override (0.25 to 4.0)

        Returns:
            bytes: Audio data in MP3 format

        Raises:
            OpenAIError: If synthesis fails
            ValueError: If text is empty or invalid
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Use specified voice or default
        selected_voice = voice or self.voice
        selected_speed = speed or self.speed

        # Check cache for common phrases (lowercase for case-insensitive matching)
        cache_key = f"{text.lower()}_{selected_voice}_{selected_speed}"
        if cache_key in self.cache:
            logger.debug(f"Using cached TTS for: {text[:30]}...")
            return self.cache[cache_key]

        try:
            logger.info(f"Synthesizing speech for: {text[:50]}...")

            # Call OpenAI TTS API
            response = await self.client.audio.speech.create(
                model=self.model,
                voice=selected_voice,
                input=text,
                response_format="mp3",  # MP3 for good compression and quality
                speed=selected_speed
            )

            # Get audio bytes
            audio_bytes = response.content

            # Cache common short phrases (< 50 chars)
            if len(text) < 50:
                self.cache[cache_key] = audio_bytes

            logger.info(f"TTS synthesis completed ({len(audio_bytes)} bytes)")

            return audio_bytes

        except OpenAIError as e:
            logger.error(f"OpenAI TTS API error: {str(e)}")
            raise

        except Exception as e:
            logger.error(f"Unexpected error in TTS service: {str(e)}")
            raise ValueError(f"TTS synthesis failed: {str(e)}")

    async def synthesize_streaming(self, text: str, voice: str = None) -> bytes:
        """
        Synthesize text with streaming (for future API support)

        Currently OpenAI TTS API returns complete audio, but this method
        is structured for future streaming support

        Args:
            text: Text to synthesize
            voice: Optional voice override

        Returns:
            bytes: Audio data in Opus format (optimized for streaming)
        """
        selected_voice = voice or self.voice

        try:
            response = await self.client.audio.speech.create(
                model=self.model,
                voice=selected_voice,
                input=text,
                response_format="opus"  # Opus for better streaming
            )

            return response.content

        except Exception as e:
            logger.error(f"TTS streaming error: {str(e)}")
            raise

    async def synthesize_sentences(self, sentences: list, voice: str = None) -> list:
        """
        Synthesize multiple sentences in parallel

        Args:
            sentences: List of sentence strings
            voice: Optional voice override

        Returns:
            list: List of (sentence_text, audio_bytes) tuples
        """
        if not sentences:
            return []

        logger.info(f"Synthesizing {len(sentences)} sentences in parallel...")

        # Create tasks for parallel synthesis
        tasks = [
            self.synthesize(sentence, voice)
            for sentence in sentences
        ]

        # Execute in parallel
        try:
            audio_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Pair sentences with their audio
            results = []
            for sentence, audio in zip(sentences, audio_results):
                if isinstance(audio, Exception):
                    logger.error(f"Failed to synthesize sentence: {sentence}")
                    continue
                results.append((sentence, audio))

            logger.info(f"Successfully synthesized {len(results)}/{len(sentences)} sentences")

            return results

        except Exception as e:
            logger.error(f"Error in parallel synthesis: {str(e)}")
            raise

    def clear_cache(self):
        """Clear the TTS cache"""
        self.cache = {}
        logger.info("TTS cache cleared")

    def set_voice(self, voice: str):
        """
        Change the voice

        Args:
            voice: New voice (alloy, echo, fable, onyx, nova, shimmer)
        """
        valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        if voice not in valid_voices:
            raise ValueError(f"Invalid voice. Must be one of: {', '.join(valid_voices)}")

        self.voice = voice
        logger.info(f"Voice changed to: {voice}")

    def set_speed(self, speed: float):
        """
        Change the speech speed

        Args:
            speed: Speed multiplier (0.25 to 4.0)
        """
        if not 0.25 <= speed <= 4.0:
            raise ValueError("Speed must be between 0.25 and 4.0")

        self.speed = speed
        logger.info(f"Speed changed to: {speed}")

    def get_available_voices(self) -> list:
        """
        Get list of available voices

        Returns:
            list: Available voice names
        """
        return ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
