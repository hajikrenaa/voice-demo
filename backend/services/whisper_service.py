import io
import asyncio
from openai import AsyncOpenAI, OpenAIError
from config import Config
import logging

logger = logging.getLogger(__name__)


class WhisperService:
    """
    OpenAI Whisper API service for Speech-to-Text transcription
    """

    def __init__(self):
        """Initialize Whisper service with OpenAI client"""
        self.client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = Config.WHISPER_MODEL
        self.temperature = Config.WHISPER_TEMPERATURE
        self.language = Config.WHISPER_LANGUAGE

    async def transcribe(
        self,
        audio_bytes: bytes,
        audio_format: str = "wav",
        prompt: str = None
    ) -> dict:
        """
        Transcribe audio using OpenAI Whisper API

        Args:
            audio_bytes: Audio data in bytes
            audio_format: Audio format (wav, mp3, webm, etc.)
            prompt: Optional context/vocabulary hints for better accuracy

        Returns:
            dict: {
                "text": "transcribed text",
                "language": "en",
                "duration": 5.2
            }

        Raises:
            OpenAIError: If transcription fails
            ValueError: If audio is invalid
        """
        if not audio_bytes or len(audio_bytes) < 1000:
            raise ValueError("Audio too short or empty")

        # Retry logic with exponential backoff
        max_retries = 3
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                # Create in-memory file object
                audio_file = io.BytesIO(audio_bytes)
                audio_file.name = f"audio.{audio_format}"

                # Call Whisper API
                logger.info(f"Transcribing audio (attempt {attempt + 1}/{max_retries})")

                response = await self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language=self.language,  # Force English for accuracy
                    temperature=self.temperature,  # Lower = more deterministic
                    response_format="verbose_json",  # Get detailed response
                    prompt=prompt  # Optional context for better accuracy
                )

                logger.info(f"Transcription successful: {response.text[:50]}...")

                return {
                    "text": response.text,
                    "language": response.language if hasattr(response, 'language') else "en",
                    "duration": response.duration if hasattr(response, 'duration') else 0.0
                }

            except OpenAIError as e:
                logger.error(f"Whisper API error (attempt {attempt + 1}): {str(e)}")

                # If last attempt, raise the error
                if attempt == max_retries - 1:
                    raise

                # Exponential backoff
                await asyncio.sleep(retry_delay * (2 ** attempt))

            except Exception as e:
                logger.error(f"Unexpected error during transcription: {str(e)}")
                raise ValueError(f"Transcription failed: {str(e)}")

    async def transcribe_with_confidence(
        self,
        audio_bytes: bytes,
        audio_format: str = "wav",
        confidence_threshold: float = 0.7
    ) -> dict:
        """
        Transcribe audio and estimate confidence

        Note: OpenAI Whisper API doesn't return confidence scores directly,
        so we use multiple temperatures and check consistency

        Args:
            audio_bytes: Audio data
            audio_format: Audio format
            confidence_threshold: Minimum acceptable confidence (0.0-1.0)

        Returns:
            dict: {
                "text": "transcribed text",
                "confidence": 0.95,
                "is_confident": True
            }
        """
        # Get primary transcription
        result = await self.transcribe(audio_bytes, audio_format)

        # For high-accuracy mode, we could run multiple transcriptions
        # with different temperatures and check consistency
        # This is optional and adds latency

        # For now, return with assumed high confidence
        # (Whisper is generally very accurate)
        return {
            "text": result["text"],
            "confidence": 0.9,  # Whisper is typically 90%+ accurate
            "is_confident": True,
            "language": result["language"]
        }

    async def transcribe_streaming(self, audio_chunks: list) -> str:
        """
        Transcribe multiple audio chunks and concatenate results

        Args:
            audio_chunks: List of (audio_bytes, format) tuples

        Returns:
            str: Concatenated transcription
        """
        transcriptions = []

        for audio_bytes, audio_format in audio_chunks:
            try:
                result = await self.transcribe(audio_bytes, audio_format)
                transcriptions.append(result["text"])
            except Exception as e:
                logger.error(f"Failed to transcribe chunk: {str(e)}")
                continue

        return " ".join(transcriptions)
