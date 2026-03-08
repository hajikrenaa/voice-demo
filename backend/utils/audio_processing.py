import io
import base64
from pydub import AudioSegment
import numpy as np
from config import Config


def convert_webm_to_wav(audio_bytes: bytes) -> bytes:
    """
    Convert WebM audio to WAV format at 16kHz mono

    Args:
        audio_bytes: Raw audio bytes in WebM format

    Returns:
        bytes: WAV audio data at 16kHz, mono, 16-bit PCM
    """
    try:
        # Load audio from bytes
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")

        # Convert to mono
        if audio.channels > 1:
            audio = audio.set_channels(1)

        # Resample to 16kHz (Whisper optimal)
        if audio.frame_rate != Config.SAMPLE_RATE:
            audio = audio.set_frame_rate(Config.SAMPLE_RATE)

        # Convert to 16-bit PCM
        audio = audio.set_sample_width(2)  # 2 bytes = 16 bits

        # Export to WAV format
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        return wav_io.read()

    except Exception as e:
        raise ValueError(f"Failed to convert WebM to WAV: {str(e)}")


def convert_to_wav(audio_bytes: bytes, source_format: str = "webm") -> bytes:
    """
    Convert audio from any format to WAV format at 16kHz mono

    Args:
        audio_bytes: Raw audio bytes
        source_format: Source audio format (webm, mp3, wav, etc.)

    Returns:
        bytes: WAV audio data at 16kHz, mono, 16-bit PCM
    """
    try:
        # Load audio from bytes
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=source_format)

        # Convert to mono
        if audio.channels > 1:
            audio = audio.set_channels(1)

        # Resample to 16kHz
        if audio.frame_rate != Config.SAMPLE_RATE:
            audio = audio.set_frame_rate(Config.SAMPLE_RATE)

        # Convert to 16-bit PCM
        audio = audio.set_sample_width(2)

        # Export to WAV format
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        return wav_io.read()

    except Exception as e:
        raise ValueError(f"Failed to convert {source_format} to WAV: {str(e)}")


def normalize_audio(audio_bytes: bytes) -> bytes:
    """
    Normalize audio volume levels

    Args:
        audio_bytes: WAV audio bytes

    Returns:
        bytes: Normalized WAV audio
    """
    try:
        audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))

        # Normalize to -20 dBFS (good level for speech)
        change_in_dBFS = -20.0 - audio.dBFS
        normalized = audio.apply_gain(change_in_dBFS)

        # Export normalized audio
        output_io = io.BytesIO()
        normalized.export(output_io, format="wav")
        output_io.seek(0)

        return output_io.read()

    except Exception as e:
        raise ValueError(f"Failed to normalize audio: {str(e)}")


def validate_audio(audio_bytes: bytes, min_duration_ms: int = 100) -> bool:
    """
    Validate audio quality and duration

    Args:
        audio_bytes: Audio bytes to validate
        min_duration_ms: Minimum duration in milliseconds

    Returns:
        bool: True if audio is valid, False otherwise
    """
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))

        # Check duration
        if len(audio) < min_duration_ms:
            return False

        # Check if audio is not silent (dBFS > -60)
        if audio.dBFS < -60:
            return False

        return True

    except Exception:
        return False


def get_audio_duration(audio_bytes: bytes) -> float:
    """
    Get audio duration in seconds

    Args:
        audio_bytes: Audio bytes

    Returns:
        float: Duration in seconds
    """
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        return len(audio) / 1000.0  # Convert ms to seconds

    except Exception as e:
        raise ValueError(f"Failed to get audio duration: {str(e)}")


def decode_base64_audio(base64_audio: str) -> bytes:
    """
    Decode Base64-encoded audio

    Args:
        base64_audio: Base64 encoded audio string

    Returns:
        bytes: Decoded audio bytes
    """
    try:
        return base64.b64decode(base64_audio)
    except Exception as e:
        raise ValueError(f"Failed to decode Base64 audio: {str(e)}")


def encode_audio_to_base64(audio_bytes: bytes) -> str:
    """
    Encode audio bytes to Base64 string

    Args:
        audio_bytes: Audio bytes

    Returns:
        str: Base64 encoded audio
    """
    try:
        return base64.b64encode(audio_bytes).decode('utf-8')
    except Exception as e:
        raise ValueError(f"Failed to encode audio to Base64: {str(e)}")


def preprocess_audio_for_whisper(audio_bytes: bytes, source_format: str = "webm") -> bytes:
    """
    Complete preprocessing pipeline for Whisper API

    Args:
        audio_bytes: Raw audio bytes
        source_format: Source format (webm, mp3, etc.)

    Returns:
        bytes: Preprocessed WAV audio ready for Whisper
    """
    # Convert to WAV 16kHz mono
    wav_audio = convert_to_wav(audio_bytes, source_format)

    # Normalize volume
    normalized_audio = normalize_audio(wav_audio)

    # Validate
    if not validate_audio(normalized_audio):
        raise ValueError("Audio validation failed: too short or too quiet")

    return normalized_audio
