import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""

    # OpenAI API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Twilio Configuration
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
    TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

    # ElevenLabs Configuration
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Default: Rachel
    ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")

    # Login Credentials (single user)
    LOGIN_USERNAME = os.getenv("LOGIN_USERNAME", "admin")
    LOGIN_PASSWORD = os.getenv("LOGIN_PASSWORD", "admin123")

    # Server URL (public URL for Twilio webhooks)
    SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")

    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Audio Settings
    SAMPLE_RATE = 16000  # Whisper optimal sample rate (Hz)
    CHANNELS = 1  # Mono audio
    CHUNK_DURATION = 3  # Audio chunk duration in seconds

    # Conversation Settings
    MAX_CONVERSATION_DURATION = 3600  # Maximum conversation duration in seconds (1 hour)
    MAX_HISTORY_MESSAGES = 8  # Keep short history to save tokens

    # Voice Activity Detection
    VAD_SILENCE_THRESHOLD = 0.8  # Reduced for faster response (was 1.5s)
    VAD_ENERGY_THRESHOLD = 0.015  # Slightly lower for better sensitivity

    # OpenAI Model Configuration
    WHISPER_MODEL = "whisper-1"
    WHISPER_TEMPERATURE = 0.0  # Lower temperature for higher accuracy
    WHISPER_LANGUAGE = "en"

    LLM_MODEL = "gpt-4o-mini"  # Cheaper & faster for voice
    LLM_TEMPERATURE = 0.7
    LLM_MAX_TOKENS = 150  # Voice responses are short

    TTS_MODEL = "tts-1"  # or "tts-1-hd" for higher quality
    TTS_VOICE = "alloy"  # Options: alloy, echo, fable, onyx, nova, shimmer
    TTS_SPEED = 1.0  # 0.25 to 4.0

    # OpenAI Realtime API Configuration (for ultra-low latency)
    REALTIME_MODEL = "gpt-4o-realtime-preview"
    REALTIME_VOICE = "coral"  # Warm, expressive, natural — great for conversation
    REALTIME_AUDIO_FORMAT = "pcm16"  # 24kHz, mono, 16-bit PCM

    # Twilio Audio Settings
    TWILIO_SAMPLE_RATE = 8000  # Twilio uses 8kHz mulaw
    TWILIO_AUDIO_FORMAT = "mulaw"  # Twilio media stream format

    # System Prompt (keep minimal to save tokens)
    SYSTEM_PROMPT = """You are a friendly, energetic AI voice assistant on a phone call. English only.
You sound like a real human — warm, expressive, with natural emotion in your voice. React with genuine enthusiasm, empathy, or curiosity as appropriate. Use filler words occasionally like "Oh!", "Ah", "Right", "Got it" to sound natural.
Rules:
- Keep replies to 1-2 short sentences. Be quick and snappy — respond fast.
- Callers may have Indian or South Asian accents — listen carefully to names, spellings, and pronunciation.
- NAMES: Repeat back and confirm. If corrected, FORGET the old name completely — only use the corrected version forever.
- If you didn't catch something, ask: "Sorry, could you say that again?" or "Could you spell that for me?"
- Greet the caller warmly when you first connect."""

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.OPENAI_API_KEY or cls.OPENAI_API_KEY == "sk-your-openai-api-key-here":
            raise ValueError(
                "OPENAI_API_KEY not set. Please set it in the .env file."
            )

    @classmethod
    def is_development(cls):
        """Check if running in development mode"""
        return cls.ENVIRONMENT == "development"


# Validate configuration on import
if __name__ != "__main__":
    try:
        Config.validate()
    except ValueError as e:
        print(f"Configuration Error: {e}")
