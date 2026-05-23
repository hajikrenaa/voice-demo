import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""

    # OpenAI API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Vobiz Configuration
    VOBIZ_AUTH_ID = os.getenv("VOBIZ_AUTH_ID")
    VOBIZ_AUTH_TOKEN = os.getenv("VOBIZ_AUTH_TOKEN")
    VOBIZ_PHONE_NUMBER = os.getenv("VOBIZ_PHONE_NUMBER")

    # ElevenLabs Configuration
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Default: Rachel
    ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")

    # Login Credentials (single user)
    LOGIN_USERNAME = os.getenv("LOGIN_USERNAME", "admin")
    LOGIN_PASSWORD = os.getenv("LOGIN_PASSWORD", "admin123")

    # Server URL (public URL for Vobiz webhooks)
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
    REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")
    REALTIME_VOICE = os.getenv("REALTIME_VOICE", "coral")
    REALTIME_AUDIO_FORMAT = "pcm16"  # 24kHz, mono, 16-bit PCM

    # Vobiz Audio Settings (mulaw 8kHz — zero-conversion)
    VOBIZ_SAMPLE_RATE = 8000  # Vobiz supports audio/x-mulaw;rate=8000
    VOBIZ_AUDIO_FORMAT = "audio/x-mulaw"  # Vobiz audio content type

    # ── Human-Like Conversation Tuning ──────────────────────
    # Smart interruption handling
    INTERRUPTION_EVAL_DELAY_MS = int(os.getenv("INTERRUPTION_EVAL_DELAY_MS", "300"))
    BACKCHANNEL_MAX_DURATION_MS = int(os.getenv("BACKCHANNEL_MAX_DURATION_MS", "500"))
    GENTLE_CLEAR_DELAY_MS = int(os.getenv("GENTLE_CLEAR_DELAY_MS", "30"))

    # Response pacing
    RESPONSE_PACING_DELAY_MS = int(os.getenv("RESPONSE_PACING_DELAY_MS", "0"))

    # Echo gate — light touch, only during AI speech
    # HARD: drop silence/quiet echo always
    # SOFT: drop medium-level audio only while AI is actively speaking
    # Cooldown: brief window after AI stops (catches late echo)
    # Permissive echo gate — only drop obvious silence, let OpenAI's
    # near_field noise reduction handle echo cancellation
    ECHO_GATE_RMS_HARD = int(os.getenv("ECHO_GATE_RMS_HARD", "600"))
    ECHO_GATE_RMS_SOFT = int(os.getenv("ECHO_GATE_RMS_SOFT", "1000"))
    ECHO_COOLDOWN_S = float(os.getenv("ECHO_COOLDOWN_S", "0.15"))  # was 0.3s

    # VAD tuning for Realtime API — aggressive settings for minimum latency
    VAD_TYPE = os.getenv("VAD_TYPE", "semantic_vad")
    VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.4"))       # Lower = triggers sooner
    VAD_PREFIX_PADDING_MS = int(os.getenv("VAD_PREFIX_PADDING_MS", "50"))    # was 150ms
    VAD_SILENCE_DURATION_MS = int(os.getenv("VAD_SILENCE_DURATION_MS", "100"))  # was 200ms
    SEMANTIC_VAD_EAGERNESS = os.getenv("SEMANTIC_VAD_EAGERNESS", "high")

    # System Prompt (structured for mini model — explicit priority rules)
    SYSTEM_PROMPT = """Real person on a phone call. Warm, concise, English only. 1-2 sentences max.
PRIORITY RULES:
1. Always answer the caller's questions first. Never ignore what they ask.
2. If caller says audio is bad or asks to repeat — apologize and repeat. Do NOT proceed.
3. If frustrated, acknowledge and ask if they want to continue.
4. NEVER guess or fabricate information the caller hasn't provided.
Confirm names, forget old if corrected. Ask to repeat if unclear."""

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
