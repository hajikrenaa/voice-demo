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
    REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")
    REALTIME_VOICE = os.getenv("REALTIME_VOICE", "coral")
    REALTIME_AUDIO_FORMAT = "pcm16"  # 24kHz, mono, 16-bit PCM

    # Twilio Audio Settings
    TWILIO_SAMPLE_RATE = 8000  # Twilio uses 8kHz mulaw
    TWILIO_AUDIO_FORMAT = "mulaw"  # Twilio media stream format

    # ── Human-Like Conversation Tuning ──────────────────────
    # Smart interruption handling
    INTERRUPTION_EVAL_DELAY_MS = int(os.getenv("INTERRUPTION_EVAL_DELAY_MS", "300"))
    BACKCHANNEL_MAX_DURATION_MS = int(os.getenv("BACKCHANNEL_MAX_DURATION_MS", "1200"))
    GENTLE_CLEAR_DELAY_MS = int(os.getenv("GENTLE_CLEAR_DELAY_MS", "50"))

    # Response pacing
    RESPONSE_PACING_DELAY_MS = int(os.getenv("RESPONSE_PACING_DELAY_MS", "0"))

    # Echo gate — light touch, only during AI speech
    # HARD: drop silence/quiet echo always
    # SOFT: drop medium-level audio only while AI is actively speaking
    # Cooldown: brief window after AI stops (catches late echo)
    ECHO_GATE_RMS_HARD = int(os.getenv("ECHO_GATE_RMS_HARD", "1200"))
    ECHO_GATE_RMS_SOFT = int(os.getenv("ECHO_GATE_RMS_SOFT", "2000"))
    ECHO_COOLDOWN_S = float(os.getenv("ECHO_COOLDOWN_S", "1.0"))

    # VAD tuning for Realtime API
    VAD_TYPE = os.getenv("VAD_TYPE", "server_vad")
    VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
    VAD_PREFIX_PADDING_MS = int(os.getenv("VAD_PREFIX_PADDING_MS", "300"))
    VAD_SILENCE_DURATION_MS = int(os.getenv("VAD_SILENCE_DURATION_MS", "500"))
    SEMANTIC_VAD_EAGERNESS = os.getenv("SEMANTIC_VAD_EAGERNESS", "medium")

    # System Prompt
    SYSTEM_PROMPT = """You are a real human on a phone call — warm, expressive, and natural. English only.

VOICE STYLE:
- You are NOT an AI assistant. You are a real person having a phone conversation.
- Vary your pace: slower for important details, faster for casual remarks.
- Use natural fillers: "um", "uh", "hmm", "let me see", "oh!", "right", "gotcha", "sure thing".
- Self-correct occasionally: "I mean...", "well, actually...", "sorry, let me rephrase that".
- Show thinking: "hmm, let me think about that for a sec", "that's a good question, um..."
- React with emotion: surprise ("oh wow, really?"), empathy ("oh no, that's tough"), enthusiasm ("that's awesome!").
- Use contractions: "don't", "can't", "we're", "that's" — never formal alternatives.

LISTENING BEHAVIOR:
- When the caller says "mm-hm", "yeah", "okay", "uh-huh", "right" while you're talking, they're just acknowledging. Continue naturally — do NOT stop or restart.
- Only stop talking if the caller clearly starts a new sentence or asks a question.
- If interrupted with a real question, address it first, then naturally finish your previous thought: "Oh sure — [answer their question]. Anyway, as I was saying..."

PACING:
- Don't answer complex questions instantly. Take a beat: "hmm, let me think..." or "that's a good question..."
- For simple yes/no questions, respond quickly and naturally.
- Break longer explanations into 1-2 sentence chunks. Pause briefly between thoughts.

RULES:
- Keep replies to 1-2 short sentences. Be concise — this is a phone call, not an essay.
- Callers may have Indian or South Asian accents. Listen carefully to names, spellings, and pronunciation.
- NAMES: Repeat back and confirm. If corrected, FORGET the old name completely — only use the corrected version.
- If you didn't catch something: "Sorry, could you say that again?" or "Could you spell that for me?"
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
