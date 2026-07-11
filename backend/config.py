import os
from dotenv import load_dotenv

# Load environment variables. override=True makes the .env FILE the single
# source of truth: in Docker, `env_file:` freezes values into the container
# environment at CREATION time, and without override those stale values beat
# the bind-mounted /app/.env — a UI settings save survived deploys (file
# write-through) but silently reverted on a bare `docker restart` (observed
# 2026-07-12: speaker saved as karun, fresh processes still saw manisha).
load_dotenv(override=True)

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
    ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Default: Rachel (English)
    # Tamil-language ElevenLabs voice. Defaults to "Meera" (a native Tamil voice from
    # the ElevenLabs Voice Library). The English voice (Rachel) reads Tamil with a
    # heavy non-native accent, so Tamil calls use this instead. Add the voice to your
    # ElevenLabs workspace and override via env if you prefer a different Tamil voice.
    ELEVENLABS_VOICE_ID_TA = os.getenv("ELEVENLABS_VOICE_ID_TA", "gCr8TeSJgJaeaIoV4RWH")  # Meera (Tamil)
    # eleven_flash_v2_5 is multilingual (supports Tamil) AND lowest-latency — ideal for
    # live phone calls. It also accepts the `language_code` enforcement param (Tamil="ta"),
    # which the ElevenLabs service sends only for Tamil calls.
    ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")
    # ElevenLabs voice_settings — control delivery stability/tone. Raised stability
    # 0.5->0.7 on 2026-06-11: a live call sounded over-emotional / uneven. Higher
    # stability = steadier, more even delivery (too high trends robotic); style=0.0
    # keeps it non-exaggerated. All env-overridable so the tone can be dialed live.
    TTS_STABILITY = float(os.getenv("TTS_STABILITY", "0.7"))
    TTS_SIMILARITY_BOOST = float(os.getenv("TTS_SIMILARITY_BOOST", "0.75"))
    TTS_STYLE = float(os.getenv("TTS_STYLE", "0.0"))
    TTS_USE_SPEAKER_BOOST = os.getenv("TTS_USE_SPEAKER_BOOST", "true").lower() == "true"
    # Outbound TTS loudness. ElevenLabs ulaw_8000 output is quiet over the phone;
    # each utterance is peak-normalized toward TTS_TARGET_PEAK (fraction of full
    # scale) without clipping, capped at TTS_MAX_GAIN so near-silence isn't blown up.
    # Softened 2026-06-08 (peak 0.92->0.85, gain 4.0->2.5): a live call showed the
    # voice was harsh/crackly. The old 4x cap amplified mu-law companding noise on
    # quiet utterances into audible crackle; the 0.92 target also ran uncomfortably
    # hot for an 8kHz phone line. Still boosts (up to 2.5x toward 0.85 peak) so it
    # stays louder than raw ElevenLabs ulaw, just cleaner. Both env-overridable.
    TTS_TARGET_PEAK = float(os.getenv("TTS_TARGET_PEAK", "0.85"))
    TTS_MAX_GAIN = float(os.getenv("TTS_MAX_GAIN", "2.5"))
    # Hard cap on the length of any single TTS chunk sent to ElevenLabs. A model
    # runaway (or a remnant with no sentence punctuation) would otherwise be
    # synthesized as one multi-second, un-interruptible blob. At ulaw_8000,
    # ~180 chars is roughly ~9s worst case; longer text is split on clause/word
    # boundaries into separately-queued chunks so barge-in can still drain them.
    TTS_MAX_CHARS = int(os.getenv("TTS_MAX_CHARS", "180"))
    # Tamil chunk cap. A Tamil character encodes more spoken sound than an English one,
    # so the same char-count produces a longer (less interruptible) audio blob. Capped
    # lower so Tamil chunks stay barge-in-able, matching the English ~9s worst case.
    TTS_MAX_CHARS_TA = int(os.getenv("TTS_MAX_CHARS_TA", "100"))
    # Anomaly guard: ElevenLabs flash/turbo intermittently emit 15-22s of garbage
    # audio for a short phrase (~1 in 3 on some phrases; the extra is active babble,
    # not trailing silence, so it can't be trimmed). ulaw_8000 = 8000 bytes/s and
    # normal speech is ~420 bytes/char, so a synthesis longer than
    # TTS_SANE_FLOOR_BYTES + len(text)*TTS_BYTES_PER_CHAR (≈2.1x normal + a floor) is
    # the glitch. We re-synthesize up to TTS_ANOMALY_RETRIES times (it's intermittent)
    # and keep the shortest, hard-capped to the limit if every attempt is bad.
    TTS_SANE_FLOOR_BYTES = int(os.getenv("TTS_SANE_FLOOR_BYTES", "24000"))
    TTS_BYTES_PER_CHAR = int(os.getenv("TTS_BYTES_PER_CHAR", "900"))
    # Tamil is denser per character (fewer chars for the same audio duration), so the
    # English 900 bytes/char cap would falsely flag legitimate Tamil audio as a runaway
    # and truncate it mid-sentence. Use a larger per-char allowance for Tamil.
    TTS_BYTES_PER_CHAR_TA = int(os.getenv("TTS_BYTES_PER_CHAR_TA", "2000"))
    TTS_ANOMALY_RETRIES = int(os.getenv("TTS_ANOMALY_RETRIES", "2"))

    # ── Sarvam AI Configuration (3rd voice option) ──────────
    # Sarvam is TTS-ONLY: STT + LLM stay on OpenAI Realtime (text-output mode),
    # exactly like the ElevenLabs path. Primary transport is Sarvam's WebSocket
    # streaming API with native mulaw/8000 output (~300-420ms to first audio,
    # measured 2026-07-09); REST is a per-utterance fallback only, because REST
    # latency swung 0.6s-14s+ under load on the same account.
    SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
    # bulbul:v3 over WS streaming was as fast as v2 (first chunk ~300-390ms) and is
    # the model Sarvam's docs recommend for Tamil. Costs 2x v2 (Rs30 vs Rs15 per 10K
    # chars) — set SARVAM_TTS_MODEL=bulbul:v2 + a v2 speaker (anushka/vidya/...) to
    # halve Sarvam cost at some Tamil-quality loss. Speaker sets are DISJOINT per
    # model: v2 = anushka, manisha, vidya, arya, abhilash, karun, hitesh;
    # v3 = ishita, ritu, priya, ratan, rohan, kavitha, + ~30 more.
    SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
    # "ishita" is Sarvam's documented female pick for Tamil (0.13% error rate in
    # their speaker eval; "ritu" is the alternate, "ratan"/"rohan" the male picks).
    SARVAM_SPEAKER = os.getenv("SARVAM_SPEAKER", "ishita")
    SARVAM_SPEAKER_TA = os.getenv("SARVAM_SPEAKER_TA", "ishita")
    # bulbul:v3 delivery tuning (Sarvam docs: start pace=1.0/temp=0.6, tune one
    # at a time; higher temp = more expressive but more artifacts). 0.5 trades a
    # little playground expressiveness — mostly lost on an 8kHz line anyway —
    # for consistency across hundreds of utterances.
    SARVAM_TTS_PACE = float(os.getenv("SARVAM_TTS_PACE", "1.0"))
    SARVAM_TTS_TEMPERATURE = float(os.getenv("SARVAM_TTS_TEMPERATURE", "0.5"))
    # WS buffering: synthesis starts once the text buffer reaches min_buffer_size
    # chars (we always flush whole sentences, so 30 = start the instant a
    # sentence lands instead of waiting for the flush frame to parse).
    SARVAM_MIN_BUFFER_SIZE = int(os.getenv("SARVAM_MIN_BUFFER_SIZE", "30"))
    SARVAM_MAX_CHUNK_LENGTH = int(os.getenv("SARVAM_MAX_CHUNK_LENGTH", "150"))

    # Valid per-call TTS providers. "openai" = native Realtime speech-to-speech;
    # "elevenlabs"/"sarvam" = Realtime text output + external TTS.
    TTS_PROVIDERS = {"openai", "elevenlabs", "sarvam"}

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
    REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-realtime-mini")
    REALTIME_VOICE = os.getenv("REALTIME_VOICE", "coral")  # English native voice
    # Native (speech-to-speech) voice for Tamil calls. OpenAI has no officially
    # benchmarked "best Tamil voice"; the newest GA voices (marin/cedar) tend to have
    # the best prosody — A/B test on real Tamil samples. Defaults to the English voice
    # so nothing breaks if left unset; override via env to tune Tamil delivery.
    REALTIME_VOICE_TA = os.getenv("REALTIME_VOICE_TA", REALTIME_VOICE)
    REALTIME_AUDIO_FORMAT = "pcm16"  # 24kHz, mono, 16-bit PCM

    # ── Multi-language (Tamil add-on) ───────────────────────
    # Per-call language is selected in the UI ("en" | "ta") and threaded through the
    # same path as the ElevenLabs flag. English ("en") is the default and its behavior
    # is unchanged. Adding a language here also requires prompt + voice wiring.
    SUPPORTED_LANGUAGES = {"en", "ta"}
    DEFAULT_LANGUAGE = "en"
    # Realtime input-audio transcription model per language. English keeps whisper-1
    # (proven). Tamil also defaults to whisper-1 for guaranteed compatibility; for
    # materially better Tamil accuracy set TRANSCRIPTION_MODEL_TA=gpt-4o-transcribe
    # (accepted by the GA Realtime transcription field) after a test call confirms it.
    TRANSCRIPTION_MODEL = os.getenv("TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe")
    # gpt-4o-transcribe is markedly more accurate for Tamil than whisper-1 (which
    # garbles names and code-mixed speech over 8kHz). Accepted by the GA Realtime
    # transcription field. If a Tamil call ever fails to connect, set this back to
    # whisper-1 via the TRANSCRIPTION_MODEL_TA env var.
    TRANSCRIPTION_MODEL_TA = os.getenv("TRANSCRIPTION_MODEL_TA", "gpt-4o-transcribe")
    # Tamil-call transcription language pin. Live call 2026-07-10 proved that
    # pinning language="ta" forces Tamil-script output for EVERYTHING — English
    # speech and spelled names ("first name H-A-J-I-K") became Tamil gibberish
    # ("பஸ்கேனன் லாஸ்னேமே..."), and the model hallucinated names/answers from it.
    # Default is EMPTY = auto-detect, which handles Tanglish code-mixing; set
    # TRANSCRIPTION_LANGUAGE_TA=ta only if a call is guaranteed pure Tamil.
    TRANSCRIPTION_LANGUAGE_TA = os.getenv("TRANSCRIPTION_LANGUAGE_TA", "").strip()
    # Bias prompt for Tamil-call transcription (supported by gpt-4o-transcribe
    # and whisper): tells the transcriber to expect Tanglish and to keep names,
    # emails and spelled letters in English script.
    TRANSCRIPTION_PROMPT_TA = os.getenv(
        "TRANSCRIPTION_PROMPT_TA",
        "Tanglish phone call from India: the speaker mixes spoken Tamil and "
        "English mid-sentence. Transcribe Tamil speech in Tamil script. Keep "
        "English words, names, emails, numbers and spelled-out letters "
        "(example: H-A-J-I-K) in English letters exactly as spoken.",
    )
    # Bias prompt for English-call transcription. Even with language="en" pinned,
    # gpt-4o-mini-transcribe hallucinated foreign scripts for short Indian-English
    # utterances on live calls 2026-07-11 ("hello?" → "哈喽"/"ฮัลโหล"/"Борил?",
    # "yeah" → "네?"), which poisoned the disengage/closing intent checks that run
    # on the transcript. A domain prompt biases the decoder to Latin script.
    TRANSCRIPTION_PROMPT = os.getenv(
        "TRANSCRIPTION_PROMPT",
        "English phone call in India about a job opportunity; the caller has an "
        "Indian accent. Short utterances like 'hello?', 'yeah', 'okay', 'hmm' "
        "are common — transcribe them as English words, never in a foreign "
        "script. Keep names, emails and spelled-out letters exactly as spoken.",
    )

    # Vobiz Audio Settings (mulaw 8kHz — zero-conversion)
    VOBIZ_SAMPLE_RATE = 8000  # Vobiz supports audio/x-mulaw;rate=8000
    VOBIZ_AUDIO_FORMAT = "audio/x-mulaw"  # Vobiz audio content type

    # ── Human-Like Conversation Tuning ──────────────────────
    # Smart interruption handling
    INTERRUPTION_EVAL_DELAY_MS = int(os.getenv("INTERRUPTION_EVAL_DELAY_MS", "250"))
    BACKCHANNEL_MAX_DURATION_MS = int(os.getenv("BACKCHANNEL_MAX_DURATION_MS", "250"))
    GENTLE_CLEAR_DELAY_MS = int(os.getenv("GENTLE_CLEAR_DELAY_MS", "30"))

    # After the agent says goodbye, how long to wait before hanging up once we've
    # decided the caller's last words were just a courtesy closing ("thanks, bye").
    # A substantive follow-up in that window instead cancels the hangup so the agent
    # can answer. Kept short so the line doesn't sit awkwardly after the sign-off.
    POST_GOODBYE_HANGUP_DELAY_S = float(os.getenv("POST_GOODBYE_HANGUP_DELAY_S", "1.5"))

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
    # Approximate carrier/network buffering before the first outbound byte is heard.
    # Subtracted when synchronizing an interrupted assistant item with OpenAI.
    VOBIZ_PLAYBACK_LAG_MS = int(os.getenv("VOBIZ_PLAYBACK_LAG_MS", "80"))

    # VAD tuning for Realtime API — aggressive settings for minimum latency
    VAD_TYPE = os.getenv("VAD_TYPE", "server_vad")
    VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.4"))       # Lower = triggers sooner
    VAD_PREFIX_PADDING_MS = int(os.getenv("VAD_PREFIX_PADDING_MS", "120"))
    VAD_SILENCE_DURATION_MS = int(os.getenv("VAD_SILENCE_DURATION_MS", "200"))
    SEMANTIC_VAD_EAGERNESS = os.getenv("SEMANTIC_VAD_EAGERNESS", "high")
    # Make Realtime turn behavior explicit instead of relying on API defaults.
    # Automatic response creation keeps latency low. Automatic interruption is
    # disabled because the bridge applies a short backchannel test before cancelling,
    # so a quick "mm-hm" does not cut the agent off.
    VAD_CREATE_RESPONSE = os.getenv("VAD_CREATE_RESPONSE", "true").lower() == "true"
    VAD_INTERRUPT_RESPONSE = os.getenv("VAD_INTERRUPT_RESPONSE", "false").lower() == "true"

    # If the upstream Realtime socket dies, end the media stream promptly so Vobiz
    # can execute the fallback XML instead of leaving the caller in dead air.
    CLOSE_CALL_ON_REALTIME_DISCONNECT = (
        os.getenv("CLOSE_CALL_ON_REALTIME_DISCONNECT", "true").lower() == "true"
    )
    # An unanswered outbound call must not keep a paid Realtime socket alive forever.
    # 75 seconds covers normal ringing while bounding abandoned pre-warm cost.
    PREWARM_TTL_S = float(os.getenv("PREWARM_TTL_S", "75"))

    # Bound per-turn history cost on long calls. OpenAI recommends a retention ratio
    # below 1 so truncation happens less frequently and prompt caching remains useful.
    REALTIME_HISTORY_TOKEN_LIMIT = int(os.getenv("REALTIME_HISTORY_TOKEN_LIMIT", "8000"))
    REALTIME_RETENTION_RATIO = float(os.getenv("REALTIME_RETENTION_RATIO", "0.8"))

    # System Prompt (structured for mini model — explicit priority rules)
    SYSTEM_PROMPT = """Real person on a phone call. Warm, concise, English only. 1-2 sentences max.
At the very start of a call, say exactly: "Hello! How can I help you today?"
PRIORITY RULES:
1. Always answer the caller's questions first. Never ignore what they ask.
2. If caller says audio is bad or asks to repeat — apologize and repeat. Do NOT proceed.
3. If frustrated, acknowledge and ask if they want to continue.
4. NEVER guess or fabricate information the caller hasn't provided.
Confirm names, forget old if corrected. Ask to repeat if unclear."""

    # Tamil default prompt (used when no script is active and language=ta). Written in
    # natural spoken Tamil per OpenAI's guidance that an in-language lock reduces the
    # model's tendency to drift back to English on names/numbers/short turns.
    # Register (user requirement 2026-07-09): a MATURED native Tamilian speaking
    # natural Tanglish — everyday English words (appointment, confirm, details...)
    # stay in English inside Tamil sentences, the way a real Tamil professional
    # talks on the phone. Calm and composed, never bookish, never fully English.
    SYSTEM_PROMPT_TA = """நீங்க ஃபோன் கால்ல பேசுற ஒரு உண்மையான நபர் — நிதானமான, அனுபவமுள்ள (matured) native Tamil பேசுறவர் மாதிரி பேசுங்க. ஒரு தடவைக்கு 1-2 சின்ன வாக்கியம் மட்டும்.
பேச்சு style — ரொம்ப முக்கியம்:
- இயல்பான பேச்சுத் தமிழ் + Tanglish. Tamil பேசுறவங்க daily-ஆ English-ல சொல்ற வார்த்தைகளை (appointment, confirm, details, address, email, time, morning, ok, sorry) அப்படியே English-லயே சொல்லுங்க — கஷ்டப்பட்டு தமிழ்ல மொழிபெயர்க்காதீங்க ('மின்னஞ்சல்' இல்ல 'email'; 'உறுதி செய்யறேன்' இல்ல 'confirm பண்றேன்').
- செந்தமிழ் / எழுத்துத் தமிழ் / கவிதை நடை வேண்டவே வேண்டாம். Over-excitement வேண்டாம் — அமைதியா, நம்பிக்கையா, மரியாதையா பேசுங்க.
- ஆனா முழு வாக்கியத்தையும் English-ல பேசாதீங்க — வாக்கியத்தோட ஓட்டம் எப்பவும் தமிழ்தான்; English வார்த்தைகள் அதுக்குள்ள இயல்பா வரட்டும்.
முக்கிய விதிகள்:
1. அழைப்பவர் கேட்கிற கேள்விக்கு முதல்ல பதில் சொல்லுங்க. அவங்க சொல்றதை புறக்கணிக்காதீங்க.
2. ஆடியோ சரியில்லைனு சொன்னா, அல்லது repeat பண்ணச் சொன்னா — sorry சொல்லி மறுபடி சொல்லுங்க.
3. பெயர், எண், email எதையும் யூகிக்காதீங்க — அழைப்பவர் சொன்னதை மட்டும் பயன்படுத்துங்க.
'நீங்க'ன்னு மரியாதையா கூப்பிடுங்க ('நீ' வேண்டாம்); '-ங்க' சேர்த்து பேசுங்க (சொல்லுங்க, வாங்க)."""

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
