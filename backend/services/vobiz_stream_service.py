"""
Vobiz Bidirectional Media Stream — OpenAI Realtime API

Ultra-low latency voice agent bridging Vobiz ↔ OpenAI Realtime.

Vobiz WebSocket protocol:
  Incoming (Vobiz → us):
    {"event":"start",  "start":{"callId":"...","streamId":"...","mediaFormat":{...}}}
    {"event":"media",  "media":{"payload":"<base64-mulaw>","contentType":"audio/x-mulaw","sampleRate":8000}}
    {"event":"stop",   "reason":"call_ended"}

  Outgoing (us → Vobiz):
    {"event":"playAudio","media":{"contentType":"audio/x-mulaw","sampleRate":8000,"payload":"<base64-mulaw>"}}
    {"event":"clearAudio"}   — clears queued audio (interruption)

Mode 1 (tts_provider="openai" — default):
  Vobiz mulaw 8kHz ──► g711_ulaw ──► OpenAI Realtime (speech-to-speech)
  Vobiz mulaw 8kHz ◄── g711_ulaw ◄──┘
  Zero audio conversion on both input AND output. ~300ms latency.

Mode 2 (tts_provider="elevenlabs" or "sarvam" — external TTS):
  Vobiz mulaw 8kHz ──► g711_ulaw ──► OpenAI Realtime (text-only response)
                                          │ text (sentence-streamed)
  Vobiz mulaw 8kHz ◄── ulaw_8000 ◄── ElevenLabs (native ulaw) / Sarvam (WAV→ulaw)
  Vobiz mulaw 8kHz ◄── PCM->mulaw ◄── OpenAI TTS fallback
"""

import asyncio
import audioop
import base64
import json
import logging
import re
import time
from contextlib import aclosing
from typing import Optional

import websockets
from openai import AsyncOpenAI

from config import Config
from services.elevenlabs_tts_service import ElevenLabsTTSService
from services.sarvam_tts_service import SarvamTTSService
from utils.audio_processing import downsample_24k_to_8k

logger = logging.getLogger(__name__)

# Vobiz REST API base
VOBIZ_API_BASE = "https://api.vobiz.ai/api/v1"

# NATO phonetic alphabet for forced-confirmation injection.
NATO_PHONETIC = {
    'A': 'Alpha', 'B': 'Bravo', 'C': 'Charlie', 'D': 'Delta',
    'E': 'Echo', 'F': 'Foxtrot', 'G': 'Golf', 'H': 'Hotel',
    'I': 'India', 'J': 'Juliet', 'K': 'Kilo', 'L': 'Lima',
    'M': 'Mike', 'N': 'November', 'O': 'Oscar', 'P': 'Papa',
    'Q': 'Quebec', 'R': 'Romeo', 'S': 'Sierra', 'T': 'Tango',
    'U': 'Uniform', 'V': 'Victor', 'W': 'Whiskey', 'X': 'X-ray',
    'Y': 'Yankee', 'Z': 'Zulu',
}

# Phrases that mean "I want to end the call now without finishing".
DISENGAGE_PHRASES = (
    "busy now", "busy right now", "i'm busy", "im busy",
    "talk to me later", "talk later", "talk to you later",
    "call me later", "call me back", "call back later",
    "not a good time", "isn't a good time", "isnt a good time", "bad time",
    "i have to go", "i gotta go", "gotta go", "got to go", "i need to go",
    "schedule another time", "schedule later", "schedule it later",
    "can we do this later", "later please", "another time",
    # Outright refusals — live call 2026-07-11 13:01: "No, I not interested.
    # Thank you." matched nothing here, so the agent answered with "no problem,
    # could you share your name?" and the caller hung up. In a job-outreach
    # call these are unambiguous end-the-call signals.
    "not interested", "no interest", "don't want", "dont want",
    "not looking for", "wrong number", "don't call", "dont call",
    "stop calling", "remove my number", "leave me alone",
)


def _extract_spelled_letters(transcript: str) -> Optional[str]:
    """Pull a letter sequence from spelling patterns.

    Handles:
      - 'A-B-C-D'   (dash pattern) → 'ABCD'
      - 'A as in Apple, B as in Boy' (phonetic pattern) → 'AB'

    Returns the uppercase letter string (>=3 chars), or None if no clear
    spelling pattern is present.
    """
    dash_matches = re.findall(r'(?:[A-Za-z]-){2,}[A-Za-z]', transcript)
    if dash_matches:
        longest = max(dash_matches, key=len)
        letters = ''.join(c.upper() for c in longest if c.isalpha())
        if len(letters) >= 3:
            return letters

    as_in_matches = re.findall(r'\b([A-Za-z])\s+as\s+in\s+[A-Za-z]+', transcript, re.IGNORECASE)
    if len(as_in_matches) >= 3:
        return ''.join(c.upper() for c in as_in_matches)

    return None


def _build_forced_confirmation(letters: str, language: str = "en") -> str:
    """Return the EXACT sentence the AI must reply with.

    Letters are read back INDIVIDUALLY ("H, A, J, I, K") rather than as full
    NATO ("H as in Hotel, A as in Alpha, ...").  A full-NATO readback of a
    10-character name/email runs 15-17s of audio — long enough that the caller
    can't interrupt to correct an error (barge-in gets suppressed as echo),
    which turned email confirmation into a ~2-minute loop.  The exact letters
    are still spoken verbatim, so letter-accuracy protection is preserved; only
    the delivery is shortened (~5s instead of ~17s).

    Tamil callers spell English names/emails with English letter names, so the
    letters are still read verbatim — only the surrounding sentence is Tamil.
    """
    spelled = ", ".join(letters)
    if language == "ta":
        return f"சரிங்க, spelling confirm பண்றேன்: {spelled}. சரிதானே?"
    return f"Got it, let me confirm the spelling: {spelled}. Is that correct?"


# Formal/bookish Tamil the model (gpt-realtime-mini) keeps emitting despite the
# prompt banning it — the user requires the mini model, so the register is
# enforced deterministically on the outgoing TTS text instead of trusting the
# model. Ordered list (longer variants first) of (formal, colloquial).
# Expanded 2026-07-11 after a probe showed the model rewriting even the script's
# own colloquial welcome ("பேசறேன்") into bookish forms ("பேசுகிறேன்").
_TA_COLLOQUIAL_MAP = (
    ("தயவு செய்து ", "கொஞ்சம் "),
    ("தயவுசெய்து ", "கொஞ்சம் "),
    ("தயவு செய்து", "கொஞ்சம்"),
    ("தயவுசெய்து", "கொஞ்சம்"),
    ("மன்னிக்கவும்", "sorry-ங்க"),
    # Common bookish words the specific rules below don't cover.
    ("இருக்கிறீர்களா", "இருக்கீங்களா"),
    ("இருக்கிறீர்கள்", "இருக்கீங்க"),
    ("இருக்கிறதா", "இருக்கா"),
    ("இருக்கிறது", "இருக்கு"),
    ("இருக்கிறேன்", "இருக்கேன்"),
    ("வருகிறது", "வருது"),
    ("தெரிகிறதா", "தெரியுதா"),
    ("தெரிகிறது", "தெரியுது"),
    ("புரிகிறதா", "புரியுதா"),
    ("புரிகிறது", "புரியுது"),
    ("புரிந்துள்ளேன்", "புரிஞ்சது"),
    # Live-call leaks 2026-07-11 (both mini and flagship emitted these).
    ("உள்ளதா", "இருக்கா"),
    ("உள்ளது", "இருக்கு"),
    ("நன்றாக", "நல்லா"),
    ("சரியாக", "சரியா"),
    ("முழுமையாக", "முழுசா"),
    ("முழுமையா", "முழுசா"),
    ("அடுத்ததாக", "அடுத்தது"),
    ("நினைத்தேன்", "நினைச்சேன்"),
    ("உங்களின்", "உங்க"),
    ("அதில்", "அதுல"),
    ("இதில்", "இதுல"),
    ("உறுதி செய்யலாமா", "confirm பண்ணலாமா"),
    ("உறுதி செய்யுங்கள்", "confirm பண்ணுங்க"),
    ("உறுதி செய்யுங்க", "confirm பண்ணுங்க"),
    ("உறுதி செய்யவும்", "confirm பண்ணுங்க"),
    ("உறுதி செய்கிறேன்", "confirm பண்றேன்"),
    ("வேண்டுமா", "வேணுமா"),
    ("வேண்டும்", "வேணும்"),
    ("உங்களுடைய", "உங்க"),
    ("உங்களது", "உங்க"),
    ("நிச்சயமாக", "நிச்சயமா"),
    ("மிகவும்", "ரொம்ப"),
    ("மிக்க நன்றி", "ரொம்ப நன்றிங்க"),
    ("சிறிது", "கொஞ்சம்"),
    ("இல்லை", "இல்ல"),
    ("ஆனால்", "ஆனா"),
    ("எனது", "என்"),
)

# Morphological bookish→spoken rules, applied AFTER the word map (order matters:
# "க்கிற" must run before the generic "கிற" or இருக்கிறார் would lose its க).
# Each pattern anchors on Tamil context so English text and word-initial "கிற"
# (e.g. கிறிஸ்துமஸ்) are never touched.
_TA_MORPH_RULES = (
    # Present-tense marker: பேசுகிறேன்→பேசுறேன், போகிறீர்களா→போறீர்களா,
    # இருக்கிறார்→இருக்கறார் (the க்-stem variant keeps the க).
    (re.compile(r"க்கிற"), "க்கற"),
    (re.compile(r"(?<=[஀-௿])கிற"), "ற"),
    # Formal 2nd-person plural: சொன்னீர்களா→சொன்னீங்களா, சொல்வீர்கள்→சொல்வீங்க.
    (re.compile(r"ீர்களா"), "ீங்களா"),
    (re.compile(r"ீர்கள்"), "ீங்க"),
    # Word-final ங்கள்: நீங்கள்→நீங்க, சொல்லுங்கள்→சொல்லுங்க, உங்கள்→உங்க.
    (re.compile(r"ங்கள்(?![஀-௿])"), "ங்க"),
    # இப்போது/எப்பொழுது→இப்போ/எப்போ — word-final only (எப்போதும் stays intact).
    (re.compile(r"([இஎஅ])ப்ப(?:ோது|ொழுது)(?![஀-௿])"), r"\1ப்போ"),
    # Bookish quotative: "சொல்லலாம் என்று கேட்டேன்"→"சொல்லலாம்னு கேட்டேன்",
    # "correct என confirm"→"correctனு confirm". Glued to the previous word the
    # way it is actually spoken — but only after an actual word (the lookbehind
    # skips "…சரியா? என்று", where gluing onto punctuation reads broken).
    # என் (my) / என்ன (what) end in Tamil letters, so the lookahead keeps them.
    (re.compile(r"(?<=[஀-௿A-Za-z0-9])\s+என்று(?![஀-௿])"), "னு"),
    (re.compile(r"(?<=[஀-௿A-Za-z0-9])\s+என(?![஀-௿])"), "னு"),
)


def _colloquialize_ta(text: str) -> str:
    """Swap bookish Tamil words/morphology for spoken Tanglish before TTS."""
    for formal, colloquial in _TA_COLLOQUIAL_MAP:
        if formal in text:
            text = text.replace(formal, colloquial)
    for pattern, repl in _TA_MORPH_RULES:
        text = pattern.sub(repl, text)
    return text


# Ack words that open replies. Two consecutive replies opening with the SAME
# ack word read robotic (live call 2026-07-11: "சரி, ..." × 12 of 17 turns);
# the repeat is trimmed in _enqueue_tts so the reply starts with substance.
_ACK_OPENER_WORDS = frozenset((
    "சரி", "சரிங்க", "ஓகே", "ஒகே", "okay", "ok", "ஆமா", "ஆமாங்க", "ம்ம்",
    "நல்லது", "சூப்பர்", "சூப்பர்ங்க", "super", "great", "good", "perfect",
))


# Neutral "thinking" fillers pre-synthesized at call start and played instantly
# when the real reply hasn't reached the caller ~400ms after their turn ends.
# Neutral hesitations only — an "okay/சரி" after a yes/no question sounds like
# an answer. Rotated so the caller never hears the same clip twice in a row.
# SHORT hums only (measured 2026-07-11: ம்ம்=0.4s, ம்ம்ம்...=0.7s, ம்ம்...=1.0s).
# Vobiz plays buffered audio in order, so the real reply queues BEHIND the
# filler — the old "ம்ம், ஒரு நிமிஷம்ங்க..." clip (~1.9s) pushed the actual
# answer from ~1.0s out to ~2.3s, ADDING latency instead of masking it.
_FILLER_TEXTS_TA = ("ம்ம்", "ம்ம்ம்...", "ம்ம்...")
_FILLER_DELAY_S = 0.4  # play only if no real audio by this point
_FILLER_MIN_GAP_TURNS = 2  # at most one filler every N turns — overuse reads fake
# Safety net for synthesis variance (temperature 0.5): a clip that comes back
# longer than this would delay the real reply more than it masks — drop it.
_FILLER_MAX_BYTES = 9600  # 1.2s of mulaw/8k

# Listening backchannels: a human listener hums a soft "ம்ம்..." while the OTHER
# side talks at length; dead silence for 10s reads robotic. Played only while
# the caller is mid-monologue (VAD started, not yet stopped) and the agent is
# fully silent. Neutral hums ONLY — an "ஆமா/சரி" could land right after a
# question (VAD stop races the timer) and read as an answer.
_LISTEN_BC_TEXTS_TA = ("ம்ம்", "ம்ம்ம்")
_LISTEN_BC_AFTER_S = 4.5     # caller must be speaking this long before the first hum
_LISTEN_BC_REPEAT_S = 7.0    # gap before a second hum in the same monologue
_LISTEN_BC_MAX_PER_UTTERANCE = 2
_LISTEN_BC_MIN_GAP_S = 12.0  # across utterances — overuse reads fake
_LISTEN_BC_GAIN = 0.5        # a listener's hum is softer than their speech
# 1.2s cap: synthesis variance (temp 0.5) pushed hums past the old 0.8s cap and
# ALL clips got silently skipped (live call 2026-07-11: "0 listen-backchannels").
_LISTEN_BC_MAX_BYTES = 9600


def _attenuate_ulaw(ulaw_bytes: bytes, gain: float) -> bytes:
    """Scale a mulaw clip's volume down (decode → scale → re-encode)."""
    try:
        pcm = audioop.ulaw2lin(ulaw_bytes, 2)
        return audioop.lin2ulaw(audioop.mul(pcm, 2, gain), 2)
    except Exception:
        return ulaw_bytes


# Tamil "I want to end / call me later / I'm busy" phrases. Matched as substrings on
# the transcript so colloquial variants are caught.
DISENGAGE_PHRASES_TA = (
    "பிறகு பேச", "பிறகு மாட்டி", "பிறகு கூப்பி", "அப்புறம் பேச", "அப்புறம் கூப்பி",
    "அப்புறமா பேச", "வேற நேரம்", "வேற டைம்", "இப்போ முடியாது", "இப்போது முடியாது",
    "இப்போ பிஸி", "பிஸியா இருக்க", "நேரம் இல்ல", "நேரமில்ல", "டைம் இல்ல", "time இல்ல",
    "பின்னாடி பேச", "பின்னாடி கூப்பி", "பிறகு call", "later பேச",
)


def _is_disengage_intent(transcript: str, language: str = "en") -> bool:
    """True if caller's words indicate they want to end the call now."""
    lower = transcript.lower().strip()
    if language == "ta":
        # Tamil callers code-switch, so check both Tamil and English phrase lists.
        if any(p in transcript for p in DISENGAGE_PHRASES_TA):
            return True
    return any(phrase in lower for phrase in DISENGAGE_PHRASES)


# Pure courtesy / closing tokens. An utterance made up ENTIRELY of these words
# (e.g. "Perfect, all right, thank you", "okay bye", "thanks a lot") is a caller
# wrapping up — not a new request. Once the agent has ALREADY said goodbye, such a
# remark should let the pending hangup proceed instead of dragging the agent back
# into the script. Deliberately conservative: a single substantive word (e.g.
# "wait", "price", "question") makes _is_closing_remark return False, so a genuine
# follow-up still keeps the caller on the line.
_CLOSING_TOKENS = frozenset((
    "thanks", "thank", "you", "u", "ok", "okay", "kay", "alright", "right",
    "all", "perfect", "great", "good", "cool", "sure", "yeah", "yep", "yes",
    "bye", "byebye", "goodbye", "cheers", "fine", "got", "it", "take", "care",
    "see", "ya", "too", "and", "a", "lot", "much", "appreciate", "that",
    "have", "day", "nice", "wonderful", "lovely", "for", "your", "time",
))

# Tamil courtesy / closing tokens — the Tamil equivalent of _CLOSING_TOKENS, used
# (combined with the English set, since Tamil callers code-switch) to decide whether a
# post-goodbye utterance is just a sign-off. Kept conservative: one substantive Tamil
# word makes the remark "not closing", so a real follow-up keeps the caller on the line.
_CLOSING_TOKENS_TA = frozenset((
    "நன்றி", "சரி", "சரிங்க", "ஓகே", "ஓக்கே", "பை", "பைபை", "வணக்கம்", "தாங்க்ஸ்",
    "தேங்க்ஸ்", "தேங்க்யூ", "போய்ட்டு", "வரேன்", "வரேங்க", "போறேன்", "நல்லது",
    "போதும்", "சூப்பர்", "ஆமா", "ம்", "ஓம்", "சரிசரி", "தாங்க்யூ",
))


def _is_closing_remark(transcript: str, language: str = "en") -> bool:
    """True if the WHOLE utterance is just courtesy/closing words. Used only after
    the agent has already said goodbye, to decide whether the caller is simply
    signing off (proceed to hang up) or has a real follow-up (stay on the line)."""
    if language == "ta":
        # Tamil has no letter case; split on non-space/punct so Tamil graphemes survive.
        words = re.findall(r"[^\s,.!?]+", transcript.lower())
        if not words:
            return False
        allowed = _CLOSING_TOKENS_TA | _CLOSING_TOKENS
        return all(w in allowed for w in words)
    words = re.findall(r"[a-z]+", transcript.lower())
    if not words:
        return False
    return all(w in _CLOSING_TOKENS for w in words)


def _amplify_ulaw(ulaw_bytes: bytes, target_peak: float, max_gain: float) -> bytes:
    """Peak-normalize an 8kHz mu-law buffer to make it audibly louder.

    ElevenLabs ulaw_8000 output is quiet over a phone line. We decode to 16-bit
    linear PCM, scale so the loudest sample reaches `target_peak` of full scale,
    then re-encode. Gain is capped at `max_gain` (so near-silence isn't blown up)
    and never applied below 1.0 (so already-loud audio is left untouched). Because
    the gain is derived from the actual peak, normalization does NOT clip.
    """
    if not ulaw_bytes:
        return ulaw_bytes
    try:
        pcm = audioop.ulaw2lin(ulaw_bytes, 2)
        peak = audioop.max(pcm, 2)  # largest absolute sample value
        if peak <= 0:
            return ulaw_bytes
        gain = (target_peak * 32767.0) / peak
        gain = min(gain, max_gain)
        if gain <= 1.0:
            return ulaw_bytes
        return audioop.lin2ulaw(audioop.mul(pcm, 2, gain), 2)
    except Exception as e:
        logger.debug(f"_amplify_ulaw failed, sending original: {e}")
        return ulaw_bytes


# Boundaries we prefer to break a long chunk on, best-first: sentence enders,
# then clause separators. Falls back to whitespace, then a hard slice.
_TTS_SPLIT_RE = re.compile(r'(?<=[.!?,;:—])\s+|\s+(?=[—])')


def _split_for_tts(text: str, max_chars: int) -> list[str]:
    """Split text into chunks no longer than max_chars for TTS synthesis.

    A single OpenAI runaway (e.g. hitting max_output_tokens with no sentence
    punctuation) must never become one multi-second, un-interruptible blob.
    We break on clause boundaries first, then whitespace, then — as a last
    resort — slice mid-word. Each returned chunk is enqueued separately so an
    interrupt can drain the rest between chunks.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""

    def emit():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for unit in _TTS_SPLIT_RE.split(text):
        if not unit:
            continue
        # A single clause longer than the cap: break it on whitespace.
        if len(unit) > max_chars:
            emit()
            for word in unit.split():
                if current and len(current) + 1 + len(word) > max_chars:
                    emit()
                # A single word longer than the cap: hard-slice it.
                while len(word) > max_chars:
                    chunks.append(word[:max_chars])
                    word = word[max_chars:]
                current = f"{current} {word}".strip() if current else word
            continue
        if current and len(current) + 1 + len(unit) > max_chars:
            emit()
        current = f"{current} {unit}".strip() if current else unit

    emit()
    return chunks


class VobizRealtimeHandler:
    """
    Bridges Vobiz ↔ OpenAI Realtime API.

    Audio format: audio/x-mulaw 8kHz (same as Twilio g711_ulaw — zero conversion!)

    When tts_provider="openai" (default):
      - g711_ulaw in/out — zero conversion, ~300ms latency.
    When tts_provider="elevenlabs" or "sarvam" (external TTS):
      - g711_ulaw input, text output from OpenAI
      - Sentence-by-sentence → TTS streaming → mulaw → Vobiz
      - Uses ordered queue so sentences never overlap or reorder.
    """

    OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"

    def __init__(self, use_elevenlabs: bool = False, active_script: dict = None,
                 language: str = "en", tts_provider: str = None,
                 called_number: str = ""):
        self.stream_id: Optional[str] = None
        self.call_id: Optional[str] = None
        self.openai_ws = None
        self._connected = False
        self._vobiz_ws = None
        # Per-call TTS provider ("openai" | "elevenlabs" | "sarvam"). The legacy
        # use_elevenlabs bool is kept as "external TTS mode" (OpenAI outputs text,
        # a TTS provider speaks it) so every existing mode check stays valid.
        if tts_provider not in Config.TTS_PROVIDERS:
            tts_provider = "elevenlabs" if use_elevenlabs else "openai"
        self.tts_provider = tts_provider
        self.use_elevenlabs = tts_provider != "openai"
        # Per-call language ("en" | "ta"). Drives prompt, transcription, native voice,
        # external TTS voice and the goodbye/closing heuristics. May be overridden from
        # the Vobiz stream `start` event's extra_headers (set by the answer_url).
        self._language = language if language in Config.SUPPORTED_LANGUAGES else "en"
        # The number WE dialed (outbound calls). Live call 2026-07-11: caller said
        # "இந்த number-க்கே WhatsApp அனுப்புங்க" and the model invented 9 wrong
        # digits — the system knew the real number the whole time. Threaded into
        # the prompt so "this number" resolves without guessing.
        self._called_number = str(called_number or "").strip()
        self._external_tts = (
            self._make_external_tts() if self.use_elevenlabs else None
        )
        # External TTS is skipped in favour of OpenAI TTS until this monotonic
        # deadline. 0.0 = available now. Set on failure, never latched forever.
        self._external_tts_retry_at = 0.0
        self._openai_tts_client = None
        self._response_text_buffer = ""
        # Text pieces of the CURRENT response that actually reached the caller
        # (external TTS mode). On interruption, OpenAI's history still holds the
        # full generated text as if it was all spoken — but drained queue pieces
        # were never heard. This is the ground truth the recovery injection
        # gives the model so it re-asks what was cut off instead of "forgetting"
        # it (live-call bug: agent abandoned mid-question after every barge-in).
        self._heard_text_this_response = ""
        # (text, seconds) per TTS piece SENT to Vobiz this response. "Sent" is
        # not "heard": Vobiz buffers several seconds ahead, so a clearAudio cuts
        # sent-but-unplayed audio. _snapshot_playback_cut() uses the durations +
        # wall clock to split sent pieces into actually-heard vs never-played.
        self._sent_pieces: list[tuple[str, float]] = []
        self._unheard_text_this_response = ""
        # When the first REAL audio chunk of this response starts playing at the
        # carrier (fillers excluded) — anchor for the playback-cut math.
        self._response_audio_started_at = 0.0
        # False until the caller has heard ANY real agent audio this call.
        # Call setup takes 2-4s of silence and humans say "hello?" into it —
        # that speech must not be treated as an interrupt of a welcome they
        # have never heard (live bug 2026-07-11: greeting silently skipped).
        self._any_real_audio_sent = False
        # Text of the current/last completed response (raw model text) — the
        # "unheard" fallback when a reply is killed before ANY audio went out.
        self._last_response_text = ""
        # True while the current response is a suppressed auto-reply to
        # pre-welcome speech — its text deltas are discarded, not synthesized.
        self._discard_response_text = False
        # Safety valve: if welcome audio never materializes (TTS totally down),
        # stop suppressing replies after a couple — a wrong reply beats a mute
        # agent.
        self._pre_welcome_suppressions = 0
        # Deadline (monotonic) until which a newly-created response is treated
        # as the server's auto-answer to a backchannel and suppressed. 0 = off.
        self._suppress_bc_response_until = 0.0
        # False until this response's first TTS piece went out — the first piece
        # may flush early at a clause boundary to cut time-to-first-audio.
        self._first_piece_flushed = False
        self._tts_queue: asyncio.Queue = asyncio.Queue(maxsize=20)
        self._tts_worker_task = None
        self._tts_generation = 0
        self._receive_task: Optional[asyncio.Task] = None
        # Background one-shot tasks (fillers, watchdogs, gentle clears). Held so
        # the GC can't collect them mid-flight and so cleanup can cancel them.
        self._bg_tasks: set[asyncio.Task] = set()
        # Script passed directly from main.py at call time
        self._script: Optional[dict] = active_script
        self._goodbye_detected = False
        self._hangup_task: Optional[asyncio.Task] = None
        # True after goodbye while we wait for the caller's last words to decide
        # whether to proceed with the hangup or cancel it for a real follow-up.
        self._awaiting_post_goodbye_eval = False
        # Interruption tracking
        self._current_ai_transcript = ""
        self._ai_is_responding = False   # True while OpenAI is generating audio
        self._tts_playing = False        # True while TTS audio is being sent to Vobiz
        # Interruption state
        self._last_interrupted_transcript = ""
        self._interrupt_pending = False
        # Audio deltas buffered during the backchannel-evaluation window. A
        # "mm-hm" ruling flushes them (words no longer vanish mid-sentence); a
        # real interrupt discards them.
        self._pending_audio_deltas: list[str] = []
        # True from the moment we send response.cancel until the next response
        # settles — gates goodbye detection/transcript clearing so a cancelled
        # response can't schedule a hangup for words the caller never heard.
        self._cancel_requested = False
        # Set when OpenAI rejects the server-VAD auto-created response because
        # the previous response hadn't finalized; response.done retries it so
        # the caller's turn is never silently dropped (live-call dead-air bug).
        self._response_create_pending = False
        self._speech_start_time = 0.0
        # audio_start_ms from the last speech_started event (buffer-relative).
        self._speech_start_audio_ms: Optional[int] = None
        self._first_audio_chunk = True
        self._audio_chunk_count = 0
        self._response_start_time = 0.0
        self._speech_stopped_time = 0.0
        self._cleanup_started = False
        self._turn_count = 0
        self._interrupt_count = 0
        self._total_realtime_tokens = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._cached_input_tokens = 0
        self._transcription_tokens = 0
        self._latency_samples_ms: list[float] = []
        # Native Realtime audio playback accounting for interruption truncation.
        self._current_response_item_id: Optional[str] = None
        self._response_audio_sent_ms = 0.0
        self._response_playback_started_at = 0.0
        # PCM16 buffer for ElevenLabs mode
        self._pcm_buffer = b""
        # Pre-warmed OpenAI WebSocket
        self._prewarm_task: Optional[asyncio.Task] = None
        self._session_preconfigured = False
        # Echo gate
        self._echo_gate_until = 0.0
        self._ECHO_COOLDOWN = Config.ECHO_COOLDOWN_S
        # Estimated time when Vobiz finishes playing all queued TTS audio
        self._estimated_playback_end = 0.0
        # First response = the welcome. Must be barge-in-able without echo-gate
        # filtering or backchannel heuristics, otherwise the caller cannot
        # interrupt the greeting.
        self._is_first_response = True
        # Pre-synthesized filler audio (mulaw 8k) for perceived-latency masking.
        self._filler_clips: list[bytes] = []
        self._filler_index = 0
        self._filler_last_turn = -_FILLER_MIN_GAP_TURNS
        self._filler_synth_task: Optional[asyncio.Task] = None
        # Soft listening backchannels ("ம்ம்...") played while the CALLER talks
        # at length — distinct from _filler_clips (which mask reply latency).
        self._listen_bc_clips: list[bytes] = []
        self._listen_bc_index = 0
        self._listen_bc_last_ts = -_LISTEN_BC_MIN_GAP_S
        self._listen_bc_task: Optional[asyncio.Task] = None
        # Opener de-dup: first short sentence of the previous response — a mini
        # model repeats "சரி, நன்றி." style openers verbatim every turn.
        self._prev_opener = ""
        # First WORD of the previous response's opener when it was an ack token
        # ("சரி"/"ஓகே"/...). Live call 2026-07-11: 12 of 17 turns began "சரி," —
        # consecutive same-word openers are trimmed so at most every other turn
        # starts with the same ack.
        self._prev_opener_word = ""
        self._openers_seen_this_response = False
        # Set when WE cancel a response programmatically (e.g. to apply a forced
        # spelling confirmation) — the "caller interrupted you" recovery message
        # must not be injected for a cancel the caller didn't cause.
        self._suppress_recovery_once = False
        # Optional observer hook used by the in-browser test-call bridge to
        # mirror transcripts to a UI. None in production (Vobiz) path — then
        # this is a no-op and behavior is identical to before.
        self._transcript_callback = None  # async (role: str, text: str, final: bool) -> None

    def _make_external_tts(self):
        """Build the external TTS service for the current provider + language."""
        if self.tts_provider == "sarvam":
            return SarvamTTSService(language=self._language)
        return ElevenLabsTTSService(language=self._language)

    # ── Vobiz message handling ──────────────────────────────────────────────

    async def handle_vobiz_message(self, vobiz_ws, message: dict):
        """Process a JSON message from Vobiz's media stream."""
        event = message.get("event")

        if event == "start":
            start_data = message.get("start", {})
            self.stream_id = start_data.get("streamId")
            self.call_id = start_data.get("callId")
            self._vobiz_ws = vobiz_ws

            # Check extra_headers for the provider/language config. Vobiz sends
            # them as "{X-VH-provider: sarvam, X-VH-language: ta}" (braces,
            # colon-separated, X-VH- prefix — observed live 2026-07-10); the
            # test-call bridge sends legacy "key=value,key2=value2".
            def _norm_key(key: str) -> str:
                key = key.strip().strip("{}").strip()
                if key.lower().startswith("x-vh-"):
                    key = key[5:]
                return key

            extra_headers_raw = message.get("extra_headers", "")
            extra_headers = {}
            if isinstance(extra_headers_raw, dict):
                extra_headers = {_norm_key(str(k)): str(v)
                                 for k, v in extra_headers_raw.items()}
            elif isinstance(extra_headers_raw, str) and extra_headers_raw:
                try:
                    parsed = json.loads(extra_headers_raw)
                    if isinstance(parsed, dict):
                        extra_headers = {_norm_key(str(k)): str(v)
                                         for k, v in parsed.items()}
                    else:
                        raise ValueError("not a dict")
                except (json.JSONDecodeError, ValueError):
                    for pair in extra_headers_raw.strip().strip("{}").split(","):
                        if "=" in pair:
                            k, v = pair.split("=", 1)
                        elif ":" in pair:
                            k, v = pair.split(":", 1)
                        else:
                            continue
                        k = _norm_key(k)
                        if k:
                            extra_headers[k] = v.strip()

            # Provider from extra_headers (set by the answer_url / test-call bridge).
            # `provider` wins; the legacy `elevenlabs` flag is honored as a fallback.
            provider_param = str(extra_headers.get("provider", "")).strip().lower()
            if provider_param in Config.TTS_PROVIDERS:
                self.tts_provider = provider_param
                self.use_elevenlabs = provider_param != "openai"
            elif str(extra_headers.get("elevenlabs", "false")).lower() == "true":
                self.tts_provider = "elevenlabs"
                self.use_elevenlabs = True

            # Language (en|ta) from extra_headers — set by the answer_url for outbound
            # calls / the test-call bridge. Falls back to whatever the constructor got.
            lang_param = extra_headers.get("language") or extra_headers.get("lang")
            if isinstance(lang_param, str) and lang_param.strip().lower() in Config.SUPPORTED_LANGUAGES:
                self._language = lang_param.strip().lower()

            # Ensure TTS worker is running if external-TTS mode (from constructor or
            # extra_headers). (Re)create the service with the final provider + language
            # so the right voice/language_code apply even when they arrived via headers.
            if self.use_elevenlabs and not self._tts_worker_task:
                self._external_tts = self._make_external_tts()
                self._external_tts_retry_at = 0.0
                self._tts_worker_task = asyncio.create_task(self._tts_worker())
                # Sarvam speaks over a persistent streaming WS; open it now so the
                # ~0.7-1s handshake overlaps OpenAI session setup instead of
                # delaying the greeting's first audio.
                if self.tts_provider == "sarvam":
                    self._external_tts.prewarm()
                    if self._language == "ta":
                        self._filler_synth_task = asyncio.create_task(
                            self._synthesize_fillers()
                        )

            mode_label = (
                f"{self.tts_provider} TTS" if self.use_elevenlabs else "speech-to-speech"
            )
            has_script = "with script" if self._script else "no script"
            logger.info(
                f"Vobiz stream started — streamId={self.stream_id}, "
                f"callId={self.call_id}, mode={mode_label}, {has_script}"
            )
            await self._connect_openai()

        elif event == "media":
            payload = message.get("media", {}).get("payload", "")
            if payload and self._connected:
                await self._forward_audio(payload)

        elif event == "stop":
            reason = message.get("reason", "unknown")
            logger.info(f"Vobiz stream stopped — callId={self.call_id}, reason={reason}")
            await self._full_cleanup()

    # ── OpenAI Realtime connection ──────────────────────────────────────────

    async def _connect_openai(self):
        """Connect to OpenAI Realtime API (with pre-warm support)."""
        try:
            if self._prewarm_task:
                print("[CALL] Awaiting pre-warmed OpenAI connection...", flush=True)
                try:
                    self.openai_ws = await asyncio.wait_for(
                        self._prewarm_task, timeout=10.0
                    )
                except Exception as e:
                    print(f"[CALL] Pre-warm failed ({e}), cold-connecting...", flush=True)
                    self.openai_ws = None
                finally:
                    self._prewarm_task = None

            if not self.openai_ws:
                self._session_preconfigured = False
                model = Config.REALTIME_MODEL
                url = f"{self.OPENAI_REALTIME_URL}?model={model}"
                headers = {
                    "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
                }
                print(f"[CALL] Connecting to OpenAI Realtime ({model})...", flush=True)
                self.openai_ws = await asyncio.wait_for(
                    websockets.connect(
                        url,
                        additional_headers=headers,
                        ping_interval=30,
                        ping_timeout=10,
                        max_size=2**20,
                        compression=None,
                    ),
                    timeout=10.0,
                )

            print("[CALL] OpenAI WebSocket connected OK", flush=True)
            confirmed = await self._configure_session()
            if not confirmed:
                print("[CALL] *** SESSION CONFIG FAILED — aborting ***", flush=True)
                await self._close_vobiz_on_failure("Session config failed")
                return

            self._connected = True
            print("[CALL] _connected=True — audio forwarding enabled", flush=True)
            self._receive_task = asyncio.create_task(self._receive_openai_events())
            await self._trigger_initial_response()

        except asyncio.TimeoutError:
            print("[CALL] *** OpenAI connection TIMED OUT (10s) ***", flush=True)
            self._connected = False
            await self._close_vobiz_on_failure("OpenAI connection timed out")
        except Exception as e:
            print(f"[CALL] *** OpenAI connection FAILED: {e} ***", flush=True)
            self._connected = False
            await self._close_vobiz_on_failure(str(e))

    async def _trigger_initial_response(self):
        """Send response.create to start the welcome message."""
        try:
            ws = self.openai_ws
            if ws:
                await ws.send(json.dumps({"type": "response.create"}))
                logger.info("Triggered initial AI response")
        except Exception as e:
            logger.error(f"Failed to trigger initial response: {e}")

    def _build_prompt(self) -> str:
        """Build system prompt from script config or use default."""
        # Tamil calls use a separate builder so the proven English prompt below is
        # never touched (zero regression risk on the English path).
        if self._language == "ta":
            return self._build_prompt_ta()
        if not self._script:
            return Config.SYSTEM_PROMPT

        s = self._script
        parts = []
        n = len(s.get("questions", []))

        parts.append(
            "You are on a live phone call. You hear the caller's voice directly. "
            "Be warm, professional, English only. Keep responses to 1-2 sentences. "
            "Speak slowly and clearly — the caller may not understand fast speech."
        )

        if s.get("behaviour"):
            parts.append(s['behaviour'])

        if s.get("welcome"):
            parts.append(
                f'Start the conversation immediately by saying: "{s["welcome"]}" '
                "Do NOT wait for the caller to speak first — you are making an outbound call, "
                "so introduce yourself right away."
            )

        questions = s.get("questions", [])
        if questions:
            q_list = [f"Q{i+1}: {q.get('question', q) if isinstance(q, dict) else q}"
                      for i, q in enumerate(questions)]
            parts.append(
                f"QUESTIONS — ask ALL {n} in order (Q1, Q2, Q3... do NOT skip any):\n"
                + "\n".join(q_list)
            )

        if s.get("goal"):
            parts.append(f"GOAL: {s['goal']}")

        if self._called_number:
            parts.append(
                f"You dialed the caller at {self._called_number}. If they say "
                "'this number' / 'send it to this number', that IS this number — "
                "never guess digits or read back a different number."
            )

        parts.append(
            "RULES YOU MUST FOLLOW:\n"
            "1. ANSWER FIRST, then continue: If the caller asks 'who are you?', 'what is this?', "
            "'what's this call about?', 'why are you calling?', or seems confused about the call's "
            "purpose — STOP your script immediately. Briefly re-state: who you are, the position you're "
            "calling about, and why. Then ask 'Would you like to continue?'. Only after they confirm, "
            "resume your questions. Do NOT push past confusion — it kills the conversation.\n"
            "2. Only move to the next question AFTER the caller answers the current one. "
            "Questions, complaints, 'I can't hear you' are NOT answers — address them, then re-ask.\n"
            "3. When caller says 'no', 'wrong', 'that's not right', 'that's wrong', or ANY disagreement — STOP. "
            "You made a mistake. Apologize and ask them to provide it again. "
            "Do NOT repeat your previous wrong answer. Do NOT move to the next question.\n"
            "4. EXACT REPETITION: When confirming what the caller said, use their EXACT words. "
            "Do NOT change numbers (8 stays 8, not 7), units (weeks stays weeks, not months), "
            "tool names (n8n stays n8n, not Jenkins), amounts (1 crore stays 1 crore). Zero changes.\n"
            "5. If caller is frustrated — acknowledge, apologize, ask if they want to continue.\n"
            "6. If audio is bad or they ask to repeat — apologize and repeat your last question.\n"
            "7. If caller asks to start over — say 'Sure, let me start fresh' and wait.\n"
            "8. ENDING THE CALL — be conservative. Only say 'Thank you for your time. Have a great day! "
            "Goodbye.' when ONE of these is unambiguously true:\n"
            "   (a) You have asked ALL questions AND given the full recap AND they confirmed.\n"
            "   (b) The caller EXPLICITLY says they want to end — e.g. 'I have to go', 'I'm done', "
            "'let's end this', 'I'm not interested', 'don't call me', 'stop calling'.\n"
            "   A standalone 'bye', 'bye bye', 'thank you bye', or 'okay bye' while questions are still "
            "pending is AMBIGUOUS — DO NOT end the call. Instead, politely re-ask your current question: "
            "'I just need a few more details — [re-ask current question]'. Never say goodbye prematurely.\n"
            "9. \"HELLO?\" IS NOT A RESTART: If the caller just says 'hello?', 'are you there?', or "
            "'can you hear me?', do NOT replay your whole introduction. Give a brief 'Yes, I can hear "
            "you — can you hear me okay?' and continue from where you left off.\n"
            "9b. INTRODUCE YOURSELF ONCE. After your opening line, NEVER re-deliver the company+role "
            "pitch again ('This is Akash from Zillion Connects, calling about the AR Caller role...'). "
            "If the caller is confused or keeps saying 'hello?', give ONE short new detail or just "
            "re-ask your current question — a real person does not repeat their whole introduction "
            "two or three times. Repeating the pitch is the #1 thing that makes this sound like a bot.\n"
            "10. CORRECTIONS WIN: If the caller restates something differently than you just confirmed "
            "(you said '500 to 1000', they say '500 to 1500'), they are CORRECTING you. Immediately say "
            "the NEW value back ('Got it, 500 to 1500 — thank you') and use it from then on. Never ignore "
            "a restated number or keep your old value.\n"
            "11. UNCLEAR SPEECH: If you could not clearly understand what the caller said — mumbled, "
            "noisy, cut off — say 'Sorry, I didn't catch that — could you say that again?' and re-ask. "
            "NEVER pretend you understood. NEVER say 'thank you for confirming' unless they clearly "
            "said yes/correct. NEVER guess a name, number, or amount you did not clearly hear — asking "
            "them to repeat is always better than confirming a wrong value.\n"
            "11b. AMBIGUOUS NUMBERS: If the caller's words do not map to ONE clear number (e.g. "
            "'thousand fifteen hundred' — that is not a single number), do NOT invent a value like "
            "'12,500'. Say what you think you heard and ask: 'Sorry, was that 1,500? Could you say the "
            "number again?'. Confirming a made-up number destroys trust instantly.\n"
            "12. NO SELF-NARRATION: Never say 'let me start over', 'let me clarify', 'I'll explain "
            "again', 'let me introduce myself again', or any commentary about what you are doing. "
            "Just say the content directly — humans don't announce their sentences.\n"
            "\n"
            "NAMES AND EMAILS — CRITICAL ACCURACY RULES:\n"
            "Phone audio is LOW quality. Single letters sound identical (B/D/P/T, M/N, S/F). "
            "Word-spelling is your TOOL for hard cases — but it is a FALLBACK, NOT the default.\n"
            "\n"
            "1. NEVER guess or assume a name. NEVER substitute what you heard with a common name. "
            "If it sounds like 'Hajik', say 'Hajik' — do NOT change it to Rajiv, Sajith, or Ranjith.\n"
            "2. CONFIRM FIRST — DO NOT force spelling. When the caller says their name, simply repeat "
            "it back ONCE to confirm: 'Got it, Priya — is that correct?'. If they confirm (say 'yes', "
            "'right', 'correct', or just continue), ACCEPT the name and move on to the next question. "
            "Do NOT ask them to spell it. Do NOT ask the same question again.\n"
            "3. ONLY ask the caller to spell their name using words ('Could you spell that — like P as "
            "in Papa?') in ONE of these cases: (a) the caller said your confirmation was WRONG, or "
            "(b) you genuinely could not make out the name at all. Otherwise, never ask for spelling.\n"
            "4. If the caller ALREADY spelled it — either with dashes like 'H-A-J-I-K' OR with words "
            "like 'H as in Hotel, A as in Apple' — ACCEPT that spelling immediately. Do NOT ask them "
            "to spell again. Take the FIRST LETTER of each word: 'H as in Hotel, A as in Apple, J as in "
            "Japan, I as in India, K as in King' → H-A-J-I-K → Hajik. Write EXACTLY those letters and "
            "confirm: 'Got it, Hajik. Is that correct?'\n"
            "5. Do NOT 'correct' a spelled result to a name you recognize. "
            "The spelled letters are the truth — they override whatever you thought you heard.\n"
            "6. When confirming back, just say the name naturally: 'Got it, Hajik. Is that correct?' "
            "Do NOT spell it back with words. Do NOT say letters with dashes. Just say the name.\n"
            "7. If they say 'no' or 'wrong': apologize, FORGET your previous attempt completely, "
            "and THEN ask them to spell it using words. Do NOT repeat your wrong version.\n"
            "8. EMAILS: NEVER guess or auto-complete. Ask them to spell the part before @ using words too. "
            "If they say 'H as in Hotel, A as in Apple, J as in Japan, I as in India, K as in King, "
            "R as in Red, E as in Echo, N as in November, A as in Apple, A as in Apple at gmail.com', "
            "the email is hajikrenaa@gmail.com. Write EXACTLY those letters. Do NOT rearrange.\n"
            "9. Do NOT end the call until you have collected ALL required information "
            "or the caller explicitly wants to leave.\n"
            "10. NAME CORRECTIONS / IDENTITY DRIFT: If the caller gives a DIFFERENT name later in the "
            "call (e.g. earlier 'Haji', later says 'Muhammad'), USE THE LATEST name as the truth. "
            "Silently discard the previous name — do NOT mix the two, do NOT challenge them with "
            "'you said X before'. Just accept the new name and re-confirm it once.\n"
            "\n"
            "CONVERSATION FLOW:\n"
            "- After a confirmed answer, react briefly, then ask the next question.\n"
            "- Use exact tool names, framework names — never substitute.\n"
            f"- After ALL {n} questions are answered and confirmed: give a complete recap of "
            "everything collected, then ask 'Did I get everything right?', then say Goodbye.\n"
            "- ENDING: When you say goodbye, include EVERYTHING in ONE response "
            "(thank you + next steps + goodbye). After saying goodbye, STOP completely. "
            "Do NOT respond to the caller's 'bye' or 'thank you' — the call is over."
        )

        return "\n\n".join(parts)

    def _build_prompt_ta(self) -> str:
        """Tamil-mode system prompt — colloquial spoken Tamil + Tanglish.

        Kept entirely separate from the English _build_prompt. The core language
        lock + register is written in Tamil (per OpenAI's Realtime prompting guidance:
        an in-language lock reduces drift back to English on names/numbers/short turns);
        the script's own welcome/questions/goal may be authored in English and the model
        is told to deliver them in natural Tamil.
        """
        # Register (user requirement 2026-07-09): a MATURED native Tamilian speaking
        # natural Tanglish — everyday English words stay in English inside Tamil
        # sentences; calm, composed delivery, never bookish, never over-excited.
        # gpt-realtime-mini drowns in the full 10KB+ prompt (live 2026-07-11:
        # broken Tamil grammar, guessed names, role reversal — while the flagship
        # on the SAME prompt behaved). Mini gets a condensed prompt: fewer,
        # sharper rules it can actually hold onto.
        is_mini = "mini" in (Config.REALTIME_MODEL or "").lower()
        if is_mini:
            lang_lock = (
                "நீங்க ஒரு உண்மையான, நிதானமான native Tamil professional — live "
                "ஃபோன் கால்ல பேசறீங்க. சரியான, இயல்பான பேச்சுத் தமிழ்ல மட்டும் "
                "பேசுங்க — உடைஞ்ச / இலக்கணத் தப்பான வாக்கியம் கூடாது.\n"
                "Tanglish விதி: வாக்கியத்தோட ஓட்டம் எப்பவும் தமிழ்; daily English "
                "வார்த்தைகள் (name, email, confirm, ok, sorry, plot, budget) "
                "English-லயே. செந்தமிழ் / புத்தகத் தமிழ் கூடாது; முழு வாக்கியம் "
                "English-லயும் கூடாது. உதாரணம்: 'சரிங்க, உங்க details confirm "
                "பண்ணிக்கிறேன்.' / 'கொஞ்சம் நிதானமா சொல்லுங்க, நான் note "
                "பண்ணிக்கிறேன்.'\n"
                "சுருக்கம்: ஒரு reply = 1-2 சின்ன வாக்கியம்; ஒரு தடவைக்கு ஒரே "
                "கேள்வி. மரியாதை: 'நீங்க' + '-ங்க' (சொல்லுங்க, வாங்க). அடுத்தடுத்த "
                "ரெண்டு replies ஒரே வார்த்தையில ஆரம்பிக்காதீங்க."
            )
        else:
            lang_lock = (
            "நீங்க ஒரு உண்மையான மனிதர் மாதிரி live ஃபோன் கால்ல பேசறீங்க — அழைப்பவரோட "
            "குரலை நேரடியா கேக்கறீங்க.\n"
            "நீங்க யாரு: நிதானமான, அனுபவமுள்ள (matured) native Tamil professional. "
            "அமைதியா, நம்பிக்கையா, மரியாதையா பேசுவீங்க — over-excitement, டீன் ஏஜ் slang, "
            "செயற்கையான உற்சாகம் எதுவும் கிடையாது.\n"
            "மொழி விதி (கண்டிப்பா கடைபிடிங்க): இயல்பான அன்றாட பேச்சுத் தமிழ் + Tanglish. "
            "செந்தமிழ் / எழுத்துத் தமிழ் / கவிதை நடை வேண்டவே வேண்டாம். "
            "அழைப்பவர் ஆங்கிலத்துல பேசினாலும் நீங்க தமிழ் ஓட்டத்துலயே பதில் சொல்லுங்க.\n"
            "Tanglish-தான் இயல்பு: Tamil பேசுறவங்க daily-ஆ English-ல சொல்ற வார்த்தைகளை "
            "(appointment, confirm, details, address, email, time, morning, call, ok, sorry, "
            "OTP, payment, WhatsApp) அப்படியே English-லயே சொல்லுங்க — கஷ்டப்பட்டு தமிழ்ல "
            "மொழிபெயர்க்காதீங்க ('மின்னஞ்சல்' இல்ல 'email'; 'உறுதி செய்யறேன்' இல்ல "
            "'confirm பண்றேன்'; 'முகவரி' இல்ல 'address'). "
            "ஆனா முழு வாக்கியத்தையும் English-ல பேசாதீங்க — வாக்கியத்தோட ஓட்டம் எப்பவும் "
            "தமிழ்தான்; English வார்த்தைகள் அதுக்குள்ள இயல்பா வரணும்.\n"
            "மரியாதை: 'நீங்க'ன்னு கூப்பிடுங்க (ஒருபோதும் 'நீ' வேண்டாம்), '-ங்க' சேர்த்து பேசுங்க "
            "(சொல்லுங்க, வாங்க). ஒரு தடவைக்கு 1-2 சின்ன வாக்கியம் மட்டும்; நிதானமா, தெளிவா பேசுங்க.\n"
            "பேசும் style (இந்த மாதிரி — நிதானமான native Tanglish): "
            "'வணக்கம்ங்க, நான் [பெயர்] பேசறேன்.' / 'சரிங்க, உங்க details confirm பண்ணிக்கிறேன்.' / "
            "'நாளைக்கு morning பத்து மணிக்கு appointment வச்சுக்கலாமா?' / "
            "'புரிஞ்சுதுங்க, கவலைப்படாதீங்க — நான் பாத்துக்கிறேன்.' / "
            "'கொஞ்சம் நிதானமா சொல்லுங்க, நான் note பண்ணிக்கிறேன்.'\n"
            "VARIETY விதி — கண்டிப்பா: அடுத்தடுத்த ரெண்டு replies ஒரே வார்த்தையில "
            "ஆரம்பிக்கவே கூடாது. 'சரி, நன்றி'னு ஒவ்வொரு தடவையும் சொல்லாதீங்க — openers "
            "மாத்தி மாத்தி use பண்ணுங்க: 'சரி' / 'ஆமா' / 'ம்ம்' / 'okay' / opener இல்லாம "
            "நேரடியா பதில். 'ம்ம்', 'ஆ', 'அப்புறம்' மாதிரி சின்ன filler வார்த்தைகள் "
            "பேச்சுல வர்றது நல்லது — அது தான் natural. ஒரு தடவைக்கு ஒரே ஒரு கேள்வி மட்டும்."
        )

        if not self._script:
            return lang_lock + "\n\n" + Config.SYSTEM_PROMPT_TA

        s = self._script
        n = len(s.get("questions", []))
        parts = [lang_lock]

        # The user's script (welcome / questions / behaviour) is usually written in
        # English and may even contain "English only" — which directly contradicts the
        # Tamil lock. State explicitly that the language rule overrides anything below,
        # otherwise weaker models (e.g. gpt-realtime-mini) obey the embedded "English only".
        # IMPORTANT: this override must say "Tamil-flow Tanglish", NOT "Tamil only" —
        # a "translate everything to Tamil / every word Tamil" instruction here made
        # the agent speak bookish pure Tamil and defeat the Tanglish register.
        parts.append(
            "மிக முக்கியம் — மொழி மேலாதிக்கம்: கீழ வர script, behaviour, கேள்விகள் "
            "ஆங்கிலத்துல இருக்கலாம்; சில சமயம் 'English only' மாதிரி வரியும் இருக்கலாம். "
            "மொழி விஷயத்துல அதையெல்லாம் முழுசா அலட்சியம் பண்ணுங்க. கீழ எதை சொன்னாலும் சரி, "
            "நீங்க மேல சொன்ன matured Tanglish style-லயே பேசணும் — வாக்கியத்தோட ஓட்டம் தமிழ், "
            "daily English வார்த்தைகள் (name, email, position, experience, salary, notice "
            "period மாதிரி) English-லயே. Script content-ஐ அந்த style-ல இயல்பா சொல்லுங்க — "
            "word-to-word மொழிபெயர்ப்பு வேண்டாம். (content-ஐ மட்டும் பயன்படுத்துங்க; அதன் "
            "மொழி வழிமுறையை அல்ல.)"
        )

        if s.get("behaviour"):
            parts.append("நடத்தை (உள்ளடக்கம் மட்டும் — மொழி தமிழ்தான்): " + s["behaviour"])

        if s.get("welcome"):
            parts.append(
                "அழைப்பை உடனே தொடங்குங்க — அழைப்பவர் பேசறதுக்கு காத்திருக்காதீங்க (இது நீங்க "
                "பண்ற outbound call). கீழ வர வரவேற்பை மேல சொன்ன matured Tanglish style-ல "
                "சொல்லுங்க — தமிழ் ஓட்டம், company/position/technical வார்த்தைகள் English-லயே. "
                "முழு வரவேற்பையும் English-ல மட்டும் படிக்காதீங்க: "
                f"\"{s['welcome']}\""
            )

        questions = s.get("questions", [])
        if questions:
            q_list = [f"Q{i+1}: {q.get('question', q) if isinstance(q, dict) else q}"
                      for i, q in enumerate(questions)]
            parts.append(
                f"கேள்விகள் — கீழ உள்ள {n} கேள்விகளையும் வரிசையா கேளுங்க (எதையும் "
                "தவிர்க்காதீங்க). கேள்வி ஆங்கிலத்துல இருந்தாலும் அதை matured Tanglish-ல "
                "கேளுங்க — தமிழ் வாக்கியம், technical/HR வார்த்தைகள் (experience, salary, "
                "notice period, skills மாதிரி) English-லயே:\n"
                + "\n".join(q_list)
            )

        if s.get("goal"):
            parts.append("இலக்கு (GOAL): " + s["goal"])

        if not is_mini:
            parts.append(
            "கடைபிடிக்க வேண்டிய விதிகள்:\n"
            "1. முதல்ல பதில், அப்புறம் தொடரவும்: அழைப்பவர் 'நீங்க யாரு?', 'இது என்ன call?', "
            "'எதுக்கு call பண்றீங்க?'னு கேட்டா — script-ஐ நிறுத்திட்டு, நீங்க யாரு, எதைப் பத்தி "
            "call பண்றீங்கனு சுருக்கமா சொல்லி 'தொடரலாமா?'னு கேளுங்க. சம்மதிச்ச பிறகே கேள்விகளை தொடருங்க.\n"
            "2. தற்போதைய கேள்விக்கு பதில் வந்த பிறகே அடுத்த கேள்விக்கு போங்க. கேள்வி, புகார், "
            "'கேக்கலை' — இது பதில் இல்ல; அதை கவனிச்சு, மறுபடி கேளுங்க.\n"
            "3. அழைப்பவர் 'இல்ல', 'தப்பு', 'அது சரியில்ல'னு சொன்னா — நிறுத்துங்க. மன்னிப்பு கேட்டு "
            "மறுபடி கேளுங்க. உங்க தப்பான பதிலை மறுபடி சொல்லாதீங்க; அடுத்த கேள்விக்கு போகாதீங்க.\n"
            "4. உறுதி செய்யும்போது அழைப்பவர் சொன்ன அதே மதிப்பை அப்படியே சொல்லுங்க — எண், அளவு, "
            "பெயர் எதையும் மாத்தாதீங்க (8 = 8, 7 இல்ல).\n"
            "5. அழைப்பவர் எரிச்சல்படா இருந்தா — புரிஞ்சுக்கிட்டு, மன்னிப்பு கேட்டு, தொடரலாமானு கேளுங்க.\n"
            "6. ஆடியோ சரியில்லைனா / மறுபடி சொல்லச் சொன்னா — மன்னிப்பு கேட்டு கடைசி கேள்வியை மறுபடி சொல்லுங்க.\n"
            "7. அழைப்பவர் சொன்னதை திருத்தினா (முன்ன சொன்னதுக்கு மாறா) — புது மதிப்பையே எடுத்துக்கங்க, "
            "'புரிஞ்சது, [புது மதிப்பு] — நன்றி'னு உறுதி செய்யுங்க.\n"
            "8. 'ஹலோ?', 'இருக்கீங்களா?'னு கேட்டா — முழு அறிமுகத்தையும் மறுபடி சொல்லாதீங்க; "
            "'ஆமா, கேக்குது சொல்லுங்க'னு சொல்லி எங்க நிறுத்தினீங்களோ அங்கருந்து தொடருங்க.\n"
            "9. ROLE — மிக முக்கியம்: இது நீங்க பண்ற OUTBOUND call; அழைப்பவர் உங்களை "
            "கூப்பிடல, நீங்க தான் அவங்களை கூப்பிட்டீங்க. அவங்க 'யாரு பேசறது?'னு கேட்டா "
            "உங்க அறிமுகத்தை ஒரு வரில மறுபடி சொல்லுங்க. ஒருபோதும் அவங்ககிட்ட 'நீங்க யாரு, "
            "எதுக்கு call பண்ணீங்க'னு திருப்பி கேக்கவே கூடாது — அது உங்க role இல்ல.\n"
            "\n"
            "பெயர் & email — மிக முக்கியம் (பேசும்போது 'email'னே சொல்லுங்க, 'மின்னஞ்சல்' வேண்டாம்):\n"
            "1. பெயரை யூகிக்காதீங்க. அழைப்பவர் பெயரை சொன்னதும் ஒரு தடவை திரும்ப சொல்லி confirm பண்ணுங்க: "
            "'புரிஞ்சது, பிரியா — சரியா?'. சரின்னா அடுத்த கேள்விக்கு போங்க; spelling கேக்காதீங்க.\n"
            "2. English பெயர்/email-ஐ spell பண்ணும்போது English எழுத்துகளையே பயன்படுத்துங்க. "
            "புரியாத எழுத்துக்கு 'B as in Bombay, V as in Victor, S as in Sugar' மாதிரி சொல்லுங்க.\n"
            "3. email: '@' = 'at the rate', '.' = 'dot'. டொமைனை முழு வார்த்தையா சொல்லுங்க "
            "('gmail dot com'); @-க்கு முன் உள்ள பகுதியை மட்டும் எழுத்து எழுத்தா confirm பண்ணுங்க. "
            "எதையும் auto-complete பண்ணாதீங்க.\n"
            "\n"
            "எண், தேதி, பணம்:\n"
            "- ஃபோன் நம்பர் / OTP-ஐ ஒவ்வொரு இலக்கமா, இடைவெளி விட்டு சொல்லுங்க (பூஜ்ஜியம், ஒன்னு, "
            "ரெண்டு, மூணு, நாலு, அஞ்சு, ஆறு, ஏழு, எட்டு, ஒம்பது). சொல்லி திரும்ப உறுதி செய்யுங்க.\n"
            "- பணம் இந்திய முறையில (லட்சம், கோடி) — 'million/billion' வேண்டாம்.\n"
            "- தேதி: நாள், மாதம் (தமிழ்ல), வருஷம் வரிசையில.\n"
            "\n"
            "எந்த பெயர்/எண்/மின்னஞ்சல்/தொகை/தேதியையும் சொன்ன பிறகு 'சரியா?'னு உறுதி கேளுங்க. "
            "திருத்தினா அந்த ஒரு விஷயத்தை மட்டும் சரிசெய்யுங்க.\n"
            "\n"
            "உரையாடல் ஓட்டம்:\n"
            f"- எல்லா {n} கேள்விகளுக்கும் பதில் வந்து உறுதியான பிறகு, சேகரிச்ச எல்லாத்தையும் சுருக்கமா "
            "recap பண்ணி 'எல்லாம் சரியா?'னு கேளுங்க.\n"
            "- கால் முடிக்கிறது: எல்லா கேள்விகளும் முடிஞ்சு உறுதியானப்புறம், அல்லது அழைப்பவர் தெளிவா "
            "முடிக்கணும்னு சொன்னப்புறம் மட்டும் — சரியா இந்த வாக்கியத்தை சொல்லி நிறுத்துங்க: "
            "'உங்க நேரத்துக்கு நன்றிங்க. நல்ல நாளா இருக்கட்டும், வணக்கம்!' வெறும் 'பை/நன்றி' சொன்னா, "
            "கேள்விகள் இன்னும் பாக்கி இருந்தா கால்ல முடிக்காதீங்க — மறுபடி தற்போதைய கேள்வியை கேளுங்க.\n"
            "- goodbye சொன்னப்புறம் முழுசா நிறுத்திடுங்க — அழைப்பவரோட 'பை/நன்றி'க்கு மறுபடி பதில் சொல்லாதீங்க."
        )

        # Live-call 2026-07-11 failure modes: the model faked understanding of
        # garbled 8kHz audio ("சரி, புரியுது" to noise), guessed names it never
        # heard ("Priya"/"Harish" for ஹாஜிக்), and re-read full digit strings
        # every turn. These rules target exactly those behaviors.
        if not is_mini:
            parts.append(
            "Phone line reality — மிக மிக முக்கியம்:\n"
            "1. இது 8kHz phone line — அழைப்பவர் பேச்சு அடிக்கடி தெளிவா வராது. சொன்னது "
            "புரியலைனா புரிஞ்ச மாதிரி நடிக்கவே கூடாது — 'சரி', 'புரியுது', 'நன்றி'னு "
            "சொல்லிட்டு அடுத்தது போகாதீங்க. 'Sorry-ங்க, சரியா கேக்கலை — இன்னொரு தடவை "
            "சொல்லுங்களேன்'னு நேரடியா கேளுங்க. புரியாம acknowledge பண்றது தான் call-ஐ "
            "unstable-ஆ காட்டுது.\n"
            "2. பேரு தெளிவா கேக்கலைனா எந்த பேரையும் யூகிச்சு confirm பண்ணவே கூடாது — "
            "தப்பான பேரை ('Priya?', 'Harish?') சொல்றது அழைப்பவரை கடுப்பாக்கும். Doubt "
            "இருந்தா 'உங்க பேரை spell பண்ணி சொல்லுங்களேன்'னு கேளுங்க; அவங்க சொன்ன "
            "letters-ஐ மட்டும் use பண்ணுங்க. ஒரு தடவை தப்புன்னு சொல்லிட்டாங்கனா, அதே "
            "பேரை இன்னொரு தடவை சொல்லாதீங்க.\n"
            "3. ஒரு number/spelling-ஐ ஒரு தடவை confirm பண்ணா போதும் — அதையே திரும்ப "
            "திரும்ப முழுசா படிக்காதீங்க. அழைப்பவர் 'இதே number'னு சொன்னா digits "
            "படிக்காம 'சரிங்க, இதே number-க்கு அனுப்பறேன்'னு சொல்லுங்க.\n"
            "4. Acknowledgment ('சரி', 'புரிஞ்சது') — அழைப்பவர் உண்மையா ஏதாவது தகவல் "
            "சொன்னா மட்டும். அவங்க கேள்வி கேட்டா acknowledgment இல்லாம நேரடியா "
            "பதிலோட ஆரம்பியுங்க.\n"
            "5. பேரை நீங்களா spell பண்ணி (C-A-L-A-I...) திருப்பி சொல்லவே கூடாது — "
            "பேரை ஒரு சாதாரண வார்த்தையா தான் சொல்லி confirm பண்ணுங்க ('கலைவாணி, "
            "சரியா?'). Spelling வேணும்னா அழைப்பவரை கேளுங்க; நீங்க letters-ஆ "
            "படிக்காதீங்க. Confirm ஆன பேரை மறுபடி எடுத்து spell பண்ணாதீங்க.\n"
            "6. எண் தெளிவா ஒரே value-ஆ வரலைனா ('thousand fifteen hundred' மாதிரி "
            "குழப்பமான சொல்) — நீங்க ஒரு number-ஐ (12,500 மாதிரி) கண்டுபிடிச்சு "
            "confirm பண்ணாதீங்க. 'Sorry-ங்க, அது 1500-ஆ? இன்னொரு தடவை சொல்லுங்களேன்' "
            "னு கேளுங்க. தப்பான எண்ணை confirm பண்றது ரொம்ப மோசம்."
        )

        if self._called_number:
            parts.append(
                f"நீங்க call பண்ண number: {self._called_number}. அழைப்பவர் 'இதே "
                "number', 'இந்த number-க்கே அனுப்புங்க'னு சொன்னா அது இந்த number "
                "தான் — digits-ஐ யூகிக்காதீங்க, வேற number சொல்லாதீங்க. தேவைனா "
                "இதையே மெதுவா படிச்சு confirm பண்ணுங்க."
            )

        # Brevity — user feedback 2026-07-11 ("romba pesudhu"): live turns ran
        # 3-4 sentences with stacked filler acknowledgments. Enforced two ways:
        # this rule + the hard max_output_tokens cap (_max_output_tokens).
        if not is_mini:
            parts.append(
                "சுருக்கம் — கண்டிப்பான விதி: ஒரு reply = அதிகபட்சம் 1-2 சின்ன "
                "வாக்கியம். ஒரே அர்த்தத்தை ரெண்டு தடவை சொல்லாதீங்க — 'சரி, புரிஞ்சது. "
                "நான் note பண்ணிக்கறேன். சொல்லுங்க' மாதிரி filler-களை அடுக்காதீங்க; "
                "ஒண்ணே போதும். கேள்வி கேக்கும்போது preamble இல்லாம நேரடியா கேளுங்க. "
                "Recap நேரத்துல மட்டும் தேவையான அளவு சொல்லலாம் — மத்த நேரம் எல்லாம் "
                "குறைவா, தெளிவா."
            )
        else:
            # Condensed rule set for mini — the six failures it actually makes,
            # nothing else to dilute them.
            parts.append(
                "முக்கிய விதிகள் (கண்டிப்பா):\n"
                "1. பேரு/எண்/email தெளிவா கேக்கலைனா யூகிக்கவே கூடாது — தப்பான "
                "பேரை confirm பண்றது பெரிய தப்பு. 'Sorry-ங்க, சரியா கேக்கலை — "
                "spell பண்ணி சொல்லுங்களேன்'னு கேளுங்க. புரியாத பேச்சுக்கு "
                "'புரியுது/சரி'னு நடிக்காதீங்க — மறுபடி கேளுங்க. எண் ஒரே value-ஆ "
                "தெளிவா வரலைனா (உ.ம். 'thousand fifteen hundred') நீங்களா ஒரு "
                "number கண்டுபிடிக்காம 'அது 1500-ஆ? மறுபடி சொல்லுங்க'னு கேளுங்க.\n"
                "2. இது நீங்க பண்ற outbound call. 'ஏன் call பண்ணீங்க, எதுக்கு "
                "பேசறோம்'னு அழைப்பவர்கிட்ட ஒருபோதும் கேக்காதீங்க — அவங்க 'யாரு "
                "பேசறது?'னு கேட்டா உங்க அறிமுகத்தை ஒரு வரில மறுபடி சொல்லுங்க.\n"
                "3. அழைப்பவர் கேள்விக்கு முதல்ல பதில் சொல்லுங்க; தற்போதைய script "
                "கேள்விக்கு பதில் வந்த பிறகே அடுத்த கேள்விக்கு போங்க.\n"
                "4. ஒரு விஷயத்தை ஒரு தடவை confirm பண்ணா போதும் — திரும்ப திரும்ப "
                "முழுசா படிக்காதீங்க. எண்களை ஒவ்வொரு இலக்கமா சொல்லுங்க; email-ல "
                "'@' = 'at the rate', '.' = 'dot'.\n"
                "5. அழைப்பவர் spell பண்ணா மட்டும் அந்த letters-ஐ அப்படியே திருப்பிச் "
                "சொல்லி 'சரிதானே?'னு கேளுங்க. நீங்களா ஒரு பேரை spell பண்ணி "
                "(C-A-L-A-I மாதிரி) சொல்லவே கூடாது — பேரை சாதாரண வார்த்தையா தான் "
                "சொல்லுங்க ('கலைவாணி, சரியா?'). Confirm ஆன பேரை மறுபடி தொடாதீங்க.\n"
                "6. எல்லா கேள்வியும் முடிஞ்சதும் சுருக்கமா recap பண்ணி confirm "
                "பண்ணுங்க. அப்புறம் சரியா இதைச் சொல்லி நிறுத்துங்க: 'உங்க "
                "நேரத்துக்கு நன்றிங்க. நல்ல நாளா இருக்கட்டும், வணக்கம்!'"
            )

        # Final hard lock — placed last on purpose (models weight the end heavily).
        # Must reinforce TANGLISH, not "every word Tamil" (the old wording here
        # overpowered the style guide and produced bookish pure Tamil on live calls).
        if is_mini:
            parts.append(
                "நினைவில் வைங்க: இயல்பான Tanglish மட்டும் (தமிழ் ஓட்டம் + English "
                "வார்த்தைகள்); 1-2 சின்ன வாக்கியம் மட்டும்; பேரு தெளிவா கேக்கலைனா "
                "spell கேளுங்க — ஒருபோதும் யூகிக்காதீங்க."
            )
            return "\n\n".join(parts)
        parts.append(
            "மறுபடியும் நினைவில் வைங்க — இது தான் உங்க voice: நிதானமான matured native "
            "Tamil பேசுறவர்; வாக்கியம் எப்பவும் தமிழ் ஓட்டத்துல, daily English வார்த்தைகள் "
            "(appointment, confirm, name, email, ok, sorry) இயல்பா English-ல கலந்து "
            "— அது தான் சரியான Tanglish. முழு வாக்கியத்தையும் English-ல மட்டும் பேசாதீங்க; "
            "மேல உள்ள 'English only' மாதிரி எந்த வரியும் இங்க பொருந்தாது. அதே மாதிரி "
            "செந்தமிழ் / புத்தகத் தமிழ் / 'தயவுசெய்து', 'மன்னிக்கவும்' மாதிரி formal "
            "வார்த்தைகளும் வேண்டாம் — 'கொஞ்சம்', 'sorry-ங்க' மாதிரி பேச்சு வழக்கே போதும். "
            "எல்லாத்துக்கும் மேல: குறைவா பேசுங்க — 1-2 சின்ன வாக்கியம், அவ்வளவே."
        )

        return "\n\n".join(parts)

    def _realtime_voice(self) -> str:
        """OpenAI native speech-to-speech voice for the current language."""
        return Config.REALTIME_VOICE_TA if self._language == "ta" else Config.REALTIME_VOICE

    def _transcription_config(self) -> dict:
        """Realtime input-audio transcription block for the current language.

        Tamil calls deliberately do NOT pin language: Tanglish callers switch to
        English for names/spellings, and a "ta" pin transliterated those into
        Tamil-script gibberish on a live call (the model then hallucinated names
        and answers from it). Auto-detect + a Tanglish bias prompt keeps English
        segments in English script.
        """
        if self._language == "ta":
            cfg = {"model": Config.TRANSCRIPTION_MODEL_TA}
            if Config.TRANSCRIPTION_LANGUAGE_TA:
                cfg["language"] = Config.TRANSCRIPTION_LANGUAGE_TA
            if Config.TRANSCRIPTION_PROMPT_TA:
                cfg["prompt"] = Config.TRANSCRIPTION_PROMPT_TA
            return cfg
        cfg = {"model": Config.TRANSCRIPTION_MODEL, "language": "en"}
        if Config.TRANSCRIPTION_PROMPT:
            cfg["prompt"] = Config.TRANSCRIPTION_PROMPT
        return cfg

    def _preferred_turn_detection_config(self) -> dict:
        if Config.VAD_TYPE == "semantic_vad":
            return {
                "type": "semantic_vad",
                "eagerness": Config.SEMANTIC_VAD_EAGERNESS,
                "create_response": Config.VAD_CREATE_RESPONSE,
                "interrupt_response": Config.VAD_INTERRUPT_RESPONSE,
            }
        return {
            "type": "server_vad",
            "threshold": Config.VAD_THRESHOLD,
            "prefix_padding_ms": Config.VAD_PREFIX_PADDING_MS,
            "silence_duration_ms": Config.VAD_SILENCE_DURATION_MS,
            "create_response": Config.VAD_CREATE_RESPONSE,
            "interrupt_response": Config.VAD_INTERRUPT_RESPONSE,
        }

    def _tts_max_chars(self) -> int:
        """Max chars per TTS chunk for the current language (Tamil is denser → smaller)."""
        return Config.TTS_MAX_CHARS_TA if self._language == "ta" else Config.TTS_MAX_CHARS

    def _max_output_tokens(self) -> int:
        """Per-turn output cap. Good Tamil turns measured 20-70 tokens
        (2026-07-11); 400 let the model ramble 4-sentence replies ("romba
        pesudhu" feedback). 180 caps the ramble with ~2.5x headroom over the
        longest good turn (recap). English script path stays at its proven 400.
        """
        if self._script and self._language == "ta":
            return 180
        return 400 if self._script else 150

    async def _configure_session(self) -> bool:
        """Configure OpenAI session — g711_ulaw format (zero-conversion for Vobiz mulaw).

        Returns True if session config was confirmed, False on failure.
        """
        # Fast path: session was already configured during pre-warm. The pre-warm
        # (main.py) sends the full session.update for BOTH modes — including the
        # ElevenLabs text/audio-pcm output format — so we can skip the redundant
        # re-config + up-to-5s re-wait and just confirm the buffered session.updated.
        if self._session_preconfigured:
            logger.info("Session pre-configured during pre-warm, waiting for confirmation...")
            temperature = 0.6 if self._script else 0.8
            max_tokens = self._max_output_tokens()
            confirmed = await self._wait_for_session_confirmation(
                Config.VAD_TYPE, temperature, max_tokens
            )
            if confirmed:
                try:
                    await self.openai_ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
                except Exception:
                    pass
                return True
            logger.warning("Pre-warm session config not confirmed, re-configuring...")

        output_modality = "text" if self.use_elevenlabs else "audio"

        prompt = self._build_prompt()
        logger.info(f"System prompt ({len(prompt)} chars): {prompt[:100]}...")

        temperature = 0.6 if self._script else 0.8
        max_tokens = self._max_output_tokens()

        # Try preferred VAD first, then server VAD. If a deployment/model rejects
        # cost-bounding truncation, make one final server-VAD attempt without it so
        # the call still connects.
        vad_configs = []
        if Config.VAD_TYPE == "semantic_vad":
            vad_configs.append({
                "type": "semantic_vad",
                "eagerness": Config.SEMANTIC_VAD_EAGERNESS,
                "create_response": Config.VAD_CREATE_RESPONSE,
                "interrupt_response": Config.VAD_INTERRUPT_RESPONSE,
            })
        server_vad = {
            "type": "server_vad",
            "threshold": Config.VAD_THRESHOLD,
            "prefix_padding_ms": Config.VAD_PREFIX_PADDING_MS,
            "silence_duration_ms": Config.VAD_SILENCE_DURATION_MS,
            "create_response": Config.VAD_CREATE_RESPONSE,
            "interrupt_response": Config.VAD_INTERRUPT_RESPONSE,
        }
        vad_configs.append(server_vad)

        truncation_config = {
            "type": "retention_ratio",
            "retention_ratio": Config.REALTIME_RETENTION_RATIO,
            "token_limits": {
                "post_instructions": Config.REALTIME_HISTORY_TOKEN_LIMIT,
            },
        }
        attempts = [(vad, truncation_config) for vad in vad_configs]
        attempts.append((server_vad, None))

        for vad_config, truncation in attempts:
            vad_type = vad_config["type"]
            output_format_obj = (
                {"type": "audio/pcm", "rate": 24000} if self.use_elevenlabs
                else {"type": "audio/pcmu"}
            )
            session = {
                "type": "realtime",
                "output_modalities": [output_modality],
                "instructions": prompt,
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcmu"},
                        "turn_detection": vad_config,
                        "transcription": self._transcription_config(),
                        "noise_reduction": {"type": "near_field"},
                    },
                    "output": {
                        "format": output_format_obj,
                        "voice": self._realtime_voice(),
                    },
                },
                "max_output_tokens": max_tokens,
            }
            if truncation:
                session["truncation"] = truncation
            session_config = {"type": "session.update", "session": session}
            cost_mode = "bounded-history" if truncation else "compatibility"
            print(
                f"[CALL] Sending session config — input=audio/pcmu, "
                f"output={output_format_obj['type']}, {vad_type}, {cost_mode}",
                flush=True,
            )
            await self.openai_ws.send(json.dumps(session_config))

            confirmed = await self._wait_for_session_confirmation(
                vad_type, temperature, max_tokens
            )
            if confirmed:
                try:
                    await self.openai_ws.send(json.dumps({
                        "type": "input_audio_buffer.clear",
                    }))
                except Exception:
                    pass
                return True

            logger.warning(
                "Session config rejected for vad=%s truncation=%s; trying fallback",
                vad_type,
                bool(truncation),
            )

        logger.error("ALL session config attempts failed — audio format is NOT g711_ulaw!")
        return False

    async def _wait_for_session_confirmation(
        self, vad_type: str, temperature: float, max_tokens: int
    ) -> bool:
        """Wait for session.updated confirmation from OpenAI."""
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                raw = await asyncio.wait_for(self.openai_ws.recv(), timeout=remaining)
                event = json.loads(raw)
                etype = event.get("type", "")
                if etype == "session.updated":
                    session = event.get("session", {})
                    audio_cfg = session.get("audio", {}) or {}
                    in_fmt_obj = (audio_cfg.get("input", {}) or {}).get("format", {}) or {}
                    out_fmt_obj = (audio_cfg.get("output", {}) or {}).get("format", {}) or {}
                    in_fmt = in_fmt_obj.get("type", "unknown") if isinstance(in_fmt_obj, dict) else "unknown"
                    out_fmt = out_fmt_obj.get("type", "unknown") if isinstance(out_fmt_obj, dict) else "unknown"
                    expected_out = "audio/pcm" if self.use_elevenlabs else "audio/pcmu"
                    print(
                        f"[CALL] session.updated -> input={in_fmt}, output={out_fmt} "
                        f"(expected={expected_out})",
                        flush=True,
                    )
                    if out_fmt != expected_out:
                        print(
                            f"[CALL] *** FORMAT MISMATCH! Got output={out_fmt} "
                            f"instead of {expected_out} — retrying ***",
                            flush=True,
                        )
                        expected_out_obj = (
                            {"type": "audio/pcm", "rate": 24000} if self.use_elevenlabs
                            else {"type": "audio/pcmu"}
                        )
                        await self.openai_ws.send(json.dumps({
                            "type": "session.update",
                            "session": {
                                "type": "realtime",
                                "output_modalities": ["text" if self.use_elevenlabs else "audio"],
                                "audio": {
                                    "input": {"format": {"type": "audio/pcmu"}},
                                    "output": {"format": expected_out_obj},
                                },
                            },
                        }))
                        retry_raw = await asyncio.wait_for(
                            self.openai_ws.recv(), timeout=3.0
                        )
                        retry_event = json.loads(retry_raw)
                        if retry_event.get("type") == "session.updated":
                            retry_audio = retry_event.get("session", {}).get("audio", {}) or {}
                            retry_out = (retry_audio.get("output", {}) or {}).get("format", {}) or {}
                            retry_fmt = retry_out.get("type", "unknown") if isinstance(retry_out, dict) else "unknown"
                            if retry_fmt == expected_out:
                                print(f"[CALL] Format corrected to {expected_out} on retry", flush=True)
                            else:
                                print(f"[CALL] *** STILL WRONG FORMAT: {retry_fmt} — aborting ***", flush=True)
                                return False
                        else:
                            print("[CALL] *** Retry did not return session.updated — aborting ***", flush=True)
                            return False

                    mode = f"{self.tts_provider} TTS" if self.use_elevenlabs else "speech-to-speech"
                    script_mode = "script" if self._script else "free"
                    print(
                        f"[CALL] Session CONFIRMED — {mode}, {script_mode}, "
                        f"in={in_fmt}, out={out_fmt}, {vad_type}, "
                        f"temp={temperature}, max={max_tokens}tok",
                        flush=True,
                    )
                    return True
                elif etype == "error":
                    error = event.get("error", {})
                    print(
                        f"[CALL] *** Session REJECTED ({vad_type}): "
                        f"{error.get('message', 'unknown')} | code={error.get('code', '?')} ***",
                        flush=True,
                    )
                    return False
                else:
                    logger.debug(f"Pre-config event: {etype}")
        except asyncio.TimeoutError:
            logger.error(f"Session config timed out (5s) for {vad_type}")
        return False

    # ── Echo gate ───────────────────────────────────────────────────────────

    def _is_echo_gate_active(self) -> bool:
        # Welcome must be fully interruptible — let all caller audio reach OpenAI
        if self._is_first_response:
            return False
        if self._ai_is_responding or self._tts_playing:
            return True
        if time.monotonic() < self._estimated_playback_end + self._ECHO_COOLDOWN:
            return True
        if time.monotonic() < self._echo_gate_until:
            return True
        return False

    async def _forward_audio(self, mulaw_b64: str):
        """Forward Vobiz mulaw 8kHz directly to OpenAI (g711_ulaw — zero conversion).

        Gated frames are SUBSTITUTED with mu-law silence instead of dropped.
        Dropping them starved OpenAI's server VAD of the silence it needs to
        emit speech_stopped: while our TTS was playing (gate active), a caller
        going quiet was simply never heard, `_interrupt_pending` wedged, and
        the agent went silent (observed live 2026-07-10 — speech_stopped never
        arrived on ANY turn where the gate was active). Substituting silence
        keeps OpenAI's audio timeline continuous while still suppressing echo.
        """
        try:
            silence_len = 0
            if self._is_echo_gate_active():
                mulaw_bytes = base64.b64decode(mulaw_b64)
                pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
                rms = audioop.rms(pcm_8k, 2)

                if rms < Config.ECHO_GATE_RMS_HARD:
                    silence_len = len(mulaw_bytes)   # Silence or quiet echo
                else:
                    playback_active = (
                        self._ai_is_responding
                        or time.monotonic() < self._estimated_playback_end
                    )
                    if rms < Config.ECHO_GATE_RMS_SOFT and playback_active:
                        # Likely echo during active AI/TTS playback
                        silence_len = len(mulaw_bytes)

            ws = self.openai_ws
            if not ws:
                return
            if silence_len:
                # 0xFF = mu-law encoding of linear 0 (true silence).
                mulaw_b64 = base64.b64encode(b"\xff" * silence_len).decode("utf-8")
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": mulaw_b64,
            }))
        except Exception as e:
            logger.error(f"Error forwarding audio: {e}")

    # ── OpenAI event processing ─────────────────────────────────────────────

    def _measure_speech_duration_ms(self, stopped_event: dict) -> float:
        """How long the caller ACTUALLY spoke, for the backchannel ruling.

        Wall-clock (speech_stopped arrival − speech_started arrival) is the wrong
        quantity: server VAD only emits speech_stopped after observing
        VAD_SILENCE_DURATION_MS of silence, so wall-clock is
        true_speech + silence_window + event RTT. With the production settings
        (silence 500ms, backchannel cap 600ms) that left ~100ms of real budget —
        a genuine 150ms "ம்ம்" measured ~650ms and was ruled a full interruption,
        cutting the agent off mid-sentence on every acknowledgement.

        OpenAI reports buffer-relative audio_start_ms / audio_end_ms, which are
        more precise than wall-clock but are NOT the raw speech boundaries: per
        the Realtime event reference audio_start_ms "includes the
        prefix_padding_ms configured in the Session" and audio_end_ms "includes
        the silence_duration_ms". Their difference therefore over-reports by
        prefix + silence (620ms live) and must have both subtracted — taking the
        span raw leaves the same 150ms hum measuring 770ms and changes nothing.

        Fall back to wall-clock (minus the silence window) when the fields are
        absent, and also when the subtraction goes negative: that would mean the
        events are not padded as documented, and clamping to zero instead would
        rule EVERY turn a backchannel and silently disable barge-in altogether.
        """
        audio_end_ms = stopped_event.get("audio_end_ms")
        audio_start_ms = self._speech_start_audio_ms
        if isinstance(audio_end_ms, (int, float)) and isinstance(audio_start_ms, (int, float)):
            span_ms = float(audio_end_ms) - float(audio_start_ms)
            true_ms = span_ms - Config.VAD_PREFIX_PADDING_MS - Config.VAD_SILENCE_DURATION_MS
            if true_ms >= 0:
                return true_ms
        wall_ms = (time.monotonic() - self._speech_start_time) * 1000
        return max(0.0, wall_ms - Config.VAD_SILENCE_DURATION_MS)

    async def _receive_openai_events(self):
        """Receive events from OpenAI and forward audio to Vobiz."""
        ws = self.openai_ws
        if not ws:
            return
        try:
            async for raw in ws:
                try:
                    event = json.loads(raw)
                    await self._handle_openai_event(event)
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.error(f"Error handling OpenAI event: {e}", exc_info=True)
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"OpenAI connection closed: {e}")
        except asyncio.CancelledError:
            logger.info("OpenAI receive task cancelled")
        except Exception as e:
            logger.error(f"Error receiving OpenAI events: {e}")
        finally:
            if ws is not self.openai_ws:
                # The socket this loop owned was swapped out from under it —
                # i.e. _restart_session ("start over") closed it deliberately and
                # a replacement receive task is already live on a new socket.
                # Reading _connected here would see the NEW connection's True and
                # close the caller's media stream; clearing it would deafen the
                # bridge for the rest of the call. Identity is self-clearing, so
                # unlike a flag it cannot leak and stays correct for N restarts.
                logger.info("Old receive task exited after session restart")
                return
            was_active = self._connected
            self._connected = False
            if (
                was_active
                and not self._cleanup_started
                and Config.CLOSE_CALL_ON_REALTIME_DISCONNECT
            ):
                logger.error(
                    "OpenAI Realtime disconnected unexpectedly; closing the media "
                    "stream to avoid leaving the caller in silence"
                )
                await self._close_vobiz_on_failure("OpenAI Realtime disconnected")

    async def _send_audio_to_vobiz(self, mulaw_b64: str):
        """Send base64-encoded mulaw audio to Vobiz via playAudio event."""
        ws = self._vobiz_ws
        if not ws or not self.stream_id:
            return
        await ws.send_json({
            "event": "playAudio",
            "media": {
                "contentType": "audio/x-mulaw",
                "sampleRate": 8000,
                "payload": mulaw_b64,
            },
        })

    async def _clear_vobiz_audio(self):
        """Send clearAudio event to Vobiz to stop any queued audio playback."""
        # Before wiping the playback estimate, record what the caller actually
        # got to hear — the interruption recovery relies on this split.
        self._snapshot_playback_cut()
        self._estimated_playback_end = 0.0
        ws = self._vobiz_ws
        if not ws:
            return
        try:
            await ws.send_json({"event": "clearAudio"})
        except Exception as e:
            logger.debug(f"clearAudio failed: {e}")

    # GA Realtime renamed some response.* events. Normalize back to Beta names
    # so the rest of the handler can stay unchanged.
    _GA_EVENT_ALIAS = {
        "response.output_audio.delta": "response.audio.delta",
        "response.output_audio.done": "response.audio.done",
        "response.output_audio_transcript.delta": "response.audio_transcript.delta",
        "response.output_audio_transcript.done": "response.audio_transcript.done",
        "response.output_text.delta": "response.text.delta",
        "response.output_text.done": "response.text.done",
    }

    async def _handle_openai_event(self, event: dict):
        """Process an event from OpenAI Realtime API."""
        t = event.get("type", "")
        t = self._GA_EVENT_ALIAS.get(t, t)

        if t == "session.created":
            logger.info("OpenAI session created")

        elif t == "session.updated":
            logger.info("OpenAI session updated")

        elif t == "response.created":
            self._ai_is_responding = True
            self._first_audio_chunk = True
            self._audio_chunk_count = 0
            self._response_start_time = time.monotonic()
            self._current_response_item_id = None
            self._response_audio_sent_ms = 0.0
            self._response_playback_started_at = 0.0
            self._pcm_buffer = b""
            # New response → the next flushed piece is the turn's first audio.
            self._first_piece_flushed = False
            self._openers_seen_this_response = False
            self._heard_text_this_response = ""
            self._sent_pieces = []
            self._unheard_text_this_response = ""
            self._response_audio_started_at = 0.0
            self._last_response_text = ""
            self._discard_response_text = False
            # Any text still in the TTS buffer belongs to a PREVIOUS response
            # that a cancel/suppression withheld — if it leaks into this one,
            # two responses merge into a single garbled TTS chunk (live
            # 2026-07-11: "What is your current CTC?I'm sorry,").
            self._response_text_buffer = ""
            if (self.use_elevenlabs and not self._is_first_response
                    and not self._any_real_audio_sent
                    and self._pre_welcome_suppressions < 2):
                # The caller spoke into the pre-welcome silence ("hello?") and
                # server VAD auto-created this reply — but they haven't heard
                # the WELCOME yet; the welcome IS the answer. Cancel this reply
                # and discard its text so an improvised second greeting never
                # replaces the scripted one (live bug 2026-07-11: the agent
                # skipped its greeting entirely on every call).
                self._pre_welcome_suppressions += 1
                self._discard_response_text = True
                self._suppress_recovery_once = True
                self._cancel_requested = True
                try:
                    ws = self.openai_ws
                    if ws:
                        await ws.send(json.dumps({"type": "response.cancel"}))
                        logger.info(
                            "Suppressed auto-reply to pre-welcome speech — "
                            "the welcome plays instead"
                        )
                except Exception as e:
                    logger.debug(f"Could not suppress pre-welcome reply: {e}")
            elif (self.use_elevenlabs
                    and time.monotonic() < self._suppress_bc_response_until):
                # This response answers a micro-turn we just ruled a BACKCHANNEL
                # ("mm-hm" during agent speech) — server VAD auto-creates it
                # anyway. Answering a backchannel re-asks the question the
                # caller already heard (live 2026-07-11: "May I know your
                # name?" delivered 3x in a row). Cancel + discard.
                self._suppress_bc_response_until = 0.0
                self._discard_response_text = True
                self._suppress_recovery_once = True
                self._cancel_requested = True
                try:
                    ws = self.openai_ws
                    if ws:
                        await ws.send(json.dumps({"type": "response.cancel"}))
                        logger.info(
                            "Suppressed auto-reply to backchannel — prior "
                            "reply already answered the caller"
                        )
                except Exception as e:
                    logger.debug(f"Could not suppress backchannel reply: {e}")

        elif t in ("response.output_item.added", "response.output_item.created"):
            item = event.get("item", {}) or {}
            if item.get("type") == "message" and item.get("role") == "assistant":
                self._current_response_item_id = item.get("id")

        elif t == "response.audio.delta":
            # Audio from OpenAI → forward to Vobiz
            # g711_ulaw mode: already mulaw, forward directly (ZERO conversion!)
            # pcm16 mode (ElevenLabs): convert PCM16 24kHz → mulaw 8kHz
            audio_b64 = event.get("delta", "")
            if audio_b64 and self._vobiz_ws and self.stream_id:
                if self._interrupt_pending:
                    # Backchannel evaluation in progress — buffer instead of
                    # dropping. Dropping made words vanish mid-sentence when the
                    # ruling was "just a mm-hm, keep talking".
                    self._pending_audio_deltas.append(audio_b64)
                    return
                await self._forward_response_audio(audio_b64)

        elif t == "response.audio_transcript.delta":
            self._current_ai_transcript += event.get("delta", "")

        elif t == "response.audio_transcript.done":
            transcript = event.get("transcript", "")
            if transcript:
                logger.info(f"AI said: {transcript[:80]}...")
                await self._emit_transcript("ai", transcript, True)
                # A cancel in flight means the caller never heard this — don't
                # arm a hangup for a goodbye that was cut off, and keep the
                # transcript so the cancelled branch can inject recovery context.
                if self._contains_goodbye(transcript) and not self._cancel_requested:
                    self._schedule_hangup(4.5)
            if not self._cancel_requested:
                self._current_ai_transcript = ""

        elif t == "response.text.delta":
            if self._discard_response_text:
                return  # suppressed pre-welcome reply — never synthesized
            if self.use_elevenlabs:
                self._response_text_buffer += event.get("delta", "")
                await self._flush_sentences()
            self._current_ai_transcript += event.get("delta", "")

        elif t == "response.text.done":
            full_text = event.get("text", "")
            if self._discard_response_text:
                self._response_text_buffer = ""
                return
            if full_text:
                # Remembered so a reply killed before ANY audio went out can
                # still be reported as fully-unheard to the recovery injection.
                self._last_response_text = full_text
            if (self.use_elevenlabs and self._response_text_buffer.strip()
                    and not self._cancel_requested):
                text = self._response_text_buffer.strip()
                self._response_text_buffer = ""
                logger.info(f"[flush-final] {text[:60]}")
                await self._enqueue_tts(text)
            if full_text:
                logger.info(f"AI said: {full_text[:120]}")
                await self._emit_transcript("ai", full_text, True)
                if self._contains_goodbye(full_text) and not self._cancel_requested:
                    self._schedule_hangup(4.5)
            if not self._cancel_requested:
                self._current_ai_transcript = ""

        elif t == "response.done":
            response = event.get("response", {}) or {}
            status = response.get("status", "")
            # Consume the one-shot flag regardless of status so a raced cancel
            # (response finished first) can't suppress a FUTURE real recovery.
            suppress_recovery = self._suppress_recovery_once
            self._suppress_recovery_once = False
            usage = response.get("usage", {}) or {}
            if usage:
                total = int(usage.get("total_tokens", 0) or 0)
                input_tokens = int(usage.get("input_tokens", 0) or 0)
                output_tokens = int(usage.get("output_tokens", 0) or 0)
                self._total_realtime_tokens += total
                self._total_input_tokens += input_tokens
                self._total_output_tokens += output_tokens
                input_details = usage.get("input_token_details", {}) or {}
                cached = int(input_details.get("cached_tokens", 0) or 0)
                self._cached_input_tokens += cached
                # Per-turn cache visibility: cached-vs-fresh input decides a 6x
                # cost swing (cached text $0.06/M vs fresh audio $10/M) — cost
                # audit 2026-07-10 flagged this as the number to watch.
                cached_details = input_details.get("cached_tokens_details", {}) or {}
                logger.info(
                    "Realtime usage turn=%s total=%s input=%s (cached=%s text=%s audio=%s) "
                    "output=%s session_total=%s",
                    self._turn_count,
                    total,
                    input_tokens,
                    cached,
                    cached_details.get("text_tokens", "?"),
                    cached_details.get("audio_tokens", "?"),
                    output_tokens,
                    self._total_realtime_tokens,
                )
            # NOTE: an unevaluated _interrupt_pending is deliberately NOT resolved
            # here. Draining the queue on response.done treated every pending
            # backchannel ("mm-hm" while the model's text finished early) as a
            # real interrupt and silently discarded the rest of the queued reply.
            # speech_stopped owns the backchannel-vs-interrupt ruling.
            if self.use_elevenlabs and not self._interrupt_pending:
                await self._flush_pcm_remainder()
            self._ai_is_responding = False
            # Welcome is now over (completed or cancelled) — mid-call behavior from here on
            if self._is_first_response:
                self._is_first_response = False
                logger.info("Welcome done — echo gate + backchannel guard now active")
            if status == "cancelled":
                saved_transcript = self._current_ai_transcript.strip()
                self._current_ai_transcript = ""
                if not suppress_recovery:
                    # Programmatic cancels (pre-welcome suppression / forced
                    # spelling) must NOT drain: the queue may hold the WELCOME
                    # this cancel exists to protect (spelling pre-drains itself).
                    self._response_text_buffer = ""
                    self._drain_tts_queue()
                    self._pending_audio_deltas.clear()
                self._discard_response_text = False
                logger.info(f"Response cancelled — saved: '{saved_transcript[:60]}'")
                if saved_transcript and len(saved_transcript) > 20 and not suppress_recovery:
                    await self._inject_interrupt_recovery()
            else:
                if self.use_elevenlabs and self._response_text_buffer.strip():
                    # This response COMPLETED but its tail never flushed at
                    # text.done — a cancel that missed its target left
                    # _cancel_requested stale-True. Withholding it cuts the
                    # answer mid-sentence; flush it now (a later interruption
                    # ruling still drains the queue normally).
                    text = self._response_text_buffer.strip()
                    self._response_text_buffer = ""
                    logger.info(f"[flush-done] {text[:60]}")
                    await self._enqueue_tts(text)
                self._current_ai_transcript = ""
                logger.debug(f"Response complete (status={status})")
            self._cancel_requested = False
            if self._response_create_pending:
                self._response_create_pending = False
                if status == "cancelled":
                    # The create for the caller's turn was rejected while the
                    # old response finalized its cancel — fire it now, otherwise
                    # the turn is never answered (dead air, live 2026-07-09).
                    try:
                        ws = self.openai_ws
                        if ws:
                            await ws.send(json.dumps({"type": "response.create"}))
                            logger.info("Retried rejected response.create after response.done")
                    except Exception as e:
                        logger.warning(f"Could not retry response.create: {e}")
                else:
                    # The response that just finalized COMPLETED — the caller's
                    # turn already has its answer. Firing the pending create on
                    # top of it produced a second full answer (re-greeting +
                    # repeated question) stacked in the TTS queue: the "agent
                    # talking over itself" live bug 2026-07-11 01:49.
                    logger.info(
                        f"Dropped pending response.create — turn already "
                        f"answered (status={status})"
                    )

        elif t == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            if transcript:
                self._turn_count += 1
                transcription_usage = event.get("usage", {}) or {}
                self._transcription_tokens += int(
                    transcription_usage.get("total_tokens", 0) or 0
                )
                logger.info(f"User said: \"{transcript}\"")
                await self._emit_transcript("user", transcript, True)
                lower = transcript.lower().strip()

                # ── Post-goodbye decision ────────────────────────────────────
                # We already said goodbye and paused the pending hangup to hear the
                # caller out. A pure courtesy closing ("thanks, bye") lets the call
                # end; anything substantive cancels the hangup so we answer normally.
                if self._awaiting_post_goodbye_eval:
                    self._awaiting_post_goodbye_eval = False
                    if _is_closing_remark(transcript, self._language):
                        logger.info("Post-goodbye closing remark — ending call, not re-engaging")
                        # Suppress the auto-generated re-engagement (mirror welcome barge-in).
                        self._drain_tts_queue()
                        self._response_text_buffer = ""
                        self._pending_audio_deltas.clear()
                        await self._clear_vobiz_audio()
                        if self._ai_is_responding:
                            try:
                                ws = self.openai_ws
                                if ws:
                                    self._cancel_requested = True
                                    await ws.send(json.dumps({"type": "response.cancel"}))
                            except Exception as e:
                                logger.debug(f"Could not cancel post-goodbye response: {e}")
                        if self._hangup_task and not self._hangup_task.done():
                            self._hangup_task.cancel()
                        self._goodbye_detected = False
                        self._schedule_hangup(Config.POST_GOODBYE_HANGUP_DELAY_S)
                        return
                    logger.info("Post-goodbye remark is substantive — answering, hangup released")
                    self._cancel_hangup()
                restart_phrases = ["start over", "start fresh", "restart",
                                   "erase everything", "begin again",
                                   "start from scratch", "reset"]
                if any(phrase in lower for phrase in restart_phrases):
                    logger.info(f"RESTART: Intent detected in '{transcript}'")
                    await self._restart_session()
                    return

                if _is_disengage_intent(transcript, self._language):
                    logger.info(f"DISENGAGE intent in '{transcript[:60]}' — injecting goodbye")
                    if self._language == "ta":
                        disengage_text = (
                            "[அழைப்பவர் இப்போ call-ஐ முடிக்க விரும்பறாங்க. உங்க அடுத்த பதில் "
                            "சரியா இப்படி மட்டும் இருக்கணும்: \"பரவாயில்லைங்க! உங்க நேரத்துக்கு "
                            "ரொம்ப நன்றி. பிறகு வசதியான நேரத்துல பேசலாம், நல்ல நாளா இருங்க!\" "
                            "வேற எந்த கேள்வியும் கேக்காதீங்க. வேற எதுவும் சேர்க்காதீங்க. "
                            "இந்த ஒரு வாக்கியத்தை மட்டும் சொல்லி நிறுத்துங்க.]"
                        )
                    else:
                        disengage_text = (
                            "[The caller wants to end this call now. Your VERY NEXT response "
                            "MUST be exactly: \"No problem at all. Thank you for your time. "
                            "Have a great day! Goodbye.\" "
                            "Do NOT ask any more questions. Do NOT offer to schedule. "
                            "Do NOT add anything else. Just say that one sentence and stop.]"
                        )
                    try:
                        ws = self.openai_ws
                        if ws:
                            await ws.send(json.dumps({
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "message",
                                    "role": "system",
                                    "content": [{
                                        "type": "input_text",
                                        "text": disengage_text,
                                    }],
                                },
                            }))
                            # The server-VAD auto-answer for this turn raced the
                            # injection — it never saw the goodbye script and
                            # free-styles something pushy ("no problem, could you
                            # share your name?", live 2026-07-11 13:01, caller
                            # hung up). Drop its audio and re-create so the
                            # polite goodbye plays instead.
                            self._drain_tts_queue()
                            self._response_text_buffer = ""
                            self._pending_audio_deltas.clear()
                            await self._clear_vobiz_audio()
                            if self._ai_is_responding:
                                self._suppress_recovery_once = True
                                self._cancel_requested = True
                                await ws.send(json.dumps({"type": "response.cancel"}))
                                self._response_create_pending = True
                                logger.info(
                                    "Cancelled in-flight response — re-creating "
                                    "so the goodbye applies now"
                                )
                            else:
                                # Turn already answered (or no response yet) —
                                # without a nudge the goodbye instruction sits
                                # unused until the caller speaks again.
                                await ws.send(json.dumps({"type": "response.create"}))
                    except Exception as e:
                        logger.debug(f"Could not inject disengage instruction: {e}")

                letters = _extract_spelled_letters(transcript)
                if letters:
                    confirmation = _build_forced_confirmation(letters, self._language)
                    try:
                        ws = self.openai_ws
                        if ws:
                            await ws.send(json.dumps({
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "message",
                                    "role": "system",
                                    "content": [{
                                        "type": "input_text",
                                        "text": (
                                            f"[The caller just spelled: {letters}. "
                                            f"Your VERY NEXT response MUST start with EXACTLY this sentence "
                                            f"(verbatim, word-for-word, letter-for-letter): "
                                            f"\"{confirmation}\". "
                                            f"Do NOT substitute any letter. Do NOT reorder letters. "
                                            f"Do NOT use different phonetic words. "
                                            f"After that sentence, stop and wait for the caller's confirmation.]"
                                        ),
                                    }],
                                },
                            }))
                            logger.info(f"Injected forced confirmation: {letters} -> {confirmation[:60]}")
                            if self._ai_is_responding:
                                # The server-VAD auto-created response raced this
                                # transcription — it never saw the forced readback
                                # and free-styles the letters (live call 2026-07-11
                                # 03:26: 15s of WRONG NATO). Cancel it and re-create
                                # so the exact confirmation applies THIS turn.
                                self._drain_tts_queue()
                                self._response_text_buffer = ""
                                self._pending_audio_deltas.clear()
                                await self._clear_vobiz_audio()
                                self._suppress_recovery_once = True
                                self._cancel_requested = True
                                await ws.send(json.dumps({"type": "response.cancel"}))
                                self._response_create_pending = True
                                logger.info(
                                    "Cancelled in-flight response — re-creating so "
                                    "the forced spelling confirmation applies now"
                                )
                    except Exception as e:
                        logger.debug(f"Could not inject forced confirmation: {e}")
            else:
                logger.warning("User audio transcription was EMPTY")

        elif t == "conversation.item.input_audio_transcription.failed":
            error = event.get("error", {})
            logger.error(f"Transcription FAILED: {error.get('message', 'unknown')}")

        elif t == "input_audio_buffer.speech_started":
            # If we've already said goodbye, don't abandon the hangup outright — pause
            # it and wait to see whether the caller is just signing off or has a real
            # follow-up (decided in the transcription handler). Mid-call (no goodbye
            # pending) this stays exactly as before: a no-op cancel.
            if self._goodbye_detected:
                self._pause_hangup()
            else:
                self._cancel_hangup()
            self._speech_start_time = time.monotonic()
            # OpenAI reports where speech actually began in the input buffer.
            # Paired with speech_stopped's audio_end_ms this gives the TRUE
            # speech duration — see the backchannel ruling in speech_stopped.
            self._speech_start_audio_ms = event.get("audio_start_ms")
            # Arm the listening backchannel for this utterance (fires only if
            # the caller is STILL talking after ~4.5s of agent silence).
            if self._listen_bc_task and not self._listen_bc_task.done():
                self._listen_bc_task.cancel()
            self._listen_bc_task = asyncio.create_task(
                self._play_listen_backchannels(self._speech_start_time)
            )
            playback_active = time.monotonic() < self._estimated_playback_end
            if ((self._ai_is_responding or playback_active)
                    and not self._any_real_audio_sent):
                # Caller is talking into the pre-welcome silence (setup takes
                # 2-4s and humans say "hello?" into it). They cannot be
                # interrupting speech they have never heard — leave the welcome
                # pipeline completely untouched.
                logger.info(
                    "Caller spoke before hearing any agent audio — "
                    "not an interrupt; welcome proceeds"
                )
            elif self._ai_is_responding or playback_active:
                if self._is_first_response:
                    # Barge-in on the welcome: cancel immediately, skip backchannel heuristic.
                    logger.info("Caller spoke during welcome — hard interrupt")
                    self._interrupt_pending = True
                    self._spawn_bg(
                        self._resolve_stuck_interrupt(self._speech_start_time)
                    )
                    self._drain_tts_queue()
                    self._response_text_buffer = ""
                    self._pending_audio_deltas.clear()
                    await self._clear_vobiz_audio()
                    if self._ai_is_responding:
                        try:
                            ws = self.openai_ws
                            if ws:
                                self._cancel_requested = True
                                await ws.send(json.dumps({"type": "response.cancel"}))
                        except Exception as e:
                            logger.debug(f"Could not cancel welcome: {e}")
                    await self._truncate_current_audio()
                else:
                    logger.info("User speech during AI response — clearing Vobiz buffer (pending eval)")
                    self._interrupt_pending = True
                    self._spawn_bg(
                        self._gentle_clear(Config.INTERRUPTION_EVAL_DELAY_MS / 1000.0)
                    )
                    # Watchdog: if speech_stopped never arrives (echo-gated
                    # silence / lost VAD event), pending would wedge True and
                    # mute the agent for the REST OF THE CALL (observed live
                    # 2026-07-10 02:07 — 30s of dead air until the caller gave
                    # up). Resolve conservatively after a bound.
                    self._spawn_bg(
                        self._resolve_stuck_interrupt(self._speech_start_time)
                    )

        elif t == "input_audio_buffer.speech_stopped":
            self._speech_stopped_time = time.monotonic()
            # The caller's monologue is over — stop any pending listening hum.
            if self._listen_bc_task and not self._listen_bc_task.done():
                self._listen_bc_task.cancel()
            # Perceived-latency mask: if the real reply hasn't reached the
            # caller shortly, play a neutral pre-synthesized "ம்ம்..." clip.
            self._spawn_bg(self._maybe_play_filler())
            if self._interrupt_pending:
                speech_duration_ms = self._measure_speech_duration_ms(event)
                self._interrupt_pending = False

                if speech_duration_ms < Config.BACKCHANNEL_MAX_DURATION_MS:
                    logger.info(
                        f"Backchannel ({speech_duration_ms:.0f}ms < "
                        f"{Config.BACKCHANNEL_MAX_DURATION_MS}ms) — not draining"
                    )
                    if not self._ai_is_responding:
                        # Server VAD will still auto-create an answer for this
                        # micro-turn; arm a short window so response.created
                        # can suppress it (a backchannel needs no reply — the
                        # prior response already played). Active generation is
                        # left alone: its create gets rejected server-side and
                        # response.done drops the retry for completed answers.
                        self._suppress_bc_response_until = time.monotonic() + 2.0
                    # Words buffered during the evaluation window resume playing.
                    await self._flush_pending_audio()
                else:
                    logger.info(
                        f"Real interruption ({speech_duration_ms:.0f}ms) — cancelling response"
                    )
                    self._interrupt_count += 1
                    self._drain_tts_queue()
                    self._response_text_buffer = ""
                    self._pending_audio_deltas.clear()
                    if self._ai_is_responding:
                        try:
                            ws = self.openai_ws
                            if ws:
                                self._cancel_requested = True
                                await ws.send(json.dumps({"type": "response.cancel"}))
                        except Exception as e:
                            logger.debug(f"Could not cancel response: {e}")
                    elif self.use_elevenlabs and (
                        self._unheard_text_this_response
                        or (not self._sent_pieces and self._last_response_text)
                    ):
                        # External TTS: the response TEXT finished long before its
                        # AUDIO (text ~1s, audio 5-8s), so this barge-in killed
                        # playback of an already-COMPLETED response — there is no
                        # response.done(cancelled) to carry the recovery, and the
                        # model believes the unheard part (greeting/question) was
                        # delivered → it skips ahead (live bug 2026-07-11: greeting
                        # and questions vanished after every "hello" barge-in).
                        # Inject the heard/unheard correction, then cancel the
                        # server-VAD auto-created answer (it raced the injection)
                        # and re-create it so it sees the correction.
                        if not self._sent_pieces and not self._unheard_text_this_response:
                            # Killed before ANY audio went out — the whole reply
                            # is unheard.
                            self._unheard_text_this_response = self._last_response_text
                            self._heard_text_this_response = ""
                        await self._inject_interrupt_recovery()
                        self._suppress_recovery_once = True
                        try:
                            ws = self.openai_ws
                            if ws:
                                self._cancel_requested = True
                                await ws.send(json.dumps({"type": "response.cancel"}))
                                self._response_create_pending = True
                        except Exception as e:
                            logger.debug(f"Could not cancel auto-created response: {e}")
                    await self._truncate_current_audio()

        elif t == "error":
            error = event.get("error", {})
            code = error.get("code", "?")
            if code == "conversation_already_has_active_response":
                # Server-VAD auto-create raced our response.cancel (the old
                # response hadn't finalized). Retry at response.done, otherwise
                # the caller's turn is never answered — dead air until they
                # speak again (observed live 2026-07-09 17:51:15).
                self._response_create_pending = True
                logger.info(
                    "response.create rejected (previous response still active) "
                    "— will retry at response.done"
                )
            elif code == "response_cancel_not_active":
                # Benign race: the response finished before our cancel arrived.
                logger.debug("response.cancel raced response completion (benign)")
                # The cancel hit nothing, so NO response.done will fire for it —
                # and response.done is the only place _cancel_requested is
                # cleared. Left stale it gates the whole NEXT response: its text
                # is withheld from TTS and, if it is the goodbye, the hangup is
                # never scheduled and the line sits open in silence. This error
                # is the unambiguous "cancel missed" signal; clearing on
                # response.created instead would wipe the flag for the barge-in
                # recovery cancel above, which deliberately targets a server-VAD
                # response before its response.created has arrived.
                self._cancel_requested = False
                if self._response_create_pending and not self._ai_is_responding:
                    # Our cancel carried a pending re-create but found nothing
                    # active (no response.done will fire it) — fire it now or
                    # the caller's turn is never answered.
                    self._response_create_pending = False
                    try:
                        ws = self.openai_ws
                        if ws:
                            await ws.send(json.dumps({"type": "response.create"}))
                            logger.info("Fired pending response.create after no-op cancel")
                    except Exception as e:
                        logger.warning(f"Could not fire pending response.create: {e}")
            else:
                logger.error(
                    f"OpenAI error: {error.get('message', 'unknown')} (code={code})"
                )

    async def _inject_interrupt_recovery(self):
        """Send the interruption-recovery system item to OpenAI."""
        ws = self.openai_ws
        if not ws:
            return
        try:
            await ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "system",
                    "content": [{
                        "type": "input_text",
                        "text": self._build_interrupt_recovery(),
                    }],
                },
            }))
            logger.info(
                "Injected interruption recovery (heard=%r unheard=%r)",
                self._heard_text_this_response[:60],
                self._unheard_text_this_response[:60],
            )
        except Exception as e:
            logger.debug(f"Could not inject interruption context: {e}")

    def _build_interrupt_recovery(self) -> str:
        """System text injected after a caller barge-in cancelled a response.

        External-TTS mode: OpenAI's history keeps the FULL generated text as if
        it was all spoken, but drained TTS-queue pieces never played. The old
        blanket "never repeat interrupted content" rule therefore made the
        model abandon questions the caller never even heard (live-call bug —
        the agent "forgot" what it was mid-way through asking after every
        interruption). The fix: tell the model exactly what WAS heard
        (_heard_text_this_response) and to re-ask the unheard part in fresh
        words. The native path truncates the audio item instead, so it only
        needs the resume rule, not the heard/unheard split.
        """
        heard = self._heard_text_this_response.strip()[-200:]
        if self._language == "ta":
            if not self.use_elevenlabs:
                return (
                    "[அழைப்பவர் உங்க பேச்சை இடைமறிச்சாங்க. முதல்ல அவங்க இப்போ "
                    "சொன்னதுக்கு பதில் சொல்லுங்க. அப்புறம் நிறுத்தின இடத்துல இருந்து "
                    "தொடருங்க — நீங்க கேட்டுட்டிருந்த கேள்விக்கு இன்னும் பதில் "
                    "வரலைனா அதை வேற வார்த்தைல சுருக்கமா மறுபடி கேளுங்க. ஏற்கனவே "
                    "சொன்னதை அப்படியே திரும்ப சொல்லாதீங்க.]"
                )
            heard_line = (
                f'அவங்க கேட்டது இவ்வளவு மட்டும்: "{heard}". '
                if heard else "உங்க பதில் அவங்க காதுல விழவே இல்லை. "
            )
            unheard = self._unheard_text_this_response.strip()[-250:]
            unheard_line = (
                f'இந்த பகுதி அவங்களுக்கு play ஆகவே இல்லை: "{unheard}". '
                if unheard else ""
            )
            return (
                "[அழைப்பவர் உங்க பேச்சை நடுவுல cut பண்ணிட்டாங்க. " + heard_line +
                unheard_line +
                "Play ஆகாத பேச்சை சொன்னதா நினைக்காதீங்க. அடுத்த பதிலுக்கான rules:\n"
                "1. முதல்ல அவங்க இப்போ சொன்னதுக்கு பதில் சொல்லுங்க.\n"
                "2. அவங்க 'hello?'னு மட்டும் சொன்னா அல்லது யாருனு கேட்டா: ஒரே ஒரு "
                "சின்ன வாக்கியத்துல (பேரு, company, ஏன் call பண்றீங்க) சொல்லிட்டு "
                "உங்க கேள்வியை கேளுங்க. முழு pitch-ஐயும் திரும்ப சொல்ல வேண்டாம்.\n"
                "3. இல்லைனா play ஆகாத முக்கியமான விஷயத்தை அதிகபட்சம் ஒரு சின்ன "
                "வாக்கியத்துல வேற வார்த்தைல சொல்லுங்க — எல்லாத்தையும் திரும்ப "
                "சொல்லாதீங்க.\n"
                "4. 'முதல்ல இருந்து சொல்றேன்', 'திரும்ப அறிமுகம் பண்ணிக்கிறேன்' "
                "மாதிரி narration வேண்டாம் — நேரடியா விஷயத்தை சொல்லுங்க.\n"
                "5. திரும்பவும் interrupt ஆனா ஒவ்வொரு தடவையும் இன்னும் சுருக்கமா "
                "பேசுங்க. அவங்க சொன்னது புரியலைனா புரிஞ்ச மாதிரி பேசாம "
                "'sorry-ங்க, என்ன சொன்னீங்க?'னு கேளுங்க.]"
            )
        if not self.use_elevenlabs:
            return (
                "[You were interrupted by the caller. First answer what the "
                "caller just said. Then pick up from where you left off — if "
                "you were asking a question they have not answered yet, ask it "
                "again briefly in fresh words. Do not repeat verbatim what "
                "they already heard.]"
            )
        heard_line = (
            f'The caller heard ONLY this much: "{heard}". '
            if heard else "The caller heard NONE of it — your reply never played. "
        )
        unheard = self._unheard_text_this_response.strip()[-250:]
        unheard_line = (
            f'This part was NEVER played to them: "{unheard}". ' if unheard else ""
        )
        # Live calls 2026-07-11: the old "say it again naturally" wording made
        # the mini model re-deliver the ENTIRE pitch with meta-commentary
        # ("Let me start over", "I'll explain again") after every "hello?" —
        # the intro played up to 5x in one call and callers hung up. The rules
        # below cap the re-delivery and ban the meta-speech.
        return (
            "[You were interrupted mid-reply. " + heard_line + unheard_line +
            "Do NOT treat the unplayed part as said. Rules for your next reply:\n"
            "1. First respond to what the caller just said.\n"
            "2. If they only said 'hello?' / a greeting / asked who you are: "
            "answer in ONE short sentence (your name, company, reason) and ask "
            "your pending question. Do NOT re-deliver your full pitch.\n"
            "3. Otherwise fold the ESSENTIAL unplayed info into at most ONE "
            "short sentence in fresh words — never restate everything.\n"
            "4. NEVER narrate what you are doing. Banned phrases: 'let me start "
            "over', 'let me introduce myself again', 'I'll explain again', "
            "'let me clarify', 'starting fresh'. Just say the content.\n"
            "5. If you get interrupted again, your replies must get SHORTER "
            "each time, never longer.]"
        )

    async def _forward_response_audio(self, audio_b64: str):
        """Send one response.audio.delta payload to Vobiz (native or PCM path)."""
        try:
            self._audio_chunk_count += 1
            raw_data = base64.b64decode(audio_b64)

            if self._audio_chunk_count <= 3:
                elapsed = time.monotonic() - self._response_start_time
                fmt = "mulaw" if not self.use_elevenlabs else "PCM16"
                print(
                    f"[CALL] Audio chunk #{self._audio_chunk_count} "
                    f"({len(raw_data)} {fmt} bytes, "
                    f"{elapsed:.2f}s since response.created)",
                    flush=True,
                )

            if self._first_audio_chunk:
                self._first_audio_chunk = False
                if self._speech_stopped_time > 0:
                    e2e_ms = (time.monotonic() - self._speech_stopped_time) * 1000
                    self._latency_samples_ms.append(e2e_ms)
                    print(
                        f"[CALL] E2E latency: {e2e_ms:.0f}ms "
                        f"(speech_stopped -> first audio to Vobiz)",
                        flush=True,
                    )
                pacing_ms = Config.RESPONSE_PACING_DELAY_MS
                if pacing_ms > 0:
                    await asyncio.sleep(pacing_ms / 1000.0)

            if not self.use_elevenlabs:
                # g711_ulaw output → send directly to Vobiz as mulaw (ZERO conversion)
                await self._send_audio_to_vobiz(audio_b64)
                self._record_native_audio_sent(len(raw_data))
            else:
                # pcm16 output → buffer and convert to mulaw for Vobiz
                self._pcm_buffer += raw_data
                PCM_CHUNK = 4800  # 100ms at 24kHz 16-bit mono

                while len(self._pcm_buffer) >= PCM_CHUNK:
                    chunk = self._pcm_buffer[:PCM_CHUNK]
                    self._pcm_buffer = self._pcm_buffer[PCM_CHUNK:]
                    resampled = downsample_24k_to_8k(chunk)
                    mulaw = audioop.lin2ulaw(resampled, 2)
                    mulaw_b64_chunk = base64.b64encode(mulaw).decode("utf-8")
                    await self._send_audio_to_vobiz(mulaw_b64_chunk)

            # Refresh echo cooldown
            self._echo_gate_until = time.monotonic() + self._ECHO_COOLDOWN
        except Exception as e:
            logger.error(f"Error sending audio to Vobiz: {e}")

    async def _flush_pending_audio(self):
        """Replay deltas buffered during a backchannel-evaluation window."""
        if not self._pending_audio_deltas:
            return
        pending, self._pending_audio_deltas = self._pending_audio_deltas, []
        for audio_b64 in pending:
            await self._forward_response_audio(audio_b64)

    # ── Observer hook (used by test-call bridge only) ───────────────────────

    async def _emit_transcript(self, role: str, text: str, final: bool):
        cb = self._transcript_callback
        if cb is None:
            return
        try:
            await cb(role, text, final)
        except Exception as e:
            logger.debug(f"Transcript callback failed: {e}")

    # ── Interruption helpers ────────────────────────────────────────────────

    def _record_native_audio_sent(self, byte_count: int):
        """Track carrier playback time for native 8kHz mu-law output."""
        if byte_count <= 0:
            return
        self._any_real_audio_sent = True
        now = time.monotonic()
        if self._response_playback_started_at <= 0:
            self._response_playback_started_at = now
        duration_s = byte_count / 8000.0
        play_start = max(now, self._estimated_playback_end)
        self._estimated_playback_end = play_start + duration_s
        self._response_audio_sent_ms += duration_s * 1000.0
        self._echo_gate_until = self._estimated_playback_end + self._ECHO_COOLDOWN

    async def _truncate_current_audio(self):
        """Remove unheard native audio from OpenAI's conversation history."""
        if (
            self.use_elevenlabs
            or not self._current_response_item_id
            or self._response_playback_started_at <= 0
            or self._response_audio_sent_ms <= 0
        ):
            return
        elapsed_ms = (time.monotonic() - self._response_playback_started_at) * 1000.0
        played_ms = max(0, int(elapsed_ms - Config.VOBIZ_PLAYBACK_LAG_MS))
        played_ms = min(played_ms, int(self._response_audio_sent_ms))
        ws = self.openai_ws
        if not ws:
            return
        item_id = self._current_response_item_id
        try:
            await ws.send(json.dumps({
                "type": "conversation.item.truncate",
                "item_id": item_id,
                "content_index": 0,
                "audio_end_ms": played_ms,
            }))
            self._current_response_item_id = None
            logger.info(
                "Truncated interrupted assistant audio item=%s at %sms/%sms sent",
                item_id,
                played_ms,
                int(self._response_audio_sent_ms),
            )
        except Exception as e:
            logger.debug(f"Could not truncate interrupted assistant audio: {e}")

    async def _gentle_clear(self, delay_s: float):
        """After a brief delay, send clearAudio to Vobiz (stops AI audio playback)."""
        await asyncio.sleep(delay_s)
        if self._interrupt_pending:
            await self._clear_vobiz_audio()

    async def _resolve_stuck_interrupt(self, armed_at: float):
        """Unwedge `_interrupt_pending` when speech_stopped never arrives.

        Server VAD occasionally never emits speech_stopped (e.g. the echo gate
        drops the trailing quiet frames, so OpenAI never 'hears' the silence).
        A stuck flag makes `_tts_utterance_stale` treat every utterance as
        stale — the agent goes permanently silent. Clear the flag WITHOUT
        draining (mirror the backchannel ruling) so pending speech resumes.
        `armed_at` ties this watchdog to its own speech event; a newer
        speech_started re-arms a fresh watchdog and this one stands down.
        """
        # 2.5s (was 4.0s): with the backchannel cap at 600ms, anything armed
        # this long is either a lost speech_stopped or a definite interrupt —
        # every extra watchdog second was pure dead air (a live call measured
        # a 4.3s E2E turn caused by this; it fired 4 times across 4 calls).
        await asyncio.sleep(2.5)
        if self._interrupt_pending and self._speech_start_time == armed_at:
            logger.warning(
                "speech_stopped never arrived (%.1fs) — clearing stuck "
                "interrupt-pending so the agent can speak again", 2.5
            )
            self._interrupt_pending = False
            await self._flush_pending_audio()

    def _drain_tts_queue(self):
        """Invalidate active synthesis and empty queued external-TTS speech."""
        self._tts_generation += 1
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
                self._tts_queue.task_done()
            except asyncio.QueueEmpty:
                break

    # ── External TTS pipeline (ElevenLabs / Sarvam) ─────────────────────────

    async def _flush_sentences(self):
        """Extract complete sentences from the text buffer and enqueue for TTS.

        Enqueues are awaited IN ORDER — using asyncio.create_task here would
        race the direct `await self._enqueue_tts(...)` at response.text.done,
        interleaving audio chunks so the caller heard sentences out of order
        (e.g. the next question playing before its lead-in acknowledgement).

        First piece of each response flushes EARLY at a clause boundary
        (",;:—" + space, >=12 chars) instead of waiting for full sentence
        punctuation — the caller hears the reply ~150-400ms sooner and the one
        prosody seam per turn is masked by turn-taking. Subsequent pieces keep
        whole-sentence splitting for natural bulbul prosody.
        """
        if not self._first_piece_flushed:
            # Scan ALL clause boundaries, not just the first: Tamil replies
            # almost always open with a short "சரி,"/"வணக்கம்," (<12 chars), and
            # bailing on that first match meant the early flush NEVER fired on
            # Tamil calls (measured 2026-07-11 — every turn waited for full
            # sentence punctuation). Take the first boundary past the minimum.
            for m in re.finditer(r'(?<=[,;:—])\s+', self._response_text_buffer):
                if m.start() < 12:
                    continue
                first = self._response_text_buffer[:m.start()].strip()
                self._response_text_buffer = self._response_text_buffer[m.end():]
                if first:
                    self._first_piece_flushed = True
                    await self._enqueue_tts(first)
                break

        sentence_end = re.compile(r'(?<=[.!?])\s+')
        parts = sentence_end.split(self._response_text_buffer)
        if len(parts) > 1:
            for sentence in parts[:-1]:
                sentence = sentence.strip()
                if sentence:
                    self._first_piece_flushed = True
                    await self._enqueue_tts(sentence)
            self._response_text_buffer = parts[-1]

        # Runaway guard: if the model streams a long span with no sentence-ending
        # punctuation, the buffer would otherwise grow until response.text.done and
        # go out as one giant un-interruptible blob (the 28s-chunk bug). Flush it
        # early once it exceeds the cap, keeping the trailing partial word in the
        # buffer so we never synthesize a mid-word fragment that is still streaming.
        if len(self._response_text_buffer) > self._tts_max_chars():
            buf = self._response_text_buffer
            cut = buf.rfind(" ")
            if cut > 0:
                await self._enqueue_tts(buf[:cut])
                self._response_text_buffer = buf[cut + 1:]

    async def _enqueue_tts(self, text: str):
        """Enqueue text for TTS processing.

        Long text is split into bounded chunks (Config.TTS_MAX_CHARS) so a model
        runaway never becomes one multi-second, un-interruptible synthesis. Each
        chunk is queued separately, so an interrupt's _drain_tts_queue() can stop
        the remaining chunks mid-utterance.
        """
        if self._language == "ta":
            # Deterministic register fix: the mini model ignores the "no bookish
            # Tamil" rule often enough that the swap happens here, not in prompt.
            text = _colloquialize_ta(text)
        # Opener de-dup: a mini model repeats the same short opener sentence
        # ("சரி, நன்றி.") verbatim every turn no matter what the prompt says.
        # Drop it when it exactly matches the previous response's opener — the
        # reply then starts with its substance, which is what a human does.
        if not self._openers_seen_this_response:
            self._openers_seen_this_response = True
            stripped = text.strip()
            if stripped and stripped == self._prev_opener and len(stripped) <= 24:
                # Clear before returning: this path used to skip the
                # _prev_opener update below, so the string stayed armed forever
                # and every later turn that matched it was dropped too. When the
                # whole reply IS the opener that means silence on the caller's
                # turn, repeating indefinitely with no way to recover. One drop
                # is the intended de-dup; disarming makes it exactly one.
                logger.info(f"Skipping repeated opener: {stripped[:30]}")
                self._prev_opener = ""
                self._prev_opener_word = ""
                return
            # Same ack WORD ("சரி,") opening two replies in a row also reads
            # robotic even when the rest differs. Trim the repeated word; the
            # alternation (trim → speak → trim) keeps it human.
            m = re.match(r"([^\s,.!?—:;]+)[,.]?(?:\s+(.+))?$", stripped, re.DOTALL)
            first_word = m.group(1).strip().lower() if m else ""
            rest = (m.group(2) or "").strip() if m else ""
            if (rest and first_word in _ACK_OPENER_WORDS
                    and first_word == self._prev_opener_word):
                logger.info(f"Trimming repeated ack opener: {first_word}")
                stripped = rest
                text = rest
                self._prev_opener_word = ""
            else:
                self._prev_opener_word = (
                    first_word if first_word in _ACK_OPENER_WORDS else ""
                )
            self._prev_opener = stripped if len(stripped) <= 24 else ""
        for piece in _split_for_tts(text, self._tts_max_chars()):
            try:
                await asyncio.wait_for(self._tts_queue.put(piece), timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning(f"TTS queue full, dropping: {piece[:40]}")

    async def _tts_worker(self):
        """Worker that processes TTS queue and sends audio to Vobiz."""
        logger.info("TTS worker started")
        try:
            while True:
                self._tts_playing = False
                text = await self._tts_queue.get()
                if text is None:
                    break
                self._tts_playing = True
                generation = self._tts_generation
                try:
                    # Hard bound per piece: worst legit case is ~9s of audio +
                    # 1.5s stale-wait + 8s REST fallback ≈ 20s. Live call
                    # 2026-07-11 06:06: synthesis cleanup wedged silently after
                    # a barge-in and the agent went MUTE for the rest of the
                    # call — no error, no logs, worker stuck. Never again: any
                    # hang self-heals with a fresh TTS client.
                    await asyncio.wait_for(
                        self._synthesize_and_send(text, generation),
                        timeout=25.0,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "TTS synthesis wedged >25s — resetting %s client "
                        "(piece dropped: %r)", self.tts_provider, text[:40]
                    )
                    try:
                        if self._external_tts:
                            await asyncio.wait_for(
                                self._external_tts.close(), timeout=5.0
                            )
                    except Exception:
                        pass
                    self._external_tts = self._make_external_tts()
                    # Fresh client — clear any cooldown so the next piece
                    # actually uses it instead of silently staying on the
                    # fallback voice for the rest of the call.
                    self._external_tts_retry_at = 0.0
                except Exception as e:
                    logger.error(f"TTS worker error: {e}")
                finally:
                    self._tts_queue.task_done()
        finally:
            self._tts_playing = False
            logger.info("TTS worker stopped")

    async def _synthesize_fillers(self):
        """Pre-synthesize the filler clips on a dedicated Sarvam connection.

        A separate service instance is used because the per-call connection is
        NOT safe for concurrent utterances — the welcome is usually synthesizing
        on it at exactly this moment.
        """
        svc = None
        try:
            svc = SarvamTTSService(language=self._language)
            for text in _FILLER_TEXTS_TA:
                try:
                    clip = await svc.synthesize(text)
                    if len(clip) > _FILLER_MAX_BYTES:
                        logger.info(
                            f"Filler {text!r} too long ({len(clip)/8000.0:.1f}s) — "
                            f"skipped (would delay the real reply)"
                        )
                        continue
                    self._filler_clips.append(clip)
                except Exception as e:
                    logger.debug(f"Filler synth failed for {text!r}: {e}")
            for text in _LISTEN_BC_TEXTS_TA:
                try:
                    clip = await svc.synthesize(text)
                    if len(clip) > _LISTEN_BC_MAX_BYTES:
                        logger.info(
                            f"Listen-backchannel {text!r} too long "
                            f"({len(clip)/8000.0:.1f}s) — skipped"
                        )
                        continue
                    self._listen_bc_clips.append(
                        _attenuate_ulaw(clip, _LISTEN_BC_GAIN)
                    )
                except Exception as e:
                    logger.debug(f"Listen-backchannel synth failed for {text!r}: {e}")
            logger.info(
                f"Filler cache ready: {len(self._filler_clips)} clips, "
                f"{len(self._listen_bc_clips)} listen-backchannels"
            )
        except Exception as e:
            logger.debug(f"Filler cache setup failed: {e}")
        finally:
            if svc is not None:
                try:
                    await svc.close()
                except Exception:
                    pass

    async def _maybe_play_filler(self):
        """Mask response latency with a pre-synthesized neutral hesitation.

        Armed at end of the caller's turn; fires only if no real audio has
        reached the caller after _FILLER_DELAY_S. Perceived latency tracks
        time-to-first-audio, so "ம்ம்..." at 400ms makes a 1.5s reply feel
        immediate. Throttled and rotated so it never reads as canned.
        """
        if not self._filler_clips or self.tts_provider != "sarvam":
            return
        if self._turn_count - self._filler_last_turn < _FILLER_MIN_GAP_TURNS:
            return
        await asyncio.sleep(_FILLER_DELAY_S)
        if self._tts_playing or time.monotonic() < self._estimated_playback_end:
            return  # real reply already reached the caller
        if self._interrupt_pending or self._goodbye_detected or self._cleanup_started:
            return
        clip = self._filler_clips[self._filler_index % len(self._filler_clips)]
        self._filler_index += 1
        self._filler_last_turn = self._turn_count
        try:
            await self._send_ulaw_chunk(clip)
            logger.info("Filler played (no real audio within %dms)",
                        int(_FILLER_DELAY_S * 1000))
        except Exception as e:
            logger.debug(f"Filler playback failed: {e}")

    async def _play_listen_backchannels(self, utterance_started_at: float):
        """Hum a soft "ம்ம்..." while the caller talks at length, like a human listener.

        Armed at speech_started, cancelled at speech_stopped. Fires only when
        the SAME utterance is still running and the agent is fully silent, at
        most twice per monologue. The clip bypasses _send_ulaw_chunk on purpose:
        it must not extend _estimated_playback_end (a later speech_started would
        then arm interruption handling against a hum) — only the echo gate is
        raised so our own hum can't come back as caller input.
        """
        if not self._listen_bc_clips:
            return
        try:
            delay = _LISTEN_BC_AFTER_S
            for _ in range(_LISTEN_BC_MAX_PER_UTTERANCE):
                await asyncio.sleep(delay)
                delay = _LISTEN_BC_REPEAT_S
                if (self._cleanup_started or self._goodbye_detected
                        or self._is_first_response):
                    return
                # Same utterance still running? (speech_stopped stamps a newer time)
                if (self._speech_start_time != utterance_started_at
                        or self._speech_stopped_time >= utterance_started_at):
                    return
                now = time.monotonic()
                # Agent audio in flight — a hum on top would talk over ourselves.
                if self._tts_playing or now < self._estimated_playback_end:
                    continue
                if now - self._listen_bc_last_ts < _LISTEN_BC_MIN_GAP_S:
                    continue
                clip = self._listen_bc_clips[
                    self._listen_bc_index % len(self._listen_bc_clips)
                ]
                self._listen_bc_index += 1
                self._listen_bc_last_ts = now
                mulaw_b64 = base64.b64encode(clip).decode("utf-8")
                await self._send_audio_to_vobiz(mulaw_b64)
                self._echo_gate_until = (
                    now + len(clip) / 8000.0 + self._ECHO_COOLDOWN
                )
                logger.info("Listening backchannel played (caller mid-monologue)")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Listen-backchannel playback failed: {e}")

    async def _tts_utterance_stale(self, generation: int) -> bool:
        """Wait out a backchannel-evaluation window; report if this text is stale.

        While `_interrupt_pending` is unresolved (<=~450ms normally) we PAUSE
        instead of discarding — a "mm-hm" ruling lets queued speech continue.
        Only a real interrupt (generation bump) marks the utterance stale.
        Previously every queued sentence was synthesized then discarded during
        the window, which on a live call became a reconnect/synthesis storm
        (one full Sarvam connect+synth+discard cycle per sentence, ~800ms each).
        Bounded wait so a stuck flag can never wedge the TTS worker.
        """
        # The wait must outlast the lost-speech_stopped watchdog (2.5s): the
        # old 1.5s bound dropped CURRENT-generation speech a full second before
        # the watchdog could clear a stuck flag — live call 2026-07-11 02:13:
        # a "Mm." backchannel with a lost stop event swallowed "Sure, take
        # your time." and left ~3.5s of dead air. If no interruption ruling
        # ever bumps the generation, this utterance is still the current
        # answer — keep holding it. Waiting is safe: the caller is (believed)
        # speaking, so staying silent is correct; a real interrupt drains via
        # the generation bump the moment it is ruled. 6s = last-resort valve
        # for a truly wedged flag (watchdog chains re-arm every 2.5s).
        waited = 0.0
        while self._interrupt_pending and waited < 6.0:
            if generation != self._tts_generation:
                return True
            await asyncio.sleep(0.05)
            waited += 0.05
        return generation != self._tts_generation or self._interrupt_pending

    def _record_heard_text(self, text: str, secs: float = 0.0):
        """Track speech sent toward the caller for this response.

        Recorded per TTS piece as soon as any of its audio was sent. Optimistic
        default: sent = heard. When playback is CUT mid-response,
        _snapshot_playback_cut() re-splits these pieces by what actually played.
        """
        self._sent_pieces.append((text.strip(), max(0.0, secs)))
        if self._heard_text_this_response:
            self._heard_text_this_response += " "
        self._heard_text_this_response += text.strip()

    def _snapshot_playback_cut(self):
        """Split sent pieces into heard/unheard when playback is cleared.

        Vobiz buffers seconds of audio ahead, so at clearAudio time a piece may
        be fully sent yet never played. Using each piece's duration against the
        wall clock since the response's audio started, decide what the caller
        actually heard. A piece counts as heard only if ≥60% of it played —
        erring toward "unheard" (re-saying a half-heard sentence is cheaper
        than skipping a never-heard question).
        """
        now = time.monotonic()
        if (not self._sent_pieces or self._response_audio_started_at <= 0
                or now >= self._estimated_playback_end):
            return  # nothing playing — nothing was cut
        played = max(
            0.0,
            now - self._response_audio_started_at
            - Config.VOBIZ_PLAYBACK_LAG_MS / 1000.0,
        )
        heard: list[str] = []
        unheard: list[str] = []
        acc = 0.0
        for text, secs in self._sent_pieces:
            if played >= acc + 0.6 * secs:
                heard.append(text)
            else:
                unheard.append(text)
            acc += secs
        self._heard_text_this_response = " ".join(heard)
        self._unheard_text_this_response = " ".join(unheard)
        if unheard:
            logger.info(
                "Playback cut at %.1fs/%.1fs — caller heard %d/%d pieces; "
                "unheard: %r",
                played, acc, len(heard), len(self._sent_pieces),
                self._unheard_text_this_response[:80],
            )

    def _note_external_first_audio(self):
        """Record E2E latency when the response's first REAL TTS audio goes out.

        The native path logs this in _forward_response_audio; without this the
        external-TTS (Sarvam/ElevenLabs) path had no latency telemetry at all.
        Fillers deliberately don't call this — they mask latency, they aren't
        the reply.
        """
        if not self._first_audio_chunk:
            return
        self._first_audio_chunk = False
        self._any_real_audio_sent = True
        # Anchor for playback-cut math: real response audio starts either now or
        # behind whatever is already queued at the carrier (e.g. a filler).
        self._response_audio_started_at = max(
            time.monotonic(), self._estimated_playback_end
        )
        if self._speech_stopped_time:
            e2e_ms = (time.monotonic() - self._speech_stopped_time) * 1000
            self._latency_samples_ms.append(e2e_ms)
            logger.info(
                f"[CALL] E2E latency: {e2e_ms:.0f}ms "
                f"(speech_stopped -> first {self.tts_provider} audio to Vobiz)"
            )

    async def _send_ulaw_chunk(self, chunk: bytes) -> int:
        """Send one mulaw chunk to Vobiz and update playback/echo estimates."""
        mulaw_b64 = base64.b64encode(chunk).decode("utf-8")
        await self._send_audio_to_vobiz(mulaw_b64)
        playback_secs = len(chunk) / 8000.0
        now = time.monotonic()
        play_start = max(now, self._estimated_playback_end)
        self._estimated_playback_end = play_start + playback_secs
        self._echo_gate_until = self._estimated_playback_end + self._ECHO_COOLDOWN
        return len(chunk)

    async def _stream_sarvam_and_send(self, text: str, generation: int):
        """Stream Sarvam WS chunks straight to Vobiz as they arrive.

        The caller hears the first chunk in ~350ms while the rest of the
        utterance is still synthesizing (Sarvam's REST path measured 0.6-14s
        under load, so the whole-utterance-then-send approach is not used).
        Sarvam audio is already peak-normalized (~0.8-1.0 measured), so the
        ElevenLabs loudness boost is skipped — normalizing chunk-by-chunk
        would pump the gain within a single utterance.

        Failure chain: WS (retried once inside synthesize_stream) → one REST
        attempt → raise, which lets _synthesize_and_send fall back to OpenAI
        TTS. If the stream dies AFTER audio was sent, the remainder is dropped
        instead — re-synthesizing would repeat the start of the sentence.
        """
        sent = 0
        try:
            try:
                # aclosing() closes the generator DETERMINISTICALLY on early return.
                # A bare `async for` leaves an abandoned generator to a GC-scheduled
                # aclose(), racing the next utterance onto a socket that still holds
                # this utterance's remaining chunks (audio bleed / off-by-one).
                async with aclosing(self._external_tts.synthesize_stream(text)) as agen:
                    async for chunk in agen:
                        if await self._tts_utterance_stale(generation):
                            logger.info("Discarding stale sarvam synthesis after interruption")
                            return
                        self._note_external_first_audio()
                        sent += await self._send_ulaw_chunk(chunk)
            except Exception as e:
                if sent:
                    logger.warning(
                        f"Sarvam stream died mid-utterance after {sent}b ({e}); "
                        f"dropping remainder"
                    )
                    return
                logger.info(f"Sarvam WS produced no audio ({e}); trying REST fallback")
                audio_bytes = await self._external_tts.synthesize_rest(text)
                CHUNK_SIZE = 4000
                for i in range(0, len(audio_bytes), CHUNK_SIZE):
                    if await self._tts_utterance_stale(generation):
                        return
                    self._note_external_first_audio()
                    sent += await self._send_ulaw_chunk(audio_bytes[i:i + CHUNK_SIZE])
            if sent:
                logger.info(
                    f"Sarvam TTS sent {sent} bytes (~{sent / 8000.0:.1f}s) "
                    f"for: {text[:50]}..."
                )
        finally:
            if sent:
                self._record_heard_text(text, sent / 8000.0)

    # How long external TTS stays benched after a failure before it is tried
    # again. Long enough to ride out a socket blip, short enough that at most a
    # turn or two of a call is spoken in the fallback voice.
    _EXTERNAL_TTS_COOLDOWN_S = 20.0

    # Hard bound on ONE external-TTS attempt, kept comfortably under the
    # _tts_worker's 25s per-piece kill. The provider's own internal timeouts do
    # not add up to a usable guarantee — Sarvam's worst path is two WS attempts
    # plus two 3s socket closes plus the REST fallback (~29-34s), which blew
    # past the worker's kill. That cancellation lands ABOVE this function, so
    # its except never ran, the OpenAI fallback was never reached, and the
    # sentence was dropped entirely: 25s of dead air, repeating on the next
    # piece. Bounding the attempt here keeps the failure INSIDE the try, so the
    # caller hears the sentence in the fallback voice instead of silence.
    _EXTERNAL_TTS_DEADLINE_S = 18.0

    def _external_tts_ready(self) -> bool:
        """True when external TTS is not in its post-failure cooldown."""
        if not self._external_tts_retry_at:
            return True
        if time.monotonic() >= self._external_tts_retry_at:
            self._external_tts_retry_at = 0.0
            logger.info(f"{self.tts_provider} TTS cooldown over — retrying it")
            return True
        return False

    async def _synthesize_and_send(self, text: str, generation: int):
        """Synthesize text and send it only while this response is still current."""
        if not self._vobiz_ws or not self.stream_id:
            return
        # Resolve any backchannel window BEFORE spending a synthesis: stale text
        # is skipped without ever opening a connection or paying for TTS.
        if await self._tts_utterance_stale(generation):
            logger.info(f"Skipping stale TTS text (pre-synthesis): {text[:40]}")
            return
        try:
            if self._external_tts and self._external_tts_ready():
                if self.tts_provider == "sarvam":
                    await asyncio.wait_for(
                        self._stream_sarvam_and_send(text, generation),
                        timeout=self._EXTERNAL_TTS_DEADLINE_S,
                    )
                    return
                audio_bytes = await asyncio.wait_for(
                    self._external_tts.synthesize(text),
                    timeout=self._EXTERNAL_TTS_DEADLINE_S,
                )
                if await self._tts_utterance_stale(generation):
                    logger.info(
                        f"Discarding stale {self.tts_provider} synthesis after interruption"
                    )
                    return
                if audio_bytes:
                    audio_bytes = _amplify_ulaw(
                        audio_bytes, Config.TTS_TARGET_PEAK, Config.TTS_MAX_GAIN
                    )
                    CHUNK_SIZE = 4000
                    for i in range(0, len(audio_bytes), CHUNK_SIZE):
                        if await self._tts_utterance_stale(generation):
                            return
                        self._note_external_first_audio()
                        if i == 0:
                            self._record_heard_text(text, len(audio_bytes) / 8000.0)
                        chunk = audio_bytes[i:i + CHUNK_SIZE]
                        mulaw_b64 = base64.b64encode(chunk).decode("utf-8")
                        await self._send_audio_to_vobiz(mulaw_b64)
                    # Estimate when Vobiz will finish playing this audio
                    playback_secs = len(audio_bytes) / 8000.0
                    now = time.monotonic()
                    play_start = max(now, self._estimated_playback_end)
                    self._estimated_playback_end = play_start + playback_secs
                    self._echo_gate_until = self._estimated_playback_end + self._ECHO_COOLDOWN
                    logger.info(f"TTS sent {len(audio_bytes)} bytes (~{playback_secs:.1f}s) for: {text[:50]}...")
                return
        except Exception as e:
            # Cooldown, not a permanent kill. This used to latch False for the
            # rest of the call, so ONE transient socket hiccup meant every
            # remaining turn came out of OpenAI tts-1/alloy — a Tamil script read
            # aloud in an English voice, and the streaming path (first audio
            # ~350ms) replaced by whole-utterance synthesis.
            logger.warning(
                f"{self.tts_provider} TTS failed ({e}), falling back to OpenAI TTS "
                f"for {self._EXTERNAL_TTS_COOLDOWN_S:.0f}s"
            )
            self._external_tts_retry_at = time.monotonic() + self._EXTERNAL_TTS_COOLDOWN_S

        # OpenAI TTS fallback
        try:
            if not self._openai_tts_client:
                self._openai_tts_client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
            async with self._openai_tts_client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="alloy",
                input=text,
                response_format="pcm",
            ) as response:
                pcm_buffer = b""
                heard_recorded = False
                async for chunk in response.iter_bytes(chunk_size=4800):
                    if await self._tts_utterance_stale(generation):
                        return
                    pcm_buffer += chunk
                    PCM_CHUNK = 4800
                    while len(pcm_buffer) >= PCM_CHUNK:
                        seg = pcm_buffer[:PCM_CHUNK]
                        pcm_buffer = pcm_buffer[PCM_CHUNK:]
                        resampled = downsample_24k_to_8k(seg)
                        mulaw = audioop.lin2ulaw(resampled, 2)
                        mulaw_b64 = base64.b64encode(mulaw).decode("utf-8")
                        await self._send_audio_to_vobiz(mulaw_b64)
                        if not heard_recorded:
                            heard_recorded = True
                            self._record_heard_text(text)
                        self._echo_gate_until = time.monotonic() + self._ECHO_COOLDOWN
                # Flush remainder
                if pcm_buffer and generation == self._tts_generation and not self._interrupt_pending:
                    resampled = downsample_24k_to_8k(pcm_buffer)
                    mulaw = audioop.lin2ulaw(resampled, 2)
                    mulaw_b64 = base64.b64encode(mulaw).decode("utf-8")
                    await self._send_audio_to_vobiz(mulaw_b64)
        except Exception as e:
            logger.error(f"OpenAI TTS fallback also failed: {e}")

    async def _flush_pcm_remainder(self):
        """Flush any leftover PCM buffer at end of response (ElevenLabs mode)."""
        if self._pcm_buffer:
            try:
                resampled = downsample_24k_to_8k(self._pcm_buffer)
                mulaw = audioop.lin2ulaw(resampled, 2)
                mulaw_b64 = base64.b64encode(mulaw).decode("utf-8")
                await self._send_audio_to_vobiz(mulaw_b64)
            except Exception as e:
                logger.error(f"Error flushing PCM remainder: {e}")
            self._pcm_buffer = b""

    # ── Goodbye / hangup ────────────────────────────────────────────────────

    def _contains_goodbye(self, text: str) -> bool:
        lower = text.lower()
        if self._language == "ta":
            # The Tamil prompt is told to close with "உங்கள் நேரத்துக்கு நன்றி" — an
            # unambiguous farewell that won't appear in a greeting (so the welcome won't
            # trigger a hangup). Also catch an English "goodbye" in case it code-switches.
            if "நேரத்துக்கு நன்றி" in text or "நேரத்திற்கு நன்றி" in text:
                return True
            if "goodbye" in lower or "good bye" in lower:
                return True
            return False
        if "goodbye" in lower or "good bye" in lower or "farewell" in lower:
            return True
        if "thank you for your time" in lower and (
            "have a great day" in lower or "take care" in lower
        ):
            return True
        return False

    def _schedule_hangup(self, delay_s: float):
        """Schedule a call hangup after delay_s seconds."""
        if not self._goodbye_detected:
            self._goodbye_detected = True
            self._hangup_task = asyncio.create_task(self._delayed_hangup(delay_s))
            logger.info(f"Goodbye detected — scheduling hangup in {delay_s}s")

    def _cancel_hangup(self):
        if self._hangup_task and not self._hangup_task.done():
            self._hangup_task.cancel()
            self._goodbye_detected = False
            logger.info("Hangup cancelled (caller spoke again)")

    def _pause_hangup(self):
        """Caller spoke after we already said goodbye. Cancel the timer so we don't
        cut them off mid-sentence, but STAY in goodbye mode and flag that we owe a
        decision once their words are transcribed (see the post-goodbye block in the
        transcription handler). Arm a generous fallback hangup in case no usable
        transcript ever arrives, so the call can't hang open. Unlike _cancel_hangup,
        this keeps _goodbye_detected True."""
        if self._hangup_task and not self._hangup_task.done():
            self._hangup_task.cancel()
        self._awaiting_post_goodbye_eval = True
        self._hangup_task = asyncio.create_task(
            self._delayed_hangup(Config.POST_GOODBYE_HANGUP_DELAY_S + 4.0)
        )
        logger.info("Hangup paused — evaluating caller's post-goodbye remark")

    async def _delayed_hangup(self, delay_s: float):
        await asyncio.sleep(delay_s)
        await self.hangup_via_api()
        try:
            if self._vobiz_ws:
                await self._vobiz_ws.close()
        except Exception:
            pass

    # ── Session restart (on "start over" command) ───────────────────────────

    async def _restart_session(self):
        """Fully reset the OpenAI session to clear conversation history.

        Runs INSIDE _receive_task (reached via _handle_openai_event), so the very
        task executing this is the one whose socket is being closed. Once
        _connect_openai() installs a replacement receive task, this task unwinds
        into its own `finally` — which must not treat the deliberate close as an
        unexpected disconnect and tear down the call. That finally detects the
        swap by comparing the socket it owns against self.openai_ws.
        """
        logger.info("Restarting OpenAI session (caller requested start over)...")
        try:
            if self.openai_ws:
                await self.openai_ws.close()
        except Exception:
            pass
        self.openai_ws = None
        self._connected = False
        self._current_ai_transcript = ""
        self._response_text_buffer = ""
        self._goodbye_detected = False
        self._awaiting_post_goodbye_eval = False
        self._pcm_buffer = b""
        self._is_first_response = True
        await self._connect_openai()

    # ── Vobiz API: hangup call via REST ────────────────────────────────────

    async def hangup_via_api(self):
        """Hang up the current call using the Vobiz REST API."""
        if not self.call_id:
            logger.warning("No call_id to hang up")
            return
        try:
            import httpx
            url = (
                f"https://api.vobiz.ai/api/v1/Account/{Config.VOBIZ_AUTH_ID}"
                f"/Call/{self.call_id}/"
            )
            headers = {
                "X-Auth-ID": Config.VOBIZ_AUTH_ID,
                "X-Auth-Token": Config.VOBIZ_AUTH_TOKEN,
            }
            async with httpx.AsyncClient() as client:
                resp = await client.delete(url, headers=headers, timeout=10.0)
            if resp.status_code in (200, 204):
                logger.info(f"Hung up call {self.call_id} via Vobiz API")
            else:
                logger.error(f"Hang up call {self.call_id} failed: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.error(f"Hang up via Vobiz API failed: {e}")

    # ── Cleanup ─────────────────────────────────────────────────────────────

    def get_metrics(self) -> dict:
        latencies = self._latency_samples_ms
        return {
            "model": Config.REALTIME_MODEL,
            "language": self._language,
            "elevenlabs": self.use_elevenlabs,
            "tts_provider": self.tts_provider,
            "turns": self._turn_count,
            "interruptions": self._interrupt_count,
            "avg_e2e_latency_ms": (
                round(sum(latencies) / len(latencies), 1) if latencies else None
            ),
            "max_e2e_latency_ms": round(max(latencies), 1) if latencies else None,
            "realtime_tokens": self._total_realtime_tokens,
            "input_tokens": self._total_input_tokens,
            "cached_input_tokens": self._cached_input_tokens,
            "output_tokens": self._total_output_tokens,
            "transcription_tokens": self._transcription_tokens,
        }

    def _spawn_bg(self, coro) -> asyncio.Task:
        """Start a one-shot background task that cleanup can actually reach.

        Bare asyncio.create_task() with no stored reference has two failure
        modes here: the loop keeps only a weak reference, so a task suspended in
        sleep() can be garbage-collected mid-flight; and _full_cleanup never saw
        these, so they kept running against closed sockets after the call ended.
        The first one bites hardest on _resolve_stuck_interrupt — the watchdog
        added for the 2026-07-10 incident where _interrupt_pending wedged True
        and muted the agent for 30s. If that task is collected, the watchdog
        silently never fires and the dead-air bug it guards against returns.
        """
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return task

    async def _close_vobiz_on_failure(self, reason: str):
        """Close the Vobiz WebSocket when OpenAI connection fails."""
        logger.error(f"Closing Vobiz stream due to: {reason}")
        try:
            if self._vobiz_ws:
                await self._vobiz_ws.close()
        except Exception:
            pass

    async def _full_cleanup(self):
        """Clean up all connections and tasks."""
        if self._cleanup_started:
            return
        self._cleanup_started = True

        # Cancel tasks. _bg_tasks holds the one-shot sleepers (fillers, gentle
        # clears, stuck-interrupt watchdogs); without them here a task that woke
        # after cleanup would write audio into an already-closed Vobiz socket and
        # mutate playback/echo-gate state on a dead handler.
        for task in [self._receive_task, self._tts_worker_task, self._hangup_task,
                     self._filler_synth_task, self._listen_bc_task,
                     *list(self._bg_tasks)]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        # Close OpenAI connection
        if self.openai_ws:
            try:
                await self.openai_ws.close()
            except Exception:
                pass
            self.openai_ws = None

        if self._external_tts:
            try:
                await self._external_tts.close()
            except Exception as e:
                logger.debug(f"Failed to close {self.tts_provider} TTS client: {e}")

        if self._openai_tts_client:
            try:
                await self._openai_tts_client.close()
            except Exception as e:
                logger.debug(f"Failed to close OpenAI TTS client: {e}")
            self._openai_tts_client = None

        self._connected = False
        logger.info(
            "VobizRealtimeHandler cleanup complete — callId=%s turns=%s "
            "interrupts=%s realtime_tokens=%s",
            self.call_id,
            self._turn_count,
            self._interrupt_count,
            self._total_realtime_tokens,
        )
