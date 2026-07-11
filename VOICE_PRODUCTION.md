# Voice Calling Production Profile

This project now defaults to the lowest-cost native path that preserves real-time conversation quality:

- Vobiz 8 kHz mu-law is passed directly to OpenAI Realtime without audio conversion.
- `gpt-realtime-mini` is the default model. The flagship model remains selectable in the UI when quality is worth the added cost.
- ElevenLabs is optional and off by default, avoiding a second synthesis provider and its latency/cost.
- English transcripts use `gpt-4o-mini-transcribe`; Tamil keeps `gpt-4o-transcribe` for accuracy.
- Server VAD is the phone-safe default. It reliably committed 8 kHz mu-law in live loopback testing where semantic VAD did not.

## Verified behavior — June 25, 2026

The following checks passed against the current implementation:

- 11 automated backend tests.
- Python syntax validation across backend source files.
- Browser PCM interruption test: the active source stops, queued audio is discarded, and the AudioContext closes.
- Live OpenAI session configuration: English native, Tamil native, and ElevenLabs text-output mode.
- Live prewarmed 8 kHz mu-law conversational loopback:
  - English: 266 ms from `speech_stopped` to first outbound audio chunk.
  - Tamil: 579 ms from `speech_stopped` to first outbound audio chunk.

These are application-path measurements to the Vobiz socket adapter. Real PSTN latency also includes carrier and handset network delay.

## Stability controls

- Every call owns its own expiring prewarm socket; concurrent calls cannot steal another call's language, prompt, or connection.
- Unanswered prewarms expire after 75 seconds.
- Failed Realtime connections close the media stream instead of leaving dead air.
- Native barge-in clears carrier audio and truncates unheard assistant audio from model history.
- ElevenLabs synthesis is generation-scoped, so completed stale synthesis cannot play after interruption.
- OpenAI and ElevenLabs clients are closed during cleanup.
- Python 3.13+ installs `audioop-lts` automatically.

## Cost controls

- Realtime mini is the fresh-install default.
- Conversation history is limited to 8,000 post-instruction tokens with a `0.8` retention ratio. A compatibility retry omits truncation if a model rejects it.
- Prompt caching remains enabled automatically by OpenAI; the call metrics report cached input tokens.
- `/api/call-state/{call_uuid}` returns measured turns, interruptions, latency, Realtime tokens, cached tokens, and transcription tokens after the stream closes.

## Recommended production environment

Copy `backend/.env.example` to `backend/.env`. The important defaults are:

```env
ENVIRONMENT=production
REALTIME_MODEL=gpt-realtime-mini
TRANSCRIPTION_MODEL=gpt-4o-mini-transcribe
TRANSCRIPTION_MODEL_TA=gpt-4o-transcribe
VAD_TYPE=server_vad
VAD_PREFIX_PADDING_MS=120
VAD_SILENCE_DURATION_MS=200
VAD_INTERRUPT_RESPONSE=false
INTERRUPTION_EVAL_DELAY_MS=250
BACKCHANNEL_MAX_DURATION_MS=250
PREWARM_TTL_S=75
REALTIME_HISTORY_TOKEN_LIMIT=8000
REALTIME_RETENTION_RATIO=0.8
```

Keep ElevenLabs disabled unless its voice quality materially improves conversion or Tamil pronunciation for your callers.

## Verification commands

```powershell
cd backend
$env:PYTHONDONTWRITEBYTECODE='1'
python -B -m pytest -q -p no:cacheprovider
python -B test_openai.py
```

Before a large rollout, make real carrier calls in quiet, noisy, and speakerphone conditions and record P50/P95 latency plus correction and interruption rates from the call metrics.