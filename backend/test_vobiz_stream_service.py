import asyncio
import json
import time

from config import Config

from services.vobiz_stream_service import (
    VobizRealtimeHandler,
    _attenuate_ulaw,
    _colloquialize_ta,
    _extract_spelled_letters,
    _is_closing_remark,
    _is_disengage_intent,
    _split_for_tts,
)


def test_split_for_tts_respects_limit_and_preserves_text():
    text = (
        "This is the first sentence. This is a longer second sentence, "
        "with enough words to require splitting safely."
    )
    chunks = _split_for_tts(text, 45)

    assert chunks
    assert all(len(chunk) <= 45 for chunk in chunks)
    assert " ".join(chunks).replace("  ", " ") == text


def test_colloquialize_ta_swaps_bookish_words():
    assert _colloquialize_ta("தயவுசெய்து சொல்லுங்க") == "கொஞ்சம் சொல்லுங்க"
    assert _colloquialize_ta("தயவு செய்து உங்க full name சொல்லுங்க") == (
        "கொஞ்சம் உங்க full name சொல்லுங்க"
    )
    assert _colloquialize_ta("மன்னிக்கவும், மறுபடி சொல்லுங்க") == (
        "sorry-ங்க, மறுபடி சொல்லுங்க"
    )
    # Text without formal words passes through untouched.
    assert _colloquialize_ta("சரிங்க, confirm பண்றேன்") == "சரிங்க, confirm பண்றேன்"


def test_colloquialize_ta_fixes_bookish_verb_morphology():
    # Observed live/probe 2026-07-11: the model rewrites the script's own
    # colloquial welcome into bookish present-tense forms.
    assert _colloquialize_ta("நான் ப்ரியா பேசுகிறேன்") == "நான் ப்ரியா பேசுறேன்"
    assert _colloquialize_ta("நான் போகிறேன்") == "நான் போறேன்"
    assert _colloquialize_ta("அவர் இருக்கிறார்") == "அவர் இருக்கறார்"
    assert _colloquialize_ta("அது இருக்கிறது") == "அது இருக்கு"
    assert _colloquialize_ta("என்ன செய்கிறீர்கள்") == "என்ன செய்றீங்க"
    assert _colloquialize_ta("நேரம் இருக்கிறதா") == "நேரம் இருக்கா"
    # Formal plural endings and word-final ங்கள்.
    assert _colloquialize_ta("நீங்கள் சொன்னீர்களா") == "நீங்க சொன்னீங்களா"
    assert _colloquialize_ta("சொல்லுங்கள்") == "சொல்லுங்க"
    assert _colloquialize_ta("உங்கள் பெயர் சொல்லுங்கள்") == "உங்க பெயர் சொல்லுங்க"
    # உங்களுக்கு (case-suffixed) is already spoken — must NOT become உங்கக்கு.
    assert _colloquialize_ta("உங்களுக்கு நன்றி") == "உங்களுக்கு நன்றி"
    # Common bookish words.
    assert _colloquialize_ta("சொல்ல வேண்டும்") == "சொல்ல வேணும்"
    assert _colloquialize_ta("வேண்டுமா") == "வேணுமா"
    assert _colloquialize_ta("அது இல்லை") == "அது இல்ல"
    assert _colloquialize_ta("ஆனால் இப்போது முடியும்") == "ஆனா இப்போ முடியும்"
    # எப்போதும் ("always") must survive the இப்போது→இப்போ rule.
    assert _colloquialize_ta("எப்போதும் இப்படித்தான்") == "எப்போதும் இப்படித்தான்"
    # Word-initial கிற (loanwords) is never a tense marker — untouched.
    assert _colloquialize_ta("கிறிஸ்துமஸ் வாழ்த்துக்கள்") == "கிறிஸ்துமஸ் வாழ்த்துக்கள்"
    # English text inside Tanglish is untouched.
    assert _colloquialize_ta("Selenium framework பயன்படுத்துகிறேன்") == (
        "Selenium framework பயன்படுத்துறேன்"
    )


def test_first_piece_flush_skips_short_opener_comma():
    async def scenario():
        handler = VobizRealtimeHandler(tts_provider="sarvam", language="ta")
        handler._first_piece_flushed = False
        # Tamil replies open with a short "சரி," — the old code checked ONLY the
        # first comma, saw it under the 12-char minimum, and never early-flushed.
        handler._response_text_buffer = (
            "சரி, உங்க details எல்லாம் note பண்ணிட்டேன், அடுத்த கேள்விக்கு போகலாம்"
        )
        await handler._flush_sentences()
        assert handler._tts_queue.qsize() == 1
        first = await handler._tts_queue.get()
        assert first == "சரி, உங்க details எல்லாம் note பண்ணிட்டேன்,"
        assert handler._first_piece_flushed is True

    asyncio.run(scenario())


def test_first_piece_flushes_early_at_clause_boundary():
    async def scenario():
        handler = VobizRealtimeHandler(tts_provider="sarvam", language="ta")
        handler._first_piece_flushed = False
        # Streaming text with a comma but no sentence-ending punctuation yet.
        handler._response_text_buffer = "சரிங்க நன்றி, உங்க full name சொல்லுங்க"
        await handler._flush_sentences()
        # The clause before the comma went to the queue without waiting for ".".
        assert handler._tts_queue.qsize() == 1
        first = await handler._tts_queue.get()
        assert first == "சரிங்க நன்றி,"
        assert handler._first_piece_flushed is True
        # Subsequent text waits for full sentences (no more early flushes).
        handler._response_text_buffer = "அப்புறம், இன்னொரு clause வந்தா"
        await handler._flush_sentences()
        assert handler._tts_queue.qsize() == 0

    asyncio.run(scenario())


def test_repeated_opener_is_dropped():
    async def scenario():
        handler = VobizRealtimeHandler(tts_provider="sarvam", language="ta")
        # Response 1 opener goes through.
        handler._openers_seen_this_response = False
        await handler._enqueue_tts("சரி, நன்றி.")
        assert handler._tts_queue.qsize() == 1
        await handler._tts_queue.get()
        # Response 2 repeats the same opener — dropped; substance still flows.
        handler._openers_seen_this_response = False
        await handler._enqueue_tts("சரி, நன்றி.")
        assert handler._tts_queue.qsize() == 0
        await handler._enqueue_tts("உங்க பெயர் சொல்லுங்க.")
        assert handler._tts_queue.qsize() == 1

    asyncio.run(scenario())


def test_interrupt_recovery_reports_heard_vs_unheard():
    class FakeOpenAIWebSocket:
        def __init__(self):
            self.sent = []

        async def send(self, message):
            self.sent.append(json.loads(message))

    async def scenario():
        handler = VobizRealtimeHandler(tts_provider="sarvam", language="ta")
        ws = FakeOpenAIWebSocket()
        handler.openai_ws = ws
        # The model generated an opener + a question, but only the opener was
        # played before the caller barged in — the question was never heard.
        handler._current_ai_transcript = (
            "சரி, note பண்ணிட்டேன். அடுத்தது, expected salary என்ன range-ல பாக்கறீங்க?"
        )
        handler._heard_text_this_response = "சரி, note பண்ணிட்டேன்."

        await handler._handle_openai_event({
            "type": "response.done",
            "response": {"status": "cancelled"},
        })

        injected = [m for m in ws.sent if m.get("type") == "conversation.item.create"]
        assert len(injected) == 1
        text = injected[0]["item"]["content"][0]["text"]
        # Tells the model exactly what the caller DID hear...
        assert "சரி, note பண்ணிட்டேன்." in text
        # ...and caps the re-delivery to one short sentence instead of a full
        # restart (2026-07-11: "say it again" wording caused 5x intro replays).
        assert "ஒரு சின்ன" in text
        assert "narration வேண்டாம்" in text
        # The old blanket ban that made the agent "forget" unheard questions is gone.
        assert "Never go back" not in text

    asyncio.run(scenario())


def test_pre_welcome_speech_is_not_an_interrupt():
    async def scenario():
        handler = VobizRealtimeHandler(tts_provider="sarvam", language="ta")
        handler._ai_is_responding = True   # welcome still generating
        handler._is_first_response = True
        assert handler._any_real_audio_sent is False
        # Caller says "hello?" into the pre-welcome silence.
        await handler._handle_openai_event({
            "type": "input_audio_buffer.speech_started",
        })
        # The welcome pipeline is untouched — no interrupt armed.
        assert handler._interrupt_pending is False

    asyncio.run(scenario())


def test_pre_welcome_auto_reply_is_suppressed_and_welcome_survives():
    class FakeOpenAIWebSocket:
        def __init__(self):
            self.sent = []

        async def send(self, message):
            self.sent.append(json.loads(message))

    async def scenario():
        handler = VobizRealtimeHandler(tts_provider="sarvam", language="ta")
        ws = FakeOpenAIWebSocket()
        handler.openai_ws = ws
        handler._is_first_response = False   # welcome response already done
        assert handler._any_real_audio_sent is False
        # The welcome piece is still queued for TTS.
        await handler._tts_queue.put("வணக்கம்ங்க! நான் திவ்யா பேசறேன்.")

        # Server VAD auto-creates a reply to the caller's pre-welcome "hello?".
        await handler._handle_openai_event({"type": "response.created"})
        assert {"type": "response.cancel"} in ws.sent
        assert handler._discard_response_text is True

        # Its text deltas are discarded — never buffered, never synthesized.
        await handler._handle_openai_event(
            {"type": "response.text.delta", "delta": "ஆமா, பேசு சொல்லுங்க."}
        )
        assert handler._response_text_buffer == ""

        # The cancelled response.done must NOT drain the queued welcome.
        await handler._handle_openai_event(
            {"type": "response.done", "response": {"status": "cancelled"}}
        )
        assert handler._tts_queue.qsize() == 1

    asyncio.run(scenario())


def test_zero_audio_interrupt_reports_whole_reply_unheard():
    class FakeOpenAIWebSocket:
        def __init__(self):
            self.sent = []

        async def send(self, message):
            self.sent.append(json.loads(message))

    async def scenario():
        handler = VobizRealtimeHandler(tts_provider="sarvam", language="ta")
        ws = FakeOpenAIWebSocket()
        handler.openai_ws = ws
        handler._vobiz_ws = _FakeVobizCapture()
        handler.stream_id = "stream-1"
        handler._is_first_response = False
        handler._any_real_audio_sent = True
        handler._ai_is_responding = False    # reply completed…
        handler._interrupt_pending = True
        handler._speech_start_time = time.monotonic() - 1.0
        handler._sent_pieces = []            # …but NO audio ever went out
        handler._unheard_text_this_response = ""
        handler._last_response_text = "Expected salary என்ன range-ல பாக்கறீங்க?"

        await handler._handle_openai_event({
            "type": "input_audio_buffer.speech_stopped",
        })

        injected = [m for m in ws.sent
                    if m.get("type") == "conversation.item.create"]
        assert len(injected) == 1
        # The WHOLE reply is reported unheard so the model re-asks it.
        assert "Expected salary" in injected[0]["item"]["content"][0]["text"]

    asyncio.run(scenario())


def test_playback_cut_splits_heard_and_unheard():
    async def scenario():
        handler = VobizRealtimeHandler(tts_provider="sarvam", language="ta")
        now = time.monotonic()
        handler._sent_pieces = [("piece one", 2.0), ("piece two", 3.0)]
        handler._heard_text_this_response = "piece one piece two"
        # Audio started 2.5s ago; ~2.4s played (minus carrier lag) — piece one
        # (2.0s) fully played, piece two barely started.
        handler._response_audio_started_at = now - 2.5
        handler._estimated_playback_end = now + 2.6

        await handler._clear_vobiz_audio()

        assert handler._heard_text_this_response == "piece one"
        assert handler._unheard_text_this_response == "piece two"
        assert handler._estimated_playback_end == 0.0

    asyncio.run(scenario())


def test_barge_in_after_response_done_recovers_unheard_speech():
    """THE greeting-skip bug: response TEXT completes ~1s in, audio plays for
    seconds more. A barge-in then has no active response to cancel, so no
    response.done(cancelled) recovery ever fired — the model believed the
    unheard greeting/question was delivered and skipped it."""
    class FakeOpenAIWebSocket:
        def __init__(self):
            self.sent = []

        async def send(self, message):
            self.sent.append(json.loads(message))

    async def scenario():
        handler = VobizRealtimeHandler(tts_provider="sarvam", language="ta")
        ws = FakeOpenAIWebSocket()
        handler.openai_ws = ws
        handler._vobiz_ws = _FakeVobizCapture()
        handler.stream_id = "stream-1"
        handler._is_first_response = False
        handler._ai_is_responding = False           # response already completed
        handler._interrupt_pending = True           # armed at speech_started
        handler._speech_start_time = time.monotonic() - 1.0  # > backchannel cap
        handler._unheard_text_this_response = "முதல்ல உங்க பேரு சொல்லுங்க?"
        handler._heard_text_this_response = "வணக்கம்ங்க, நான் Alex பேசறேன்."

        await handler._handle_openai_event({
            "type": "input_audio_buffer.speech_stopped",
        })

        types = [m.get("type") for m in ws.sent]
        assert "conversation.item.create" in types   # recovery injected
        assert "response.cancel" in types            # auto-created answer killed
        assert handler._response_create_pending is True
        injected = next(m for m in ws.sent
                        if m.get("type") == "conversation.item.create")
        text = injected["item"]["content"][0]["text"]
        assert "முதல்ல உங்க பேரு சொல்லுங்க?" in text  # unheard part named

        # If the cancel finds nothing active, the pending create must still fire.
        ws.sent.clear()
        await handler._handle_openai_event({
            "type": "error",
            "error": {"code": "response_cancel_not_active", "message": "x"},
        })
        assert {"type": "response.create"} in ws.sent
        assert handler._response_create_pending is False

    asyncio.run(scenario())


def test_interrupt_recovery_when_nothing_was_heard():
    handler = VobizRealtimeHandler(tts_provider="sarvam", language="ta")
    handler._heard_text_this_response = ""
    text = handler._build_interrupt_recovery()
    assert "காதுல விழவே இல்லை" in text

    handler_en = VobizRealtimeHandler(tts_provider="elevenlabs", language="en")
    handler_en._heard_text_this_response = "Sure, let me confirm that."
    text_en = handler_en._build_interrupt_recovery()
    assert 'heard ONLY this much: "Sure, let me confirm that."' in text_en


def test_repeated_ack_opener_word_is_trimmed():
    async def scenario():
        handler = VobizRealtimeHandler(tts_provider="sarvam", language="ta")
        # Response 1: "சரி, ..." goes through untouched.
        handler._openers_seen_this_response = False
        await handler._enqueue_tts("சரி, உங்க பேரு சொல்லுங்க.")
        assert await handler._tts_queue.get() == "சரி, உங்க பேரு சொல்லுங்க."
        # Response 2 also opens with "சரி," — the repeated ack word is trimmed.
        handler._openers_seen_this_response = False
        await handler._enqueue_tts("சரி, எத்தனை வருஷம் experience?")
        assert await handler._tts_queue.get() == "எத்தனை வருஷம் experience?"
        # Response 3: alternation — "சரி," speaks again.
        handler._openers_seen_this_response = False
        await handler._enqueue_tts("சரி, அடுத்த கேள்வி.")
        assert await handler._tts_queue.get() == "சரி, அடுத்த கேள்வி."
        # A non-ack opener is never trimmed.
        handler._openers_seen_this_response = False
        await handler._enqueue_tts("உங்க email சொல்லுங்க.")
        assert await handler._tts_queue.get() == "உங்க email சொல்லுங்க."

    asyncio.run(scenario())


def test_spelled_letters_cancel_inflight_response():
    class FakeOpenAIWebSocket:
        def __init__(self):
            self.sent = []

        async def send(self, message):
            self.sent.append(json.loads(message))

    class FakeVobiz:
        async def send_json(self, message):
            return None

    async def scenario():
        handler = VobizRealtimeHandler(tts_provider="sarvam", language="ta")
        ws = FakeOpenAIWebSocket()
        handler.openai_ws = ws
        handler._vobiz_ws = FakeVobiz()
        handler.stream_id = "stream-1"
        # A response auto-created by server VAD is mid-flight — it has never
        # seen the spelled letters (transcription always lands later).
        handler._ai_is_responding = True

        await handler._handle_openai_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "H-A-J-I-K",
        })

        types = [m.get("type") for m in ws.sent]
        assert "conversation.item.create" in types   # forced confirmation injected
        assert "response.cancel" in types            # stale response cancelled
        assert handler._response_create_pending is True
        assert handler._suppress_recovery_once is True

        # The cancelled response.done must NOT inject "caller interrupted you".
        handler._current_ai_transcript = "A as in Alpha, B as in Bravo, C as in..."
        ws.sent.clear()
        await handler._handle_openai_event({
            "type": "response.done",
            "response": {"status": "cancelled"},
        })
        injected = [m for m in ws.sent if m.get("type") == "conversation.item.create"]
        assert injected == []
        # response.create retried so the forced confirmation applies this turn.
        assert {"type": "response.create"} in ws.sent
        assert handler._suppress_recovery_once is False

    asyncio.run(scenario())


def test_max_output_tokens_caps_tamil_script_calls():
    # Tamil script calls: hard cap so the model can't ramble (turns measured
    # 20-70 tokens; 400 allowed 4-sentence replies).
    ta = VobizRealtimeHandler(tts_provider="sarvam", language="ta",
                              active_script={"questions": []})
    assert ta._max_output_tokens() == 180
    # English script path keeps its proven 400; no-script keeps 150.
    en = VobizRealtimeHandler(tts_provider="openai", language="en",
                              active_script={"questions": []})
    assert en._max_output_tokens() == 400
    free = VobizRealtimeHandler(tts_provider="sarvam", language="ta")
    assert free._max_output_tokens() == 150


def test_called_number_lands_in_prompts():
    handler = VobizRealtimeHandler(
        tts_provider="sarvam", language="ta",
        active_script={"questions": []}, called_number="+918110016139",
    )
    assert "+918110016139" in handler._build_prompt()

    handler_en = VobizRealtimeHandler(
        tts_provider="openai", language="en",
        active_script={"questions": []}, called_number="+918110016139",
    )
    assert "+918110016139" in handler_en._build_prompt()


def test_heard_text_resets_each_response():
    async def scenario():
        handler = VobizRealtimeHandler(tts_provider="sarvam", language="ta")
        handler._record_heard_text("first piece")
        handler._record_heard_text("second piece")
        assert handler._heard_text_this_response == "first piece second piece"
        await handler._handle_openai_event({"type": "response.created"})
        assert handler._heard_text_this_response == ""

    asyncio.run(scenario())


def test_attenuate_ulaw_reduces_volume():
    import audioop as ao
    pcm = b"".join(
        (20000 if i % 2 else -20000).to_bytes(2, "little", signed=True)
        for i in range(200)
    )
    ulaw = ao.lin2ulaw(pcm, 2)
    soft = _attenuate_ulaw(ulaw, 0.5)
    assert ao.max(ao.ulaw2lin(soft, 2), 2) < ao.max(ao.ulaw2lin(ulaw, 2), 2)


class _FakeVobizCapture:
    def __init__(self):
        self.payloads = []

    async def send_json(self, message):
        if message.get("event") == "playAudio":
            self.payloads.append(message["media"]["payload"])


def test_listening_backchannel_hums_during_long_caller_monologue():
    async def scenario():
        handler = VobizRealtimeHandler(tts_provider="sarvam", language="ta")
        vobiz = _FakeVobizCapture()
        handler._vobiz_ws = vobiz
        handler.stream_id = "stream-1"
        handler._is_first_response = False
        handler._listen_bc_clips = [b"\xff" * 800]
        # Caller started talking and has NOT stopped.
        handler._speech_start_time = 100.0
        handler._speech_stopped_time = 0.0

        original_sleep = asyncio.sleep

        async def fast_sleep(_s):
            await original_sleep(0)

        asyncio.sleep = fast_sleep
        try:
            await handler._play_listen_backchannels(100.0)
        finally:
            asyncio.sleep = original_sleep

        # Exactly one hum: the second attempt is throttled by the min-gap.
        assert len(vobiz.payloads) == 1

    asyncio.run(scenario())


def test_listening_backchannel_aborts_when_utterance_ended():
    async def scenario():
        handler = VobizRealtimeHandler(tts_provider="sarvam", language="ta")
        vobiz = _FakeVobizCapture()
        handler._vobiz_ws = vobiz
        handler.stream_id = "stream-1"
        handler._is_first_response = False
        handler._listen_bc_clips = [b"\xff" * 800]
        # speech_stopped already stamped AFTER this utterance began.
        handler._speech_start_time = 100.0
        handler._speech_stopped_time = 105.0

        original_sleep = asyncio.sleep

        async def fast_sleep(_s):
            await original_sleep(0)

        asyncio.sleep = fast_sleep
        try:
            await handler._play_listen_backchannels(100.0)
        finally:
            asyncio.sleep = original_sleep

        assert vobiz.payloads == []

    asyncio.run(scenario())


def test_intent_and_spelling_helpers():
    assert _is_disengage_intent("Sorry, I am busy right now")
    assert _is_closing_remark("Okay, thank you, bye")
    assert not _is_closing_remark("Okay, one more question")
    assert _extract_spelled_letters("H-A-J-I-K") == "HAJIK"


def test_cleanup_closes_tts_clients_once():
    class FakeClient:
        def __init__(self):
            self.closed = 0

        async def close(self):
            self.closed += 1

    async def scenario():
        handler = VobizRealtimeHandler()
        external_tts = FakeClient()
        openai_tts = FakeClient()
        handler._external_tts = external_tts
        handler._openai_tts_client = openai_tts

        await handler._full_cleanup()
        await handler._full_cleanup()

        assert external_tts.closed == 1
        assert openai_tts.closed == 1

    asyncio.run(scenario())

def test_native_audio_truncation_tracks_only_heard_audio():
    class FakeOpenAIWebSocket:
        def __init__(self):
            self.messages = []

        async def send(self, message):
            self.messages.append(json.loads(message))

    async def scenario():
        handler = VobizRealtimeHandler(use_elevenlabs=False)
        ws = FakeOpenAIWebSocket()
        handler.openai_ws = ws
        handler._current_response_item_id = "assistant-item-1"
        handler._response_audio_sent_ms = 1000.0
        handler._response_playback_started_at = time.monotonic() - 0.2

        await handler._truncate_current_audio()

        assert len(ws.messages) == 1
        event = ws.messages[0]
        assert event["type"] == "conversation.item.truncate"
        assert event["item_id"] == "assistant-item-1"
        assert event["content_index"] == 0
        assert 50 <= event["audio_end_ms"] <= 250
        assert handler._current_response_item_id is None

    asyncio.run(scenario())


def test_native_audio_updates_playback_queue_estimate():
    handler = VobizRealtimeHandler(use_elevenlabs=False)
    before = time.monotonic()
    handler._record_native_audio_sent(8000)

    assert handler._response_audio_sent_ms == 1000.0
    assert handler._estimated_playback_end >= before + 0.9

def test_session_config_falls_back_when_truncation_is_rejected():
    class FakeOpenAIWebSocket:
        def __init__(self):
            self.sent = []
            self.events = [
                {"type": "error", "error": {"message": "unsupported truncation"}},
                {"type": "error", "error": {"message": "unsupported truncation"}},
                {
                    "type": "session.updated",
                    "session": {
                        "audio": {
                            "input": {"format": {"type": "audio/pcmu"}},
                            "output": {"format": {"type": "audio/pcmu"}},
                        }
                    },
                },
            ]

        async def send(self, message):
            self.sent.append(json.loads(message))

        async def recv(self):
            return json.dumps(self.events.pop(0))

    async def scenario():
        original_vad = Config.VAD_TYPE
        Config.VAD_TYPE = "semantic_vad"
        try:
            handler = VobizRealtimeHandler(use_elevenlabs=False)
            ws = FakeOpenAIWebSocket()
            handler.openai_ws = ws

            assert await handler._configure_session() is True

            session_updates = [
                event for event in ws.sent if event.get("type") == "session.update"
            ]
            assert len(session_updates) == 3
            assert "truncation" in session_updates[0]["session"]
            assert "truncation" in session_updates[1]["session"]
            assert "truncation" not in session_updates[2]["session"]
        finally:
            Config.VAD_TYPE = original_vad

    asyncio.run(scenario())

def test_metrics_report_latency_cache_and_transcription_usage():
    handler = VobizRealtimeHandler(language="ta")
    handler._turn_count = 3
    handler._interrupt_count = 1
    handler._latency_samples_ms = [300.0, 500.0]
    handler._total_realtime_tokens = 120
    handler._total_input_tokens = 80
    handler._cached_input_tokens = 40
    handler._total_output_tokens = 40
    handler._transcription_tokens = 12

    metrics = handler.get_metrics()

    assert metrics["avg_e2e_latency_ms"] == 400.0
    assert metrics["max_e2e_latency_ms"] == 500.0
    assert metrics["cached_input_tokens"] == 40
    assert metrics["transcription_tokens"] == 12
    assert metrics["language"] == "ta"

def test_rejected_response_create_is_retried_at_response_done():
    class FakeOpenAIWebSocket:
        def __init__(self):
            self.sent = []

        async def send(self, message):
            self.sent.append(json.loads(message))

    async def scenario():
        handler = VobizRealtimeHandler()
        ws = FakeOpenAIWebSocket()
        handler.openai_ws = ws

        await handler._handle_openai_event({
            "type": "error",
            "error": {
                "code": "conversation_already_has_active_response",
                "message": "Conversation already has an active response",
            },
        })
        assert handler._response_create_pending is True
        assert not ws.sent  # retry waits for the active response to finalize

        await handler._handle_openai_event({
            "type": "response.done",
            "response": {"status": "cancelled"},
        })
        assert handler._response_create_pending is False
        assert {"type": "response.create"} in ws.sent

    asyncio.run(scenario())


def test_backchannel_flushes_buffered_audio_and_keeps_queue():
    class FakeVobiz:
        def __init__(self):
            self.payloads = []

        async def send_json(self, message):
            if message.get("event") == "playAudio":
                self.payloads.append(message["media"]["payload"])

    async def scenario():
        import base64 as b64mod
        handler = VobizRealtimeHandler()
        vobiz = FakeVobiz()
        handler._vobiz_ws = vobiz
        handler.stream_id = "stream-1"
        handler._is_first_response = False
        handler._ai_is_responding = True
        handler._any_real_audio_sent = True  # mid-call, caller has heard audio

        delta = b64mod.b64encode(b"\xff" * 160).decode()
        await handler._handle_openai_event({
            "type": "input_audio_buffer.speech_started",
        })
        assert handler._interrupt_pending is True

        # Deltas during the evaluation window are buffered, not dropped.
        await handler._handle_openai_event(
            {"type": "response.audio.delta", "delta": delta}
        )
        assert vobiz.payloads == []
        assert len(handler._pending_audio_deltas) == 1

        # A response completing mid-evaluation must NOT resolve the interrupt.
        await handler._handle_openai_event({
            "type": "response.done",
            "response": {"status": "completed"},
        })
        assert handler._interrupt_pending is True

        # Short speech = backchannel — buffered audio is flushed to the caller.
        handler._speech_start_time = time.monotonic() - 0.1  # 100ms < 250ms cap
        await handler._handle_openai_event({
            "type": "input_audio_buffer.speech_stopped",
        })
        assert handler._interrupt_pending is False
        assert vobiz.payloads == [delta]
        assert handler._pending_audio_deltas == []

    asyncio.run(scenario())


def test_stuck_interrupt_pending_is_cleared_by_watchdog():
    async def scenario():
        handler = VobizRealtimeHandler()
        handler._is_first_response = False
        handler._ai_is_responding = True
        handler._interrupt_pending = True
        handler._speech_start_time = 123.0

        # speech_stopped never arrives — the watchdog must unwedge the flag
        # (a stuck flag mutes the agent for the rest of the call).
        original_sleep = asyncio.sleep

        async def fast_sleep(_s):
            await original_sleep(0)

        asyncio.sleep = fast_sleep
        try:
            await handler._resolve_stuck_interrupt(123.0)
            assert handler._interrupt_pending is False

            # A NEWER speech event must not be resolved by an old watchdog.
            handler._interrupt_pending = True
            handler._speech_start_time = 456.0
            await handler._resolve_stuck_interrupt(123.0)
            assert handler._interrupt_pending is True
        finally:
            asyncio.sleep = original_sleep

    asyncio.run(scenario())


def test_interruption_discards_inflight_external_tts_audio():
    class DelayedTTS:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def synthesize(self, text):
            self.started.set()
            await self.release.wait()
            return b"x" * 8000

        async def close(self):
            return None

    class FakeVobiz:
        def __init__(self):
            self.messages = []

        async def send_json(self, message):
            self.messages.append(message)

    async def scenario():
        handler = VobizRealtimeHandler(use_elevenlabs=True)
        tts = DelayedTTS()
        vobiz = FakeVobiz()
        handler._external_tts = tts
        handler._vobiz_ws = vobiz
        handler.stream_id = "stream-1"
        generation = handler._tts_generation

        task = asyncio.create_task(
            handler._synthesize_and_send("This response is now stale.", generation)
        )
        await tts.started.wait()
        handler._drain_tts_queue()
        tts.release.set()
        await task

        assert vobiz.messages == []

    asyncio.run(scenario())


def test_pending_create_dropped_when_response_completed():
    # Live 2026-07-11 01:49: the barge-in recovery armed a re-create, but its
    # response.cancel missed — the server's auto-answer COMPLETED, and the
    # retried create generated a second full answer (re-greeting + repeated
    # question) stacked behind the first in the TTS queue.
    class FakeOpenAIWebSocket:
        def __init__(self):
            self.sent = []

        async def send(self, message):
            self.sent.append(json.loads(message))

    async def scenario():
        handler = VobizRealtimeHandler()
        ws = FakeOpenAIWebSocket()
        handler.openai_ws = ws
        handler._is_first_response = False
        handler._response_create_pending = True

        await handler._handle_openai_event({
            "type": "response.done",
            "response": {"status": "completed"},
        })

        assert handler._response_create_pending is False
        assert {"type": "response.create"} not in ws.sent

    asyncio.run(scenario())


def test_response_created_clears_stale_text_buffer():
    # Live 2026-07-11 01:51: a withheld tail from a cancelled response merged
    # into the next response's text — one TTS chunk read
    # "What is your current CTC?I'm sorry," with no boundary.
    async def scenario():
        handler = VobizRealtimeHandler()
        handler._is_first_response = False
        handler._any_real_audio_sent = True
        handler._response_text_buffer = "What is your current CTC?"

        await handler._handle_openai_event({"type": "response.created"})

        assert handler._response_text_buffer == ""

    asyncio.run(scenario())


def test_backchannel_auto_reply_is_suppressed():
    # Live 2026-07-11 01:49: a 564ms "mm-hm" was correctly ruled a backchannel,
    # but server VAD still auto-created an answer to it — the agent re-asked a
    # question the caller had already heard (3x in a row).
    class FakeOpenAIWebSocket:
        def __init__(self):
            self.sent = []

        async def send(self, message):
            self.sent.append(json.loads(message))

    async def scenario():
        handler = VobizRealtimeHandler(use_elevenlabs=True)
        ws = FakeOpenAIWebSocket()
        handler.openai_ws = ws
        handler._is_first_response = False
        handler._any_real_audio_sent = True
        handler._ai_is_responding = False
        handler._estimated_playback_end = time.monotonic() + 5.0

        await handler._handle_openai_event({
            "type": "input_audio_buffer.speech_started",
        })
        assert handler._interrupt_pending is True

        handler._speech_start_time = time.monotonic() - 0.1  # 100ms → backchannel
        await handler._handle_openai_event({
            "type": "input_audio_buffer.speech_stopped",
        })
        assert handler._suppress_bc_response_until > time.monotonic()

        await handler._handle_openai_event({"type": "response.created"})
        assert {"type": "response.cancel"} in ws.sent
        assert handler._discard_response_text is True
        assert handler._suppress_bc_response_until == 0.0  # one-shot

        # The suppressed reply's text never reaches the TTS buffer.
        await handler._handle_openai_event({
            "type": "response.text.delta", "delta": "May I know your name?",
        })
        assert handler._response_text_buffer == ""

    asyncio.run(scenario())


def test_backchannel_suppression_not_armed_while_generating():
    # An actively-generating response must be left alone: its rejected
    # auto-create is dropped at response.done instead (see
    # test_pending_create_dropped_when_response_completed).
    async def scenario():
        handler = VobizRealtimeHandler()
        handler._is_first_response = False
        handler._any_real_audio_sent = True
        handler._ai_is_responding = True

        await handler._handle_openai_event({
            "type": "input_audio_buffer.speech_started",
        })
        handler._speech_start_time = time.monotonic() - 0.1
        await handler._handle_openai_event({
            "type": "input_audio_buffer.speech_stopped",
        })

        assert handler._suppress_bc_response_until == 0.0

    asyncio.run(scenario())


def test_completed_response_tail_is_flushed_at_done():
    # A cancel that missed leaves _cancel_requested stale-True, which withheld
    # the completed response's tail at text.done — the caller heard only half
    # the answer. response.done must flush it.
    async def scenario():
        handler = VobizRealtimeHandler(use_elevenlabs=True)
        handler._is_first_response = False
        handler._cancel_requested = True  # stale — the cancel never landed
        handler._response_text_buffer = "May I know your name?"
        enqueued = []

        async def fake_enqueue(text):
            enqueued.append(text)

        handler._enqueue_tts = fake_enqueue

        await handler._handle_openai_event({
            "type": "response.done",
            "response": {"status": "completed"},
        })

        assert enqueued == ["May I know your name?"]
        assert handler._response_text_buffer == ""
        assert handler._cancel_requested is False

    asyncio.run(scenario())


def test_refusals_are_disengage_intents():
    # Live 2026-07-11 13:01: "No, I not interested. Thank you." matched no
    # disengage phrase — the agent replied "no problem, could you share your
    # name?" and the caller hung up.
    from services.vobiz_stream_service import _is_disengage_intent

    assert _is_disengage_intent("No, I not interested. Thank you.")
    assert _is_disengage_intent("I'm not interested in this job")
    assert _is_disengage_intent("please stop calling me")
    assert _is_disengage_intent("you have the wrong number")
    assert _is_disengage_intent("I don't want this")
    # Positive engagement must not match.
    assert not _is_disengage_intent("yes I am interested")
    assert not _is_disengage_intent("sounds interesting, tell me more")


def test_disengage_cancels_inflight_response_and_forces_goodbye():
    # The server-VAD auto-answer races the disengage injection; it must be
    # cancelled (and re-created) so the scripted goodbye plays instead of a
    # free-styled pushy follow-up.
    class FakeOpenAIWebSocket:
        def __init__(self):
            self.sent = []

        async def send(self, message):
            self.sent.append(json.loads(message))

    class FakeVobiz:
        async def send_json(self, message):
            return None

    async def scenario():
        handler = VobizRealtimeHandler(use_elevenlabs=True)
        ws = FakeOpenAIWebSocket()
        handler.openai_ws = ws
        handler._vobiz_ws = FakeVobiz()
        handler.stream_id = "stream-1"
        handler._is_first_response = False
        handler._ai_is_responding = True

        await handler._handle_openai_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "No, I not interested. Thank you.",
        })

        types = [m.get("type") for m in ws.sent]
        assert "conversation.item.create" in types   # goodbye instruction injected
        assert "response.cancel" in types            # pushy auto-answer cancelled
        assert handler._response_create_pending is True

        # With no response in flight, the goodbye must be nudged out directly.
        handler2 = VobizRealtimeHandler(use_elevenlabs=True)
        ws2 = FakeOpenAIWebSocket()
        handler2.openai_ws = ws2
        handler2._vobiz_ws = FakeVobiz()
        handler2.stream_id = "stream-2"
        handler2._is_first_response = False
        handler2._ai_is_responding = False

        await handler2._handle_openai_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "I am busy now, call me later",
        })
        assert {"type": "response.create"} in ws2.sent

    asyncio.run(scenario())


def test_recovery_text_bans_full_reintroduction():
    # Live 2026-07-11: "say it again naturally" made the mini model re-deliver
    # the whole pitch ("Let me start over...") after every hello-barge-in.
    handler = VobizRealtimeHandler(use_elevenlabs=True)
    handler._heard_text_this_response = "Hi!"
    handler._unheard_text_this_response = "This is Akash calling from Zillion Connects."

    text = handler._build_interrupt_recovery()

    assert "ONE short sentence" in text
    assert "let me start over" in text          # listed as a banned phrase
    assert "SHORTER" in text
    assert "full pitch" in text


def test_english_transcription_config_pins_language_and_prompt():
    # Foreign-script hallucinations ("hello?" → "哈喽"/"Борил?") poisoned the
    # transcript-based intent checks — the EN config biases the decoder.
    handler = VobizRealtimeHandler(use_elevenlabs=True, language="en")
    cfg = handler._transcription_config()
    assert cfg["language"] == "en"
    assert "English" in cfg.get("prompt", "")


def test_normalize_phone_number_defaults_to_india():
    # Live 2026-07-11 13:20: '+7010873682' (bare + before a 10-digit Indian
    # mobile) went out with a Russia prefix instead of +91.
    from main import _normalize_phone_number

    assert _normalize_phone_number("+7010873682") == "+917010873682"
    assert _normalize_phone_number("7010873682") == "+917010873682"
    assert _normalize_phone_number("07010873682") == "+917010873682"
    assert _normalize_phone_number("917010873682") == "+917010873682"
    assert _normalize_phone_number("+917010873682") == "+917010873682"
    assert _normalize_phone_number("70108 73682") == "+917010873682"
    # Genuine international numbers pass through untouched.
    assert _normalize_phone_number("+14155551234") == "+14155551234"
    assert _normalize_phone_number("+79261234567") == "+79261234567"
    assert _normalize_phone_number("") == ""


def test_tts_stale_check_holds_current_speech_through_watchdog_window():
    # Live 2026-07-11 02:13: with a lost speech_stopped, the old 1.5s stale
    # bound dropped the reply's lead-in a second before the 2.5s watchdog
    # cleared the stuck flag ("Sure, take your time." swallowed, ~3.5s dead
    # air). Current-generation speech must be HELD until the flag resolves.
    async def scenario():
        handler = VobizRealtimeHandler(use_elevenlabs=True)
        handler._interrupt_pending = True
        generation = handler._tts_generation

        async def clear_flag_late():
            await asyncio.sleep(2.0)  # past the old 1.5s drop bound
            handler._interrupt_pending = False

        asyncio.create_task(clear_flag_late())
        assert await handler._tts_utterance_stale(generation) is False

        # A real interruption ruling (generation bump) still marks it stale.
        handler._interrupt_pending = True

        async def bump():
            await asyncio.sleep(0.2)
            handler._drain_tts_queue()

        asyncio.create_task(bump())
        assert await handler._tts_utterance_stale(generation) is True

    asyncio.run(scenario())