"""Regression tests for the 2026-07-20 bug-fix pass.

Each test pins a bug that shipped to live calls and that the existing suite did
not catch. Named for the symptom the caller experienced, not the code path.
"""

import asyncio
import gc
import json
import time

from config import Config

from services.vobiz_stream_service import VobizRealtimeHandler


class _FakeVobiz:
    """Minimal Vobiz media socket that records whether the call was closed."""

    def __init__(self):
        self.payloads = []
        self.closed = False

    async def send_json(self, message):
        if message.get("event") == "playAudio":
            self.payloads.append(message["media"]["payload"])

    async def close(self):
        self.closed = True


class _ExhaustedSocket:
    """An OpenAI socket whose iteration ends immediately (i.e. it closed)."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


def _handler():
    return VobizRealtimeHandler(tts_provider="sarvam", language="ta")


# ── Backchannel duration ────────────────────────────────────────────────────


def test_backchannel_duration_excludes_vad_silence_window():
    """A short "mm-hm" must read as a backchannel, not a full interruption.

    speech_stopped only fires after the server waits out
    VAD_SILENCE_DURATION_MS of silence, so wall-clock over-reports the caller's
    speech by that entire window. With the live settings (silence 500ms, cap
    600ms) a genuine 150ms hum measured ~650ms and cut the agent off mid-reply.
    """
    handler = _handler()
    # Both reported boundaries are padded: audio_start_ms includes
    # prefix_padding_ms and audio_end_ms includes silence_duration_ms, so the
    # raw span over-reports true speech by their sum. Taking the span raw left
    # a 150ms hum measuring 770ms — still over the 600ms cap, i.e. not fixed.
    pad = Config.VAD_PREFIX_PADDING_MS + Config.VAD_SILENCE_DURATION_MS

    handler._speech_start_audio_ms = 2000
    assert handler._measure_speech_duration_ms(
        {"audio_end_ms": 2000 + pad + 150}
    ) == 150.0
    assert handler._measure_speech_duration_ms(
        {"audio_end_ms": 2000 + pad + 2000}
    ) == 2000.0

    # A 150ms hum must land UNDER the cap — the whole point of the fix.
    handler._speech_start_time = time.monotonic()
    assert handler._measure_speech_duration_ms(
        {"audio_end_ms": 2000 + pad + 150}
    ) < Config.BACKCHANNEL_MAX_DURATION_MS

    # A genuine 2s interruption must still land OVER it.
    assert handler._measure_speech_duration_ms(
        {"audio_end_ms": 2000 + pad + 2000}
    ) > Config.BACKCHANNEL_MAX_DURATION_MS

    # Fallback path (fields absent): wall-clock minus the silence window.
    handler._speech_start_audio_ms = None
    handler._speech_start_time = time.monotonic() - (
        (Config.VAD_SILENCE_DURATION_MS + 150) / 1000.0
    )
    measured = handler._measure_speech_duration_ms({})
    assert 100 <= measured <= 260, measured
    assert measured < Config.BACKCHANNEL_MAX_DURATION_MS

    # Never negative, however the clocks land.
    handler._speech_start_time = time.monotonic()
    assert handler._measure_speech_duration_ms({}) == 0.0

    # If the events are NOT padded as documented the subtraction goes negative.
    # That must fall back to wall-clock, not clamp to 0 — a hard 0 is always
    # under the cap, which would rule every turn a backchannel and silently
    # disable barge-in entirely (a worse failure than the one being fixed).
    handler._speech_start_audio_ms = 2000
    handler._speech_start_time = time.monotonic() - 3.0   # 3s of real speech
    unpadded = handler._measure_speech_duration_ms({"audio_end_ms": 2050})
    assert unpadded > Config.BACKCHANNEL_MAX_DURATION_MS, unpadded


def test_short_backchannel_leaves_the_agents_reply_intact():
    """End-to-end: a 150ms hum must not drain the queued reply."""

    async def scenario():
        handler = _handler()
        handler._vobiz_ws = _FakeVobiz()
        handler.stream_id = "stream-1"
        handler._is_first_response = False
        handler._ai_is_responding = True
        handler._interrupt_pending = True
        handler._speech_start_audio_ms = 2000
        handler._response_text_buffer = "still speaking"
        generation_before = handler._tts_generation

        pad = Config.VAD_PREFIX_PADDING_MS + Config.VAD_SILENCE_DURATION_MS
        await handler._handle_openai_event({
            "type": "input_audio_buffer.speech_stopped",
            "audio_end_ms": 2000 + pad + 150,     # 150ms of real speech
        })

        assert handler._tts_generation == generation_before, "reply was drained"
        assert handler._response_text_buffer == "still speaking"
        assert handler._interrupt_pending is False

    asyncio.run(scenario())


# ── Session restart ─────────────────────────────────────────────────────────


def test_restart_session_does_not_drop_the_call():
    """"Start over" must reset the session, not hang up on the caller.

    Drives the REAL path: the receive loop calls _restart_session from inside
    itself (as _handle_openai_event does), so the loop then unwinds into its own
    finally with the replacement connection already live. That finally used to
    read the NEW connection's _connected=True and close the media stream.
    """

    class _RestartTriggeringSocket:
        """Yields one event then ends — the socket _restart_session closes."""

        def __init__(self):
            self._done = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._done:
                raise StopAsyncIteration
            self._done = True
            return json.dumps({"type": "restart.trigger"})

        async def close(self):
            pass

    async def scenario():
        handler = _handler()
        vobiz = _FakeVobiz()
        handler._vobiz_ws = vobiz
        handler.stream_id = "stream-1"
        handler._connected = True

        new_socket = _ExhaustedSocket()

        async def fake_connect():
            # What _connect_openai does: install a new socket and mark the
            # bridge live. A replacement receive task runs on that new socket.
            handler.openai_ws = new_socket
            handler._connected = True

        handler._connect_openai = fake_connect
        handler.openai_ws = _RestartTriggeringSocket()

        async def handle(event):
            if event.get("type") == "restart.trigger":
                await handler._restart_session()

        handler._handle_openai_event = handle

        # The old loop restarts mid-iteration, then unwinds into its finally.
        await handler._receive_openai_events()

        assert vobiz.closed is False, "old receive task closed the live call"
        assert handler._connected is True, "bridge was deafened after restart"
        assert handler.openai_ws is new_socket, "restart did not install new socket"

    asyncio.run(scenario())


def test_real_openai_disconnect_still_closes_the_call():
    """The restart carve-out must not swallow a genuine disconnect."""

    async def scenario():
        handler = _handler()
        vobiz = _FakeVobiz()
        handler._vobiz_ws = vobiz
        handler.stream_id = "stream-1"
        handler._restarting = False
        handler._connected = True
        handler.openai_ws = _ExhaustedSocket()

        await handler._receive_openai_events()

        assert handler._connected is False
        if Config.CLOSE_CALL_ON_REALTIME_DISCONNECT:
            assert vobiz.closed is True

    asyncio.run(scenario())


# ── Stale cancel flag ───────────────────────────────────────────────────────


def test_stale_cancel_flag_cleared_so_the_goodbye_can_hang_up():
    """response.cancel with nothing active never gets a response.done, the only
    place the flag is cleared. Left stale it gated the NEXT response, so a
    goodbye never scheduled a hangup and the line sat open in silence."""

    async def scenario():
        handler = _handler()
        handler._vobiz_ws = _FakeVobiz()
        handler.stream_id = "stream-1"

        # The unambiguous "cancel hit nothing" signal clears the flag.
        handler._cancel_requested = True
        await handler._handle_openai_event({
            "type": "error",
            "error": {"code": "response_cancel_not_active", "message": "x"},
        })
        assert handler._cancel_requested is False

    asyncio.run(scenario())


def test_barge_in_cancel_survives_the_response_it_targets():
    """The clear must NOT live on response.created.

    The barge-in recovery path deliberately cancels a server-VAD reply BEFORE
    its response.created arrives (it fires while _ai_is_responding is False), so
    clearing there wipes a live, intentional cancel. The doomed response's
    text.done would then see the flag clear and arm a 4.5s hangup on a goodbye
    the caller never heard — dropping the call.
    """

    async def scenario():
        handler = _handler()
        handler._vobiz_ws = _FakeVobiz()
        handler.stream_id = "stream-1"

        handler._ai_is_responding = False      # cancel aimed ahead of the create
        handler._cancel_requested = True
        await handler._handle_openai_event({"type": "response.created"})

        assert handler._cancel_requested is True, (
            "response.created wiped a deliberate barge-in cancel"
        )

    asyncio.run(scenario())


# ── External TTS availability ───────────────────────────────────────────────


def test_tts_failure_is_a_cooldown_not_a_permanent_downgrade():
    """One transient TTS error used to latch the rest of the call onto OpenAI
    tts-1/alloy — a Tamil script read aloud in an English voice."""
    handler = _handler()

    assert handler._external_tts_ready() is True

    handler._external_tts_retry_at = time.monotonic() + 5.0
    assert handler._external_tts_ready() is False

    # Once the cooldown lapses the provider is retried and the flag disarms.
    handler._external_tts_retry_at = time.monotonic() - 0.01
    assert handler._external_tts_ready() is True
    assert handler._external_tts_retry_at == 0.0


# ── Opener de-dup ───────────────────────────────────────────────────────────


def test_repeated_opener_is_dropped_once_not_forever():
    """The skip path returned before refreshing _prev_opener, so the same short
    reply was discarded on every later turn — dead air that never self-healed."""

    async def scenario():
        handler = _handler()
        handler._prev_opener = "சரிங்க, நன்றி."

        # Turn 1: matches the armed opener → dropped, and the trap disarms.
        handler._openers_seen_this_response = False
        await handler._enqueue_tts("சரிங்க, நன்றி.")
        assert handler._tts_queue.qsize() == 0
        assert handler._prev_opener == ""

        # Turn 2: the identical reply now actually reaches the caller.
        handler._openers_seen_this_response = False
        await handler._enqueue_tts("சரிங்க, நன்றி.")
        assert handler._tts_queue.qsize() > 0

    asyncio.run(scenario())


# ── Background task tracking ────────────────────────────────────────────────


def test_background_task_survives_garbage_collection():
    """The actual claim: a suspended one-shot task must not be collected.

    The event loop keeps only a weak reference to a running task, so a bare
    create_task() whose result nobody stores can be collected mid-sleep. That is
    how _resolve_stuck_interrupt — the watchdog for the 2026-07-10 incident where
    _interrupt_pending wedged True and muted the agent for 30s — could silently
    never fire, bringing the dead-air bug back.
    """

    async def scenario():
        handler = _handler()
        finished = asyncio.Event()

        async def watchdog():
            await asyncio.sleep(0.05)      # suspended: the collectable window
            finished.set()

        handler._spawn_bg(watchdog())      # deliberately keep NO local reference
        gc.collect()                       # force the hazard

        await asyncio.wait_for(finished.wait(), timeout=2.0)
        assert finished.is_set(), "watchdog was collected mid-flight"

    asyncio.run(scenario())


def test_finished_background_tasks_are_reaped():
    """The tracking set must not grow without bound over a long call."""

    async def scenario():
        handler = _handler()

        async def quick():
            return None

        task = handler._spawn_bg(quick())
        await task
        await asyncio.sleep(0)             # let the done-callback run
        assert task not in handler._bg_tasks

    asyncio.run(scenario())


def test_browser_path_handles_ga_realtime_event_names():
    """_configure_session sends the GA session shape, so GA event names are what
    actually arrive. The handler only matched the old Beta names, so the browser
    got its own transcript and then nothing — no audio, no text, no error."""

    class _FakeClientWS:
        def __init__(self):
            self.sent = []

        async def send_json(self, message):
            self.sent.append(message)

    async def scenario():
        from services.realtime_service import RealtimeEventHandler, RealtimeService

        client = _FakeClientWS()
        handler = RealtimeEventHandler(client, RealtimeService())

        await handler.handle_event({
            "type": "response.output_audio.delta",
            "delta": "AAAA",
        })
        await handler.handle_event({
            "type": "response.output_audio_transcript.delta",
            "delta": "hello",
        })

        kinds = [m.get("type") for m in client.sent]
        assert "response_audio" in kinds, "GA audio deltas never reached browser"
        assert "response_text" in kinds, "GA transcript deltas never reached browser"
        assert client.sent[0]["data"] == "AAAA"

    asyncio.run(scenario())


def test_browser_barge_in_tracks_server_created_responses():
    """Under server_vad OpenAI creates responses itself; create_response() is
    never called, so is_responding stayed False forever and barge-in was dead."""

    async def scenario():
        import json as _json

        from services.realtime_service import RealtimeService

        class _ScriptedSocket:
            """Replays a server-VAD event sequence — no client response.create."""

            def __init__(self, events):
                self._events = list(events)

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._events:
                    raise StopAsyncIteration
                return _json.dumps(self._events.pop(0))

        service = RealtimeService()
        service._connected = True
        service.ws = _ScriptedSocket([
            {"type": "response.created"},
            {"type": "response.output_audio.delta", "delta": "AAAA"},
            {"type": "response.done"},
        ])

        assert service.is_responding is False

        seen = []
        async for event in service.receive_events():
            seen.append(event["type"])
            if event["type"] == "response.output_audio.delta":
                # Mid-response: this is when a barge-in arrives, and it is the
                # window in which the flag has to be True for cancel to fire.
                assert service.is_responding is True, "barge-in would be a no-op"

        assert seen == [
            "response.created",
            "response.output_audio.delta",
            "response.done",
        ]
        assert service.is_responding is False

    asyncio.run(scenario())


def test_full_cleanup_cancels_background_tasks():
    """A filler/watchdog surviving cleanup writes into a closed Vobiz socket."""

    async def scenario():
        handler = _handler()
        handler._vobiz_ws = _FakeVobiz()
        handler.stream_id = "stream-1"
        started = asyncio.Event()

        async def sleeper():
            started.set()
            await asyncio.sleep(30)

        task = handler._spawn_bg(sleeper())
        await started.wait()

        await handler._full_cleanup()

        assert task.cancelled(), "background task survived cleanup"

    asyncio.run(scenario())


# ── Log level parsing ───────────────────────────────────────────────────────


def test_lowercase_log_level_does_not_crash_startup():
    """getattr(logging, "info") returns the FUNCTION, not 20, so the INFO
    default never fired and basicConfig raised at import."""
    import logging

    from main import _resolve_log_level

    assert _resolve_log_level("info") == logging.INFO
    assert _resolve_log_level("INFO") == logging.INFO
    assert _resolve_log_level("debug") == logging.DEBUG
    assert _resolve_log_level("WARNING") == logging.WARNING
    # Junk and empty values fall back rather than crashing.
    assert _resolve_log_level("nonsense") == logging.INFO
    assert _resolve_log_level("") == logging.INFO
    assert _resolve_log_level(None) == logging.INFO
