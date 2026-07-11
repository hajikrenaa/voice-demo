import asyncio
import base64
import io
import json
import wave

import pytest

from services.sarvam_tts_service import SarvamTTSService
from services.vobiz_stream_service import VobizRealtimeHandler


def _make_wav(rate=8000, channels=1, sampwidth=2, n_samples=800) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        w.writeframes(b"\x00" * n_samples * sampwidth * channels)
    return buf.getvalue()


def test_wav_to_ulaw8k_native_format():
    # 8kHz mono PCM16 (what Sarvam returns for speech_sample_rate=8000):
    # mulaw output is 1 byte per sample, no resampling.
    ulaw = SarvamTTSService._wav_to_ulaw8k(_make_wav(n_samples=800))
    assert len(ulaw) == 800


def test_wav_to_ulaw8k_converts_rate_and_channels():
    # Defensive path: if the API ever returns 16kHz stereo, we downmix and
    # resample instead of sending garbled audio to the call.
    ulaw = SarvamTTSService._wav_to_ulaw8k(
        _make_wav(rate=16000, channels=2, n_samples=1600)
    )
    assert len(ulaw) == 800


class FakeSarvamWS:
    """Scripted stand-in for Sarvam's streaming WebSocket."""

    def __init__(self, chunks=(b"aa", b"bb"), fail_after=None):
        self.sent = []
        self.closed = False
        self._queue = []
        self._chunks = chunks
        self._fail_after = fail_after  # raise after yielding N audio messages

    async def send(self, raw):
        msg = json.loads(raw)
        self.sent.append(msg)
        if msg.get("type") == "flush":
            for i, chunk in enumerate(self._chunks):
                if self._fail_after is not None and i >= self._fail_after:
                    self._queue.append(None)  # sentinel -> raise on recv
                    return
                self._queue.append(json.dumps({
                    "type": "audio",
                    "data": {"audio": base64.b64encode(chunk).decode()},
                }))
            self._queue.append(json.dumps({
                "type": "event", "data": {"event_type": "final"},
            }))

    async def recv(self):
        if not self._queue:
            raise AssertionError("recv called with no scripted messages")
        item = self._queue.pop(0)
        if item is None:
            raise ConnectionError("scripted mid-stream failure")
        return item

    async def close(self):
        self.closed = True


def test_ws_stream_sends_protocol_and_yields_chunks():
    async def scenario():
        service = SarvamTTSService(language="ta")
        fake = FakeSarvamWS(chunks=(b"one", b"two"))
        service._ws = fake

        chunks = [c async for c in service.synthesize_stream("வணக்கம்")]

        assert chunks == [b"one", b"two"]
        types = [m.get("type") for m in fake.sent]
        assert types == ["text", "flush"]
        assert fake.sent[0]["data"]["text"] == "வணக்கம்"
        # Completed utterance keeps the connection for the next sentence.
        assert not fake.closed and service._ws is fake

    asyncio.run(scenario())


def test_ws_stream_mid_utterance_failure_discards_connection():
    async def scenario():
        service = SarvamTTSService(language="ta")
        fake = FakeSarvamWS(chunks=(b"one", b"two"), fail_after=1)
        service._ws = fake

        received = []
        with pytest.raises(ConnectionError):
            async for chunk in service.synthesize_stream("வணக்கம்"):
                received.append(chunk)

        # First chunk was yielded, then the failure must NOT retry (audio would
        # repeat) and must drop the connection so remnants can't bleed through.
        assert received == [b"one"]
        assert fake.closed and service._ws is None

    asyncio.run(scenario())


def test_ws_stream_abandoned_by_caller_discards_connection():
    async def scenario():
        service = SarvamTTSService(language="ta")
        fake = FakeSarvamWS(chunks=(b"one", b"two"))
        service._ws = fake

        gen = service.synthesize_stream("வணக்கம்")
        assert await gen.__anext__() == b"one"
        await gen.aclose()  # barge-in: bridge stops consuming mid-utterance

        assert fake.closed and service._ws is None

    asyncio.run(scenario())


def test_synthesize_rest_posts_expected_payload_and_decodes_response():
    wav = _make_wav(n_samples=400)

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"request_id": "x", "audios": [base64.b64encode(wav).decode()]}

    class FakeClient:
        def __init__(self):
            self.calls = []
            self.is_closed = False

        async def post(self, url, json=None, headers=None):
            self.calls.append({"url": url, "json": json, "headers": headers})
            return FakeResponse()

    async def scenario():
        service = SarvamTTSService(language="ta")
        fake = FakeClient()
        service._client = fake

        ulaw = await service.synthesize_rest("வணக்கம்")

        assert len(ulaw) == 400
        call = fake.calls[0]
        assert call["url"] == SarvamTTSService.REST_URL
        assert call["headers"]["api-subscription-key"] == service.api_key
        assert call["json"]["text"] == "வணக்கம்"
        assert call["json"]["target_language_code"] == "ta-IN"
        assert call["json"]["speech_sample_rate"] == 8000
        assert call["json"]["model"]
        assert call["json"]["speaker"]

    asyncio.run(scenario())


def test_synthesize_rejects_empty_text_and_empty_audio():
    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"request_id": "x", "audios": []}

    class FakeClient:
        is_closed = False

        async def post(self, url, json=None, headers=None):
            return FakeResponse()

    async def scenario():
        service = SarvamTTSService()
        service._client = FakeClient()
        # No WS scripted: make the WS leg fail immediately so synthesize()
        # exercises the REST fallback, which returns no audio.
        service._connect_ws = _raise_connect

        with pytest.raises(ValueError):
            await service.synthesize("   ")
        with pytest.raises(ValueError):
            await service.synthesize("hello")

    asyncio.run(scenario())


async def _raise_connect():
    raise ConnectionError("no ws in tests")


def test_synthesize_uses_ws_then_falls_back_to_rest():
    wav = _make_wav(n_samples=320)

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"request_id": "x", "audios": [base64.b64encode(wav).decode()]}

    class FakeClient:
        is_closed = False

        async def post(self, url, json=None, headers=None):
            return FakeResponse()

    async def scenario():
        # WS healthy -> WS result, REST untouched.
        service = SarvamTTSService(language="ta")
        service._ws = FakeSarvamWS(chunks=(b"xy",))
        assert await service.synthesize("வணக்கம்") == b"xy"

        # WS down -> REST fallback result.
        service2 = SarvamTTSService(language="ta")
        service2._connect_ws = _raise_connect
        service2._client = FakeClient()
        assert len(await service2.synthesize("வணக்கம்")) == 320

    asyncio.run(scenario())


def test_handler_selects_sarvam_provider():
    handler = VobizRealtimeHandler(tts_provider="sarvam", language="ta")
    assert handler.tts_provider == "sarvam"
    assert handler.use_elevenlabs is True  # external-TTS (text output) mode
    assert isinstance(handler._external_tts, SarvamTTSService)
    assert handler._external_tts.target_language_code == "ta-IN"
    assert handler.get_metrics()["tts_provider"] == "sarvam"


def test_handler_provider_from_extra_headers():
    async def scenario():
        handler = VobizRealtimeHandler()

        async def no_connect():
            return None

        handler._connect_openai = no_connect
        await handler.handle_vobiz_message(object(), {
            "event": "start",
            "start": {"streamId": "s1", "callId": "c1"},
            "extra_headers": "provider=sarvam,elevenlabs=false,language=ta",
        })

        assert handler.tts_provider == "sarvam"
        assert handler.use_elevenlabs is True
        assert handler._language == "ta"
        assert isinstance(handler._external_tts, SarvamTTSService)

        await handler._full_cleanup()

    asyncio.run(scenario())


def test_bridge_streams_sarvam_chunks_and_interruption_stops_stream():
    class FakeVobiz:
        def __init__(self):
            self.payloads = []

        async def send_json(self, message):
            if message.get("event") == "playAudio":
                self.payloads.append(
                    base64.b64decode(message["media"]["payload"])
                )

    class GatedStreamTTS:
        """Yields one chunk, then waits until released before the second."""

        def __init__(self):
            self.first_sent = asyncio.Event()
            self.release = asyncio.Event()
            self.aborted = False

        async def synthesize_stream(self, text):
            yield b"\xff" * 100
            self.first_sent.set()
            try:
                await self.release.wait()
                yield b"\xff" * 100
            except GeneratorExit:
                self.aborted = True
                raise

        async def close(self):
            return None

    async def scenario():
        handler = VobizRealtimeHandler(tts_provider="sarvam", language="ta")
        vobiz = FakeVobiz()
        tts = GatedStreamTTS()
        handler._external_tts = tts
        handler._vobiz_ws = vobiz
        handler.stream_id = "stream-1"

        task = asyncio.create_task(
            handler._synthesize_and_send("வணக்கம், சோதனை.", handler._tts_generation)
        )
        await tts.first_sent.wait()
        handler._drain_tts_queue()  # barge-in invalidates the generation
        tts.release.set()
        await task

        # First chunk reached Vobiz; the post-interrupt chunk did not.
        assert len(vobiz.payloads) == 1

    asyncio.run(scenario())


def test_handler_legacy_elevenlabs_flag_still_works():
    from services.elevenlabs_tts_service import ElevenLabsTTSService

    handler = VobizRealtimeHandler(use_elevenlabs=True)
    assert handler.tts_provider == "elevenlabs"
    assert isinstance(handler._external_tts, ElevenLabsTTSService)

    handler = VobizRealtimeHandler()
    assert handler.tts_provider == "openai"
    assert handler.use_elevenlabs is False
    assert handler._external_tts is None
