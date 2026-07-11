import asyncio

from services.prewarm_registry import PrewarmRegistry


class FakeSocket:
    def __init__(self):
        self.closed = 0

    async def close(self):
        self.closed += 1


def test_entries_are_isolated_and_claimed_once():
    async def scenario():
        registry = PrewarmRegistry(ttl_seconds=10)
        socket_en = FakeSocket()
        socket_ta = FakeSocket()

        async def make_en():
            return socket_en

        async def make_ta():
            return socket_ta

        token_en = registry.register(
            make_en,
            use_elevenlabs=False,
            language="en",
            active_script={"name": "English"},
        )
        token_ta = registry.register(
            make_ta,
            use_elevenlabs=True,
            language="ta",
            active_script={"name": "Tamil"},
        )

        assert token_en != token_ta
        entry_ta = registry.claim(token_ta)
        entry_en = registry.claim(token_en)
        assert entry_ta.language == "ta" and entry_ta.use_elevenlabs is True
        assert entry_en.language == "en" and entry_en.use_elevenlabs is False
        assert registry.claim(token_en) is None

        await asyncio.gather(entry_en.task, entry_ta.task)
        await PrewarmRegistry.close_task(entry_en.task)
        await PrewarmRegistry.close_task(entry_ta.task)
        assert socket_en.closed == 1
        assert socket_ta.closed == 1

    asyncio.run(scenario())


def test_unclaimed_entry_expires_and_closes_socket():
    async def scenario():
        registry = PrewarmRegistry(ttl_seconds=0.01)
        socket = FakeSocket()

        async def factory():
            return socket

        token = registry.register(
            factory,
            use_elevenlabs=False,
            language="en",
            active_script=None,
        )
        await asyncio.sleep(0.05)

        assert not registry.contains(token)
        assert socket.closed == 1

    asyncio.run(scenario())


def test_discard_cancels_pending_factory():
    async def scenario():
        registry = PrewarmRegistry(ttl_seconds=10)
        cancelled = asyncio.Event()

        async def factory():
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        token = registry.register(
            factory,
            use_elevenlabs=False,
            language="en",
            active_script=None,
        )
        await asyncio.sleep(0)
        await registry.discard(token)

        assert cancelled.is_set()
        assert not registry.contains(token)

    asyncio.run(scenario())