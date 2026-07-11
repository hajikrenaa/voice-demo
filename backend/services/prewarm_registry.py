"""Concurrency-safe lifecycle management for pre-warmed Realtime sockets."""

import asyncio
import logging
import secrets
import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class PrewarmEntry:
    task: asyncio.Task
    use_elevenlabs: bool
    language: str
    active_script: Optional[dict]
    expiry_task: Optional[asyncio.Task] = None
    # Per-call TTS provider ("openai" | "elevenlabs" | "sarvam"). use_elevenlabs is
    # the legacy external-TTS flag kept for compatibility.
    tts_provider: str = "openai"
    # Number dialed for this outbound call ("" for inbound/test calls) — threaded
    # into the prompt so "send it to this number" resolves without guessing.
    called_number: str = ""


class PrewarmRegistry:
    """Owns one expiring pre-warm task per call.

    Entries are claimed exactly once. Unclaimed entries are cancelled after the TTL,
    and a successfully opened WebSocket is closed so unanswered calls do not leak an
    active Realtime session.
    """

    def __init__(self, ttl_seconds: float = 75.0):
        self.ttl_seconds = ttl_seconds
        self._entries: dict[str, PrewarmEntry] = {}

    def register(
        self,
        factory: Callable[[], Awaitable[Any]],
        *,
        use_elevenlabs: bool,
        language: str,
        active_script: Optional[dict],
        tts_provider: str = None,
        called_number: str = "",
    ) -> str:
        token = secrets.token_urlsafe(18)
        task = asyncio.create_task(factory())
        entry = PrewarmEntry(
            task=task,
            use_elevenlabs=use_elevenlabs,
            language=language,
            active_script=active_script,
            tts_provider=tts_provider
            or ("elevenlabs" if use_elevenlabs else "openai"),
            called_number=called_number,
        )
        self._entries[token] = entry
        entry.expiry_task = asyncio.create_task(self._expire(token))
        return token

    def contains(self, token: str | None) -> bool:
        return bool(token and token in self._entries)

    def claim(self, token: str | None) -> Optional[PrewarmEntry]:
        if not token:
            return None
        entry = self._entries.pop(token, None)
        if entry and entry.expiry_task and not entry.expiry_task.done():
            entry.expiry_task.cancel()
            entry.expiry_task = None
        return entry

    async def discard(self, token: str | None) -> None:
        if not token:
            return
        entry = self._entries.pop(token, None)
        if not entry:
            return
        current = asyncio.current_task()
        if (
            entry.expiry_task
            and entry.expiry_task is not current
            and not entry.expiry_task.done()
        ):
            entry.expiry_task.cancel()
        await self.close_task(entry.task)

    async def close_all(self) -> None:
        tokens = list(self._entries)
        for token in tokens:
            await self.discard(token)

    async def _expire(self, token: str) -> None:
        try:
            await asyncio.sleep(self.ttl_seconds)
            logger.info("Expiring unclaimed Realtime pre-warm: %s", token[:8])
            await self.discard(token)
        except asyncio.CancelledError:
            return

    @staticmethod
    async def close_task(task: asyncio.Task) -> None:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                return
            return

        try:
            resource = task.result()
        except (asyncio.CancelledError, Exception):
            return

        close = getattr(resource, "close", None)
        if close:
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                logger.debug("Failed to close expired pre-warm resource: %s", exc)