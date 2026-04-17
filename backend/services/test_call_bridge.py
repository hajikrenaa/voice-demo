"""
In-browser Test Call bridge — zero Vobiz credits used.

Reuses VobizRealtimeHandler unchanged so the test call exercises the
EXACT same prompt, VAD, echo gate, interruption, script, ElevenLabs and
goodbye logic as a real Vobiz call. A FakeVobizSocket adapter converts
between browser PCM16/24k audio and the mulaw/8k format the handler
expects, so the handler sees input that matches a real phone call.
"""

import asyncio
import audioop
import base64
import json
import logging
import uuid
from typing import Optional

from fastapi import WebSocket

from services.vobiz_stream_service import VobizRealtimeHandler
from utils.audio_processing import downsample_24k_to_8k, upsample_8k_to_24k

logger = logging.getLogger(__name__)


class FakeVobizSocket:
    """
    Drop-in replacement for the Vobiz-side WebSocket.

    VobizRealtimeHandler only ever calls `.send_json({"event": ...})` and
    `.close()` on its _vobiz_ws. We translate those into browser messages
    and never touch the real Vobiz platform.
    """

    def __init__(self, browser_ws: WebSocket):
        self._browser_ws = browser_ws
        self._closed = False

    async def send_json(self, obj: dict):
        if self._closed:
            return
        event = obj.get("event")
        if event == "playAudio":
            mulaw_b64 = obj.get("media", {}).get("payload", "")
            if not mulaw_b64:
                return
            try:
                mulaw = base64.b64decode(mulaw_b64)
                pcm16_8k = audioop.ulaw2lin(mulaw, 2)
                pcm16_24k = upsample_8k_to_24k(pcm16_8k)
                pcm16_b64 = base64.b64encode(pcm16_24k).decode("utf-8")
                await self._browser_ws.send_json({
                    "type": "response_audio",
                    "data": pcm16_b64,
                })
            except Exception as e:
                logger.error(f"FakeVobizSocket playAudio relay failed: {e}")
        elif event == "clearAudio":
            try:
                await self._browser_ws.send_json({"type": "clear_audio"})
            except Exception as e:
                logger.debug(f"FakeVobizSocket clearAudio relay failed: {e}")
        else:
            logger.debug(f"FakeVobizSocket ignoring unknown event: {event}")

    async def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            await self._browser_ws.send_json({"type": "ended"})
        except Exception:
            pass


async def run_test_call(
    browser_ws: WebSocket,
    active_script: Optional[dict],
    prewarm_task: Optional[asyncio.Task] = None,
    prewarm_use_elevenlabs: bool = False,
):
    """
    Bridge a browser WebSocket into a real VobizRealtimeHandler session.

    Browser protocol:
      Client -> server:
        {"type": "start", "elevenlabs": bool}
        {"type": "audio_chunk", "data": "<pcm16-24k-b64>"}
        {"type": "stop"}
      Server -> client:
        {"type": "response_audio", "data": "<pcm16-24k-b64>"}
        {"type": "clear_audio"}
        {"type": "transcript", "role": "ai"|"user", "text": str, "final": bool}
        {"type": "ended"}
    """
    fake_socket = FakeVobizSocket(browser_ws)
    handler: Optional[VobizRealtimeHandler] = None
    prewarm_consumed = False

    async def transcript_hook(role: str, text: str, final: bool):
        try:
            await browser_ws.send_json({
                "type": "transcript",
                "role": role,
                "text": text,
                "final": final,
            })
        except Exception:
            pass

    try:
        while True:
            try:
                raw = await browser_ws.receive_text()
            except Exception:
                break

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            mtype = msg.get("type")

            if mtype == "start":
                use_elevenlabs = bool(msg.get("elevenlabs", False))
                handler = VobizRealtimeHandler(
                    use_elevenlabs=use_elevenlabs,
                    active_script=active_script,
                )
                handler._transcript_callback = transcript_hook

                if (
                    prewarm_task
                    and not prewarm_consumed
                    and prewarm_use_elevenlabs == use_elevenlabs
                ):
                    handler._prewarm_task = prewarm_task
                    handler._session_preconfigured = True
                    prewarm_consumed = True

                stream_id = f"test-{uuid.uuid4().hex[:8]}"
                call_id = f"test-{uuid.uuid4().hex[:8]}"
                el_hdr = "true" if use_elevenlabs else "false"

                await handler.handle_vobiz_message(fake_socket, {
                    "event": "start",
                    "start": {
                        "streamId": stream_id,
                        "callId": call_id,
                    },
                    "extra_headers": f"elevenlabs={el_hdr}",
                })
                logger.info(
                    f"Test call started — streamId={stream_id}, "
                    f"elevenlabs={use_elevenlabs}, "
                    f"script={'yes' if active_script else 'no'}"
                )

            elif mtype == "audio_chunk":
                if not handler:
                    continue
                pcm16_b64 = msg.get("data", "")
                if not pcm16_b64:
                    continue
                try:
                    pcm16_24k = base64.b64decode(pcm16_b64)
                    pcm16_8k = downsample_24k_to_8k(pcm16_24k)
                    if not pcm16_8k:
                        continue
                    mulaw = audioop.lin2ulaw(pcm16_8k, 2)
                    mulaw_b64 = base64.b64encode(mulaw).decode("utf-8")
                    await handler.handle_vobiz_message(fake_socket, {
                        "event": "media",
                        "media": {"payload": mulaw_b64},
                    })
                except Exception as e:
                    logger.error(f"Test call audio relay failed: {e}")

            elif mtype == "stop":
                if handler:
                    await handler.handle_vobiz_message(fake_socket, {
                        "event": "stop",
                        "reason": "client_stop",
                    })
                break

    finally:
        if handler:
            try:
                await handler._full_cleanup()
            except Exception as e:
                logger.error(f"Test call cleanup failed: {e}")
        logger.info("Test call session ended")
