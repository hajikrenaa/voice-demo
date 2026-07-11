"""Manual zero-audio smoke test for the configured OpenAI Realtime session."""

import asyncio
import json
import time

import websockets

from config import Config


async def run_realtime_check():
    url = f"wss://api.openai.com/v1/realtime?model={Config.REALTIME_MODEL}"
    headers = {"Authorization": f"Bearer {Config.OPENAI_API_KEY}"}

    async with websockets.connect(
        url,
        additional_headers=headers,
        ping_interval=30,
        ping_timeout=10,
        compression=None,
    ) as ws:
        config = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "output_modalities": ["audio"],
                "instructions": "Reply briefly and naturally.",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcmu"},
                        "transcription": {
                            "model": Config.TRANSCRIPTION_MODEL,
                            "language": "en",
                        },
                        "turn_detection": {
                            "type": "semantic_vad",
                            "eagerness": Config.SEMANTIC_VAD_EAGERNESS,
                            "create_response": True,
                            "interrupt_response": False,
                        },
                    },
                    "output": {
                        "format": {"type": "audio/pcmu"},
                        "voice": Config.REALTIME_VOICE,
                    },
                },
                "max_output_tokens": 50,
            },
        }
        await ws.send(json.dumps(config))

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            event = json.loads(
                await asyncio.wait_for(ws.recv(), timeout=deadline - time.monotonic())
            )
            event_type = event.get("type")
            if event_type == "session.updated":
                print(
                    "Realtime session smoke test passed:",
                    Config.REALTIME_MODEL,
                    Config.TRANSCRIPTION_MODEL,
                )
                return
            if event_type == "error":
                raise RuntimeError(event.get("error", {}).get("message", "Realtime error"))

        raise TimeoutError("Timed out waiting for session.updated")


if __name__ == "__main__":
    asyncio.run(run_realtime_check())