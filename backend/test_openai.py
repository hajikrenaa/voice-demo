import asyncio
import os
import json
import websockets
from dotenv import load_dotenv

load_dotenv()

async def test():
    API_KEY = os.getenv("OPENAI_API_KEY")
    url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "OpenAI-Beta": "realtime=v1"
    }
    
    async with websockets.connect(url, additional_headers=headers) as ws:
        print("Connected!")
        # receive initial events
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                print("Initial:", json.loads(msg)["type"])
            except asyncio.TimeoutError:
                break

        print("Sending session.update with semantic_vad")
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "voice": "coral",
                "input_audio_format": "g711_ulaw",
                "output_audio_format": "g711_ulaw",
                "input_audio_transcription": {
                    "model": "whisper-1",
                    "language": "en",
                },
                "turn_detection": {
                    "type": "semantic_vad",
                    "eagerness": "medium",
                },
            }
        }))
        
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
                event = json.loads(msg)
                print("Received:", event["type"])
                if event["type"] == "session.updated":
                    print("Session updated formats:")
                    print("  input:", event.get("session", {}).get("input_audio_format"))
                    print("  output:", event.get("session", {}).get("output_audio_format"))
                    break
                elif event["type"] == "error":
                    print("ERROR:", json.dumps(event))
                    break
            except asyncio.TimeoutError:
                print("Timed out!")
                break

        print("---\nSending session.update with server_vad")
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "voice": "coral",
                "input_audio_format": "g711_ulaw",
                "output_audio_format": "g711_ulaw",
                "input_audio_transcription": {
                    "model": "whisper-1",
                    "language": "en",
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                },
            }
        }))
        
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
                event = json.loads(msg)
                print("Received:", event["type"])
                if event["type"] == "session.updated":
                    print("Session updated formats:")
                    print("  input:", event.get("session", {}).get("input_audio_format"))
                    print("  output:", event.get("session", {}).get("output_audio_format"))
                    break
                elif event["type"] == "error":
                    print("ERROR:", json.dumps(event))
                    break
            except asyncio.TimeoutError:
                print("Timed out!")
                break

asyncio.run(test())
