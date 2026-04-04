import asyncio
import json
import requests
import websockets
import os

BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"

def run_tests():
    print("Testing Backend APIs...")
    session = requests.Session()

    print("\n1. Testing Health Check")
    r = session.get(f"{BASE_URL}/api/health")
    assert r.status_code == 200, f"Health check failed: {r.text}"
    print("Health Check: OK")

    print("\n2. Testing Login")
    r = session.post(f"{BASE_URL}/api/login", json={"username": "admin", "password": "admin123"})
    if r.status_code == 200:
        token = r.json().get("token")
        session.headers.update({"Authorization": f"Bearer {token}"})
        print("Login: OK")
    else:
        print(f"Login failed: {r.text}")
        return

    print("\n3. Testing Settings Fetch")
    r = session.get(f"{BASE_URL}/api/settings")
    if r.status_code == 200:
        print("Settings Fetch: OK")
    else:
        print(f"Settings Fetch failed: {r.text}")

    print("\n4. Testing Script Creation")
    script_data = {
        "name": "Test Script",
        "welcome": "Hello there",
        "questions": [{"id": "1", "question": "Are you there?"}],
        "goal": "Just test",
        "behaviour": "Friendly"
    }
    r = session.post(f"{BASE_URL}/api/scripts", json=script_data)
    if r.status_code == 200:
        print("Script Create: OK")
        sid = r.json()["script"]["id"]
    else:
        print(f"Script Create failed: {r.text}")
        return

    print("\n5. Testing Script Activation")
    r = session.post(f"{BASE_URL}/api/script/activate", json=script_data)
    if r.status_code == 200:
        print("Script Activate: OK")
    else:
        print(f"Script Activate failed: {r.text}")

    print("\nAll HTTP tests passed.")

async def test_websocket():
    try:
        async with websockets.connect(f"{WS_URL}/ws/realtime") as ws:
            print("\n6. Testing WebSocket Realtime connection: OK")
            await ws.send(json.dumps({"type": "end_conversation"}))
    except Exception as e:
        print(f"\nWebSocket connection failed: {e}")

if __name__ == "__main__":
    run_tests()
    asyncio.run(test_websocket())
