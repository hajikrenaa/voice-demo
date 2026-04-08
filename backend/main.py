from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, Response
import os
import uvicorn
import logging
import sys
import asyncio
import json
import uuid
import time
import secrets
import websockets
from pathlib import Path
from html import escape as html_escape

# Import services
from services.whisper_service import WhisperService
from services.llm_service import LLMService
from services.tts_service import TTSService
from services.conversation_manager import ConversationManager
from services.realtime_service import RealtimeService, RealtimeEventHandler
from services.twilio_stream_service import TwilioRealtimeHandler
from utils.audio_processing import (
    decode_base64_audio,
    encode_audio_to_base64,
    convert_to_wav,
)
from config import Config

# Active sessions — maps token -> expiry timestamp (24h TTL)
_SESSION_TTL = 86400  # 24 hours
_MAX_SESSIONS = 100
_active_sessions: dict[str, float] = {}

# Scripts storage files
SCRIPTS_FILE = Path(__file__).parent / "data" / "scripts.json"
ACTIVE_SCRIPT_FILE = Path(__file__).parent / "data" / "active_script.json"


def _load_active_script() -> dict | None:
    """Load active script from disk (survives server restarts/reloads)."""
    try:
        if ACTIVE_SCRIPT_FILE.exists():
            data = json.loads(ACTIVE_SCRIPT_FILE.read_text(encoding="utf-8"))
            return data if data else None
    except Exception:
        pass
    return None


def _save_active_script(script: dict | None):
    """Persist active script to disk."""
    ACTIVE_SCRIPT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_SCRIPT_FILE.write_text(json.dumps(script), encoding="utf-8")


# Active script — loaded from disk so it survives hot-reloads
_active_script: dict | None = _load_active_script()
if _active_script:
    print(f"[STARTUP] Active script loaded from disk: {_active_script.get('name', 'unnamed')}")
else:
    print("[STARTUP] No active script — calls will use default prompt")


def _load_scripts() -> list:
    """Load saved scripts from disk."""
    try:
        if SCRIPTS_FILE.exists():
            return json.loads(SCRIPTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def _save_scripts(scripts: list):
    """Save scripts to disk."""
    SCRIPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCRIPTS_FILE.write_text(json.dumps(scripts, indent=2), encoding="utf-8")

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Voice Agent",
    description="Professional AI voice agent with OpenAI integration",
    version="1.0.0"
)

# CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
    logger.info(f"Mounted frontend from: {frontend_path}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML"""
    try:
        index_path = frontend_path / "index.html"
        if index_path.exists():
            return HTMLResponse(content=index_path.read_text(encoding='utf-8'))
        else:
            return HTMLResponse(
                content="""
                <html>
                    <head><title>AI Voice Agent</title></head>
                    <body>
                        <h1>AI Voice Agent Backend is Running</h1>
                        <p>Frontend not found. Please create frontend/index.html</p>
                        <p>WebSocket endpoint: ws://localhost:8000/ws/voice</p>
                    </body>
                </html>
                """
            )
    except Exception as e:
        logger.error(f"Error serving root: {e}")
        return HTMLResponse(content=f"<h1>Error: {html_escape(str(e))}</h1>")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "services": {
            "whisper": "configured",
            "gpt4o": "configured",
            "tts": "configured",
            "twilio": "configured" if Config.TWILIO_ACCOUNT_SID else "not configured",
            "elevenlabs": "configured" if Config.ELEVENLABS_API_KEY else "not configured",
        },
        "environment": Config.ENVIRONMENT
    })


@app.get("/api/config")
async def get_config():
    """Get client configuration"""
    return JSONResponse({
        "available_voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        "current_voice": Config.TTS_VOICE,
        "max_conversation_duration": Config.MAX_CONVERSATION_DURATION,
        "supported_audio_formats": ["webm", "wav", "mp3"],
        "sample_rate": Config.SAMPLE_RATE,
        "realtime_available": True
    })


# ==========================================
# Authentication
# ==========================================

def _verify_token(token: str | None) -> bool:
    """Check if a session token is valid and not expired."""
    if not token or token not in _active_sessions:
        return False
    if time.time() > _active_sessions[token]:
        _active_sessions.pop(token, None)
        return False
    return True


def _require_auth(request: Request) -> bool:
    """Check Authorization header. Returns True if valid, False otherwise."""
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    return _verify_token(token)


def _cleanup_sessions():
    """Evict expired sessions."""
    now = time.time()
    expired = [t for t, exp in _active_sessions.items() if now > exp]
    for t in expired:
        _active_sessions.pop(t, None)


@app.post("/api/login")
async def login(request: Request):
    """Authenticate the single user and return a session token."""
    body = await request.json()
    username = body.get("username", "")
    password = body.get("password", "")

    user_ok = secrets.compare_digest(username, Config.LOGIN_USERNAME)
    pass_ok = secrets.compare_digest(password, Config.LOGIN_PASSWORD)
    if user_ok and pass_ok:
        _cleanup_sessions()
        # Cap max sessions to prevent memory growth
        if len(_active_sessions) >= _MAX_SESSIONS:
            oldest = min(_active_sessions, key=_active_sessions.get)
            _active_sessions.pop(oldest, None)
        token = secrets.token_hex(32)
        _active_sessions[token] = time.time() + _SESSION_TTL
        logger.info(f"User '{username}' logged in")
        return JSONResponse({"success": True, "token": token})

    logger.warning(f"Failed login attempt for user '{username}'")
    return JSONResponse({"success": False, "error": "Invalid username or password"}, status_code=401)


@app.get("/api/auth/check")
async def auth_check(request: Request):
    """Check if the current session token is still valid."""
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if _verify_token(token):
        return JSONResponse({"authenticated": True})
    return JSONResponse({"authenticated": False}, status_code=401)


@app.post("/api/logout")
async def logout(request: Request):
    """Invalidate the session token."""
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    _active_sessions.pop(token, None)
    return JSONResponse({"success": True})


# ==========================================
# Settings / API Keys Management
# ==========================================

ENV_FILE = Path(__file__).parent / ".env"

# Keys that can be read and updated from the UI
_EDITABLE_KEYS = [
    "OPENAI_API_KEY",
    "ELEVENLABS_API_KEY",
    "ELEVENLABS_VOICE_ID",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "TWILIO_PHONE_NUMBER",
    "SERVER_URL",
    "LOGIN_USERNAME",
    "LOGIN_PASSWORD",
]


def _read_env() -> dict:
    """Parse the .env file into a dict."""
    result = {}
    if not ENV_FILE.exists():
        return result
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    return result


def _write_env(values: dict):
    """Write updated values back to .env, preserving comments and order."""
    if not ENV_FILE.exists():
        # Create with all values
        lines = []
        for k, v in values.items():
            lines.append(f"{k}={v}")
        ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    original_lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    updated_keys = set()
    new_lines = []

    for line in original_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in values:
                new_lines.append(f"{key}={values[key]}")
                updated_keys.add(key)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    # Append any new keys not already in the file
    for k, v in values.items():
        if k not in updated_keys:
            new_lines.append(f"{k}={v}")

    ENV_FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _mask_key(value: str) -> str:
    """Mask a secret key for display — show first 6 and last 4 chars."""
    if not value or len(value) <= 12:
        return value
    return value[:6] + "*" * (len(value) - 10) + value[-4:]


@app.get("/api/settings")
async def get_settings(request: Request):
    """Return current settings with masked secrets."""
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not _verify_token(token):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    env = _read_env()
    # Build response — mask secret keys, show others in full
    secret_keys = {"OPENAI_API_KEY", "ELEVENLABS_API_KEY", "TWILIO_AUTH_TOKEN", "LOGIN_PASSWORD"}
    settings = {}
    for key in _EDITABLE_KEYS:
        raw = env.get(key, "")
        settings[key] = {
            "value": _mask_key(raw) if key in secret_keys else raw,
            "masked": key in secret_keys,
        }

    return JSONResponse({"success": True, "settings": settings})


@app.put("/api/settings")
async def update_settings(request: Request):
    """Update one or more settings in the .env file and reload Config."""
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not _verify_token(token):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    body = await request.json()
    updates = body.get("settings", {})

    if not updates:
        return JSONResponse({"error": "No settings provided"}, status_code=400)

    # Only allow editable keys
    filtered = {}
    for key, value in updates.items():
        if key in _EDITABLE_KEYS and isinstance(value, str) and value.strip():
            filtered[key] = value.strip()

    if not filtered:
        return JSONResponse({"error": "No valid settings to update"}, status_code=400)

    # Read current env, merge, write back
    env = _read_env()
    env.update(filtered)
    _write_env(env)

    # Reload Config class attributes from updated values
    for key, value in filtered.items():
        if hasattr(Config, key):
            setattr(Config, key, value)
        os.environ[key] = value

    logger.info(f"Settings updated: {list(filtered.keys())}")
    return JSONResponse({"success": True, "updated": list(filtered.keys())})


# ==========================================
# Script Management Endpoints
# ==========================================

@app.post("/api/script/activate")
async def activate_script(request: Request):
    """Activate a script so all outbound calls use it."""
    if not _require_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    global _active_script
    body = await request.json()
    _active_script = body
    _save_active_script(body)
    logger.info(f"Script activated: welcome='{body.get('welcome', '')[:40]}', {len(body.get('questions', []))} questions")
    return JSONResponse({"success": True, "active": True})


@app.post("/api/script/deactivate")
async def deactivate_script(request: Request):
    """Deactivate the current script — calls use default prompt."""
    if not _require_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    global _active_script
    _active_script = None
    _save_active_script(None)
    logger.info("Script deactivated")
    return JSONResponse({"success": True, "active": False})


@app.get("/api/script/status")
async def script_status(request: Request):
    """Check if a script is currently active."""
    if not _require_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return JSONResponse({
        "active": _active_script is not None,
        "script": _active_script
    })


# ==========================================
# Saved Scripts CRUD
# ==========================================

@app.get("/api/scripts")
async def list_scripts(request: Request):
    """List all saved scripts."""
    if not _require_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return JSONResponse(_load_scripts())


@app.post("/api/scripts")
async def save_script(request: Request):
    """Save a new script (or update existing by id)."""
    if not _require_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    body = await request.json()
    scripts = _load_scripts()

    script_id = body.get("id") or str(uuid.uuid4())[:8]
    name = body.get("name") or body.get("welcome", "Untitled")[:40]

    # Check if updating existing
    existing = next((i for i, s in enumerate(scripts) if s["id"] == script_id), None)

    script_obj = {
        "id": script_id,
        "name": name,
        "welcome": body.get("welcome", ""),
        "questions": body.get("questions", []),
        "goal": body.get("goal", ""),
        "behaviour": body.get("behaviour", ""),
    }

    if existing is not None:
        scripts[existing] = script_obj
    else:
        scripts.append(script_obj)

    _save_scripts(scripts)
    logger.info(f"Script saved: {script_id} — {name}")
    return JSONResponse({"success": True, "script": script_obj})


@app.delete("/api/scripts/{script_id}")
async def delete_script(script_id: str, request: Request):
    """Delete a saved script."""
    if not _require_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    global _active_script
    scripts = _load_scripts()
    scripts = [s for s in scripts if s["id"] != script_id]
    _save_scripts(scripts)

    # Deactivate if this was the active script
    if _active_script and _active_script.get("id") == script_id:
        _active_script = None
        _save_active_script(None)

    logger.info(f"Script deleted: {script_id}")
    return JSONResponse({"success": True})


# ==========================================
# Twilio Voice Calling Endpoints
# ==========================================

# Pre-warmed OpenAI connection — started at TwiML time, consumed by the
# WebSocket handler ~0.6-1s later so the handshake is already done.
_prewarm_openai_task: asyncio.Task | None = None


async def _prewarm_openai_connection(active_script: dict | None = None, use_elevenlabs: bool = False):
    """Open an OpenAI Realtime WebSocket and pre-send session config.

    By sending session.update during the pre-warm (while Twilio is still
    setting up the media stream), we save ~500ms on the first response.
    """
    model = Config.REALTIME_MODEL
    url = f"{TwilioRealtimeHandler.OPENAI_REALTIME_URL}?model={model}"
    headers = {
        "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }
    logger.info("Pre-warming OpenAI Realtime connection...")
    ws = await asyncio.wait_for(
        websockets.connect(
            url,
            additional_headers=headers,
            ping_interval=30,
            ping_timeout=10,
            max_size=2**20,
            compression=None,
        ),
        timeout=10.0,
    )
    logger.info("Pre-warmed OpenAI connection ready")

    # Pre-send session config for the default (non-ElevenLabs) case
    if not use_elevenlabs:
        try:
            temp_handler = TwilioRealtimeHandler(active_script=active_script)
            prompt = temp_handler._build_prompt()
            temperature = 0.7 if active_script else 0.8
            max_tokens = 400 if active_script else 150

            # Build VAD config — try semantic_vad first (matches _configure_session logic)
            vad_config = {"type": "semantic_vad", "eagerness": Config.SEMANTIC_VAD_EAGERNESS}

            session_config = {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "instructions": prompt,
                    "voice": Config.REALTIME_VOICE,
                    "input_audio_format": "g711_ulaw",
                    "output_audio_format": "g711_ulaw",
                    "temperature": temperature,
                    "max_response_output_tokens": max_tokens,
                    "input_audio_transcription": {"model": "whisper-1", "language": "en"},
                    "turn_detection": vad_config,
                    "input_audio_noise_reduction": {"type": "near_field"},
                },
            }
            await ws.send(json.dumps(session_config))
            # Don't pre-trigger response — let the caller speak first (say hello)
            # The model will respond after hearing the caller's voice via VAD
            logger.info("Pre-sent session config during pre-warm (waiting for caller to speak first)")
        except Exception as e:
            logger.warning(f"Failed to pre-send session config: {e} (will cold-connect)")
            # WebSocket is likely dead after the error — discard it
            try:
                await ws.close()
            except Exception:
                pass
            return None

    return ws

@app.post("/twilio/voice")
async def twilio_voice_webhook(request: Request):
    """
    Twilio voice webhook - called when an inbound/outbound call connects.
    Returns TwiML that tells Twilio to open a bidirectional media stream.
    """
    server_url = Config.SERVER_URL
    ws_url = server_url.replace("https://", "wss://").replace("http://", "ws://")

    # Sanitize to prevent TwiML injection — only allow "true" or "false"
    use_elevenlabs = "true" if request.query_params.get("elevenlabs", "false").lower() == "true" else "false"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}/ws/twilio-stream">
            <Parameter name="caller" value="{{{{From}}}}" />
            <Parameter name="elevenlabs" value="{use_elevenlabs}" />
        </Stream>
    </Connect>
    <Say voice="alice">Sorry, the voice agent disconnected. Please try calling again.</Say>
</Response>"""

    # Start OpenAI handshake + session config now — by the time Twilio opens
    # the WebSocket (~0.6-1s later), the session will already be configured.
    global _prewarm_openai_task
    elevenlabs_flag = use_elevenlabs == "true"
    _prewarm_openai_task = asyncio.create_task(
        _prewarm_openai_connection(_active_script, elevenlabs_flag)
    )

    logger.info(f"Twilio voice webhook called, script_active={_active_script is not None}")
    logger.info(f"Generated TwiML: {twiml}")
    return Response(content=twiml, media_type="application/xml")


@app.post("/twilio/outbound-call")
async def make_outbound_call(request: Request):
    """
    API endpoint to initiate an outbound call via Twilio.

    Request body: { "to": "+1234567890" }
    """
    if not _require_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        from twilio.rest import Client as TwilioClient

        body = await request.json()
        to_number = body.get("to")
        use_elevenlabs = body.get("elevenlabs", False)

        if not to_number:
            return JSONResponse(
                {"error": "Missing 'to' phone number"}, status_code=400
            )

        client = TwilioClient(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)

        el = 'true' if use_elevenlabs else 'false'
        voice_url = f"{Config.SERVER_URL}/twilio/voice?elevenlabs={el}"
        status_cb = f"{Config.SERVER_URL}/twilio/call-status"
        call = client.calls.create(
            to=to_number,
            from_=Config.TWILIO_PHONE_NUMBER,
            url=voice_url,
            status_callback=status_cb,
            status_callback_event=["initiated", "ringing", "answered", "completed"],
        )

        logger.info(f"Outbound call initiated: {call.sid} to {to_number}")

        return JSONResponse({
            "success": True,
            "call_sid": call.sid,
            "to": to_number,
            "from": Config.TWILIO_PHONE_NUMBER,
            "status": call.status,
        })

    except Exception as e:
        logger.error(f"Failed to make outbound call: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/twilio/hangup")
async def hangup_call(request: Request):
    """Hang up an active Twilio call."""
    if not _require_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        from twilio.rest import Client as TwilioClient

        body = await request.json()
        call_sid = body.get("call_sid")

        if not call_sid:
            return JSONResponse({"error": "Missing call_sid"}, status_code=400)

        client = TwilioClient(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)
        call = client.calls(call_sid).update(status="completed")

        logger.info(f"Call {call_sid} hung up")
        return JSONResponse({"success": True, "call_sid": call_sid, "status": call.status})

    except Exception as e:
        logger.error(f"Hangup failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.websocket("/ws/twilio-stream")
async def twilio_media_stream(websocket: WebSocket):
    """
    WebSocket endpoint for Twilio bidirectional media streams.
    Uses OpenAI Realtime API for ~300-500ms latency.

    Audio flows:
      Twilio (mulaw 8kHz) → PCM16 24kHz → OpenAI Realtime API
      OpenAI Realtime API → PCM16 24kHz → mulaw 8kHz → Twilio
    """
    logger.info(f"Incoming WebSocket connection to /ws/twilio-stream. Headers: {dict(websocket.headers)}")
    await websocket.accept()
    logger.info("Twilio media stream WebSocket connected and accepted")

    if _active_script:
        logger.info(f"Creating handler WITH script: {_active_script.get('name', 'unnamed')} ({len(_active_script.get('questions', []))} questions)")
    else:
        logger.info("Creating handler WITHOUT script — using default prompt")
    handler = TwilioRealtimeHandler(active_script=_active_script)

    # Pass pre-warmed OpenAI connection (with session config already sent) if available
    global _prewarm_openai_task
    if _prewarm_openai_task:
        handler._prewarm_task = _prewarm_openai_task
        handler._session_preconfigured = True
        _prewarm_openai_task = None

    try:
        async for raw_message in websocket.iter_text():
            try:
                message = json.loads(raw_message)
                event = message.get("event")
                if event != "media":
                    logger.info(f"Twilio stream event: {event}")
                await handler.handle_twilio_message(websocket, message)
            except json.JSONDecodeError:
                logger.error("Failed to parse Twilio stream message")
            except Exception as e:
                logger.error(f"Error handling Twilio stream message: {e}")

    except WebSocketDisconnect:
        logger.info("Twilio media stream disconnected")
    except Exception as e:
        logger.error(f"Twilio stream WebSocket error: {e}")
    finally:
        await handler._full_cleanup()
        logger.info(f"Twilio stream closed - CallSID: {handler.call_sid}")

@app.get("/ws/twilio-stream")
async def twilio_stream_fallback_get(request: Request):
    """Temporary endpoint to log what headers we're receiving if WebSocket upgrade fails."""
    logger.error(f"Received HTTP GET instead of WebSocket on /ws/twilio-stream. Headers: {dict(request.headers)}")
    return {"error": "Expected WebSocket connection"}


@app.post("/twilio/call-status")
async def twilio_call_status(request: Request):
    """Twilio status callback — logs call lifecycle events for debugging."""
    form = await request.form()
    call_sid = form.get("CallSid", "?")
    status = form.get("CallStatus", "?")
    duration = form.get("CallDuration", "")
    error_code = form.get("ErrorCode", "")
    error_msg = form.get("ErrorMessage", "")

    if error_code:
        logger.error(f"Call {call_sid}: {status} — ERROR {error_code}: {error_msg}")
    else:
        extra = f" ({duration}s)" if duration else ""
        logger.info(f"Call {call_sid}: {status}{extra}")

    return Response(status_code=204)


@app.get("/twilio/status")
async def twilio_status():
    """Check Twilio configuration status."""
    has_twilio = bool(Config.TWILIO_ACCOUNT_SID and Config.TWILIO_AUTH_TOKEN)
    has_elevenlabs = bool(Config.ELEVENLABS_API_KEY)

    return JSONResponse({
        "twilio_configured": has_twilio,
        "elevenlabs_configured": has_elevenlabs,
        "twilio_phone": Config.TWILIO_PHONE_NUMBER if has_twilio else None,
        "server_url": Config.SERVER_URL,
        "webhook_url": f"{Config.SERVER_URL}/twilio/voice",
        "stream_url": f"{Config.SERVER_URL}/ws/twilio-stream",
    })


# ==========================================
# Browser-based Voice Endpoints (existing)
# ==========================================

@app.websocket("/ws/realtime")
async def websocket_realtime_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for ultra-low latency voice communication using OpenAI Realtime API.
    
    Provides ~300-500ms latency vs ~2900ms with traditional 3-API-call approach.
    
    Message format (Client -> Server):
    {
        "type": "audio_chunk",
        "data": "<base64-pcm16-audio>"
    }
    
    Message format (Server -> Client):
    {
        "type": "response_audio",
        "data": "<base64-pcm16-audio>",
        "format": "pcm16"
    }
    """
    await websocket.accept()
    logger.info("Realtime WebSocket connection established")
    
    # Get voice from query params or use default
    voice = websocket.query_params.get("voice", Config.REALTIME_VOICE)
    
    # Initialize Realtime service
    realtime_service = RealtimeService(voice=voice)
    
    try:
        # Connect to OpenAI Realtime API
        connected = await realtime_service.connect()
        if not connected:
            await websocket.send_json({
                "type": "error",
                "code": "REALTIME_CONNECTION_FAILED",
                "message": "Failed to connect to OpenAI Realtime API"
            })
            await websocket.close()
            return
        
        # Create event handler
        event_handler = RealtimeEventHandler(websocket, realtime_service)
        
        # Task to receive events from OpenAI and forward to client
        async def receive_from_openai():
            try:
                async for event in realtime_service.receive_events():
                    await event_handler.handle_event(event)
            except asyncio.CancelledError:
                logger.info("OpenAI event receiver cancelled")
            except Exception as e:
                logger.exception(f"Error in OpenAI event receiver: {e}")
        
        # Task to receive audio from client and forward to OpenAI
        async def receive_from_client():
            try:
                async for message in websocket.iter_json():
                    message_type = message.get("type")
                    
                    if message_type == "audio_chunk":
                        audio_base64 = message.get("data")
                        if audio_base64:
                            await realtime_service.send_audio(audio_base64)
                    
                    elif message_type == "commit_audio":
                        await realtime_service.commit_audio()
                        await realtime_service.create_response()
                    
                    elif message_type == "interrupt":
                        await realtime_service.cancel_response()
                    
                    elif message_type == "end_conversation":
                        logger.info("Client ended realtime conversation")
                        break
                        
            except WebSocketDisconnect:
                logger.info("Client disconnected from realtime endpoint")
            except RuntimeError as e:
                if "WebSocket is not connected" in str(e) or "Cannot call" in str(e):
                    logger.info("Client disconnected abruptly (RuntimeError)")
                else:
                    logger.error(f"RuntimeError in client receiver: {e}")
            except asyncio.CancelledError:
                logger.info("Client receiver cancelled")
            except Exception as e:
                logger.exception(f"Error in client receiver: {e}")
        
        # Run both tasks concurrently
        task_openai = asyncio.create_task(receive_from_openai())
        task_client = asyncio.create_task(receive_from_client())
        
        done, pending = await asyncio.wait(
            [task_openai, task_client],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        for task in pending:
            task.cancel()
    except Exception as e:
        logger.error(f"Realtime WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "code": "REALTIME_ERROR",
                "message": str(e)
            })
        except Exception:
            pass
    
    finally:
        await realtime_service.disconnect()
        logger.info("Realtime WebSocket connection closed")


@app.websocket("/ws/voice")
async def websocket_voice_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time voice communication

    Message format (Client -> Server):
    {
        "type": "audio_chunk",
        "data": "<base64-audio>",
        "format": "webm"
    }

    {
        "type": "end_conversation"
    }

    Message format (Server -> Client):
    {
        "type": "transcription",
        "text": "...",
        "is_final": true
    }

    {
        "type": "response_text",
        "text": "...",
        "is_complete": false
    }

    {
        "type": "response_audio",
        "data": "<base64-audio>",
        "sequence": 0
    }

    {
        "type": "summary",
        "summary": {...},
        "metadata": {...}
    }

    {
        "type": "error",
        "code": "...",
        "message": "..."
    }
    """
    await websocket.accept()
    logger.info("WebSocket connection established")

    # Initialize services for this session
    whisper_service = WhisperService()
    llm_service = LLMService()
    tts_service = TTSService()
    conversation_manager = ConversationManager()

    # Start conversation
    conversation_manager.start_conversation()

    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to AI Voice Agent"
        })

        async for message in websocket.iter_json():
            try:
                message_type = message.get("type")

                # Handle audio chunk from client
                if message_type == "audio_chunk":
                    logger.info(f"Received audio chunk message")
                    audio_base64 = message.get("data")
                    audio_format = message.get("format", "webm")

                    if not audio_base64:
                        await websocket.send_json({
                            "type": "error",
                            "code": "MISSING_AUDIO",
                            "message": "No audio data provided"
                        })
                        continue

                    # Decode audio
                    try:
                        audio_bytes = decode_base64_audio(audio_base64)
                    except Exception as e:
                        logger.error(f"Failed to decode audio: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "code": "DECODE_ERROR",
                            "message": "Failed to decode audio data"
                        })
                        continue

                    # Enhanced validation - check size and format (lowered for faster response)
                    if len(audio_bytes) < 2000:  # Reduced minimum for shorter utterances
                        logger.warning(f"Audio too short: {len(audio_bytes)} bytes, skipping")
                        # Don't send error to client, just skip silently
                        continue

                    # Check if audio is likely corrupted (reduced threshold for faster processing)
                    if audio_format == "webm" and len(audio_bytes) < 5000:
                        logger.warning(f"WebM audio suspiciously small: {len(audio_bytes)} bytes, skipping")
                        continue

                    logger.info(f"Audio chunk received: {len(audio_bytes)} bytes, format: {audio_format}")

                    # Convert to WAV format for reliable transcription
                    # MediaRecorder chunks aren't standalone valid WebM files
                    try:
                        processed_audio = convert_to_wav(audio_bytes, audio_format)
                        logger.info(f"Converted to WAV: {len(processed_audio)} bytes")
                    except Exception as e:
                        logger.error(f"Audio conversion failed: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "code": "CONVERSION_ERROR",
                            "message": "Failed to process audio format"
                        })
                        continue

                    # Transcribe with Whisper
                    try:
                        transcription_result = await whisper_service.transcribe(
                            processed_audio,
                            audio_format="wav"
                        )
                        transcription_text = transcription_result["text"]

                        logger.info(f"Transcription: {transcription_text}")

                        # Send transcription to client
                        await websocket.send_json({
                            "type": "transcription",
                            "text": transcription_text,
                            "is_final": True
                        })

                        # Add to conversation history
                        conversation_manager.add_message("user", transcription_text)

                    except Exception as e:
                        logger.error(f"Transcription failed: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "code": "STT_ERROR",
                            "message": "Failed to transcribe audio"
                        })
                        continue

                    # Generate and stream LLM response
                    try:
                        sentence_sequence = 0

                        async for sentence in llm_service.generate_response_stream(transcription_text):
                            # Send text response to client
                            await websocket.send_json({
                                "type": "response_text",
                                "text": sentence,
                                "sequence": sentence_sequence
                            })

                            # Add to conversation manager
                            if sentence_sequence == 0:
                                conversation_manager.add_message("assistant", sentence)
                            else:
                                # Append to last assistant message
                                if conversation_manager.messages:
                                    conversation_manager.messages[-1]["content"] += " " + sentence

                            # Generate TTS for sentence
                            try:
                                audio_bytes = await tts_service.synthesize(sentence)

                                # Encode and send audio
                                audio_base64 = encode_audio_to_base64(audio_bytes)
                                await websocket.send_json({
                                    "type": "response_audio",
                                    "data": audio_base64,
                                    "sequence": sentence_sequence
                                })

                                sentence_sequence += 1

                            except Exception as e:
                                logger.error(f"TTS failed for sentence: {e}")
                                # Continue with text-only response
                                continue

                    except Exception as e:
                        logger.error(f"LLM response generation failed: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "code": "LLM_ERROR",
                            "message": "Failed to generate response"
                        })
                        continue

                # Handle interrupt (user wants to stop AI mid-speech)
                elif message_type == "interrupt":
                    logger.info("User interrupted - clearing response")
                    # Just acknowledge - frontend already stopped playback
                    # Future chunks will be ignored by client
                    continue

                # Handle end conversation
                elif message_type == "end_conversation":
                    logger.info("Ending conversation and generating summary")

                    # End conversation
                    conversation_manager.end_conversation()

                    # Generate summary
                    try:
                        summary = await conversation_manager.generate_summary()

                        await websocket.send_json({
                            "type": "summary",
                            "summary": summary
                        })

                        logger.info("Summary sent to client")

                    except Exception as e:
                        logger.error(f"Failed to generate summary: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "code": "SUMMARY_ERROR",
                            "message": "Failed to generate conversation summary"
                        })

                    # Close connection gracefully
                    await websocket.send_json({
                        "type": "conversation_ended",
                        "message": "Conversation ended successfully"
                    })

                    break  # Exit loop and close connection

                else:
                    await websocket.send_json({
                        "type": "error",
                        "code": "UNKNOWN_MESSAGE_TYPE",
                        "message": f"Unknown message type: {message_type}"
                    })

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "code": "PROCESSING_ERROR",
                    "message": f"Error processing message: {str(e)}"
                })

    except WebSocketDisconnect:
        logger.info("Client disconnected")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")

    finally:
        logger.info("WebSocket connection closed")


if __name__ == "__main__":
    logger.info("Starting AI Voice Agent server...")
    logger.info(f"Environment: {Config.ENVIRONMENT}")
    logger.info(f"Frontend path: {frontend_path}")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=Config.is_development(),
        log_level=Config.LOG_LEVEL.lower()
    )
