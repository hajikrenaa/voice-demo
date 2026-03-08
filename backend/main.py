from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, Response
import uvicorn
import logging
import sys
import asyncio
import json
from pathlib import Path

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
    preprocess_audio_for_whisper,
    validate_audio
)
from config import Config

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
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
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
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
        return HTMLResponse(content=f"<h1>Error: {str(e)}</h1>")


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
# Twilio Voice Calling Endpoints
# ==========================================

@app.post("/twilio/voice")
async def twilio_voice_webhook(request: Request):
    """
    Twilio voice webhook - called when an inbound/outbound call connects.
    Returns TwiML that tells Twilio to open a bidirectional media stream.
    """
    server_url = Config.SERVER_URL
    ws_url = server_url.replace("https://", "wss://").replace("http://", "ws://")

    # Check for elevenlabs param (passed from outbound call URL)
    use_elevenlabs = request.query_params.get("elevenlabs", "false")

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}/ws/twilio-stream">
            <Parameter name="caller" value="{{{{From}}}}" />
            <Parameter name="elevenlabs" value="{use_elevenlabs}" />
        </Stream>
    </Connect>
</Response>"""

    logger.info(f"Twilio voice webhook called - streaming to {ws_url}/ws/twilio-stream")
    return Response(content=twiml, media_type="application/xml")


@app.post("/twilio/outbound-call")
async def make_outbound_call(request: Request):
    """
    API endpoint to initiate an outbound call via Twilio.

    Request body: { "to": "+1234567890" }
    """
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

        voice_url = f"{Config.SERVER_URL}/twilio/voice?elevenlabs={'true' if use_elevenlabs else 'false'}"
        call = client.calls.create(
            to=to_number,
            from_=Config.TWILIO_PHONE_NUMBER,
            url=voice_url,
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
    await websocket.accept()
    logger.info("Twilio media stream WebSocket connected")

    # ElevenLabs flag comes via Twilio <Parameter> in the "start" event
    # We create handler later after reading the start event
    handler = TwilioRealtimeHandler(use_elevenlabs=False)

    try:
        async for raw_message in websocket.iter_text():
            try:
                message = json.loads(raw_message)
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
        await handler._disconnect_openai()
        logger.info(f"Twilio stream closed - CallSID: {handler.call_sid}")


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
            except Exception as e:
                logger.error(f"Error in OpenAI event receiver: {e}")
        
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
            except Exception as e:
                logger.error(f"Error in client receiver: {e}")
        
        # Run both tasks concurrently
        await asyncio.gather(
            receive_from_openai(),
            receive_from_client(),
            return_exceptions=True
        )
        
    except Exception as e:
        logger.error(f"Realtime WebSocket error: {e}")
        await websocket.send_json({
            "type": "error",
            "code": "REALTIME_ERROR",
            "message": str(e)
        })
    
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
