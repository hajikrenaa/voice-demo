"""
OpenAI Realtime API Service

Provides ultra-low latency voice-to-voice conversation using OpenAI's
Realtime API with native speech-to-speech capabilities.
"""

import asyncio
import base64
import json
import logging
from typing import AsyncIterator, Callable, Optional
import websockets
from config import Config

logger = logging.getLogger(__name__)


class RealtimeService:
    """
    OpenAI Realtime API WebSocket client for voice conversations.
    
    Connects to wss://api.openai.com/v1/realtime for native audio-to-audio
    processing with ~300-500ms latency.
    """
    
    REALTIME_URL = "wss://api.openai.com/v1/realtime"
    
    def __init__(self, voice: str = None):
        """
        Initialize Realtime service.
        
        Args:
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
        """
        self.api_key = Config.OPENAI_API_KEY
        self.model = getattr(Config, 'REALTIME_MODEL', 'gpt-4o-realtime-preview')
        self.voice = voice or getattr(Config, 'REALTIME_VOICE', Config.TTS_VOICE)
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.session_id: Optional[str] = None
        self._connected = False
        self._response_in_progress = False
        
    async def connect(self) -> bool:
        """
        Establish WebSocket connection to OpenAI Realtime API.
        
        Returns:
            bool: True if connection successful
        """
        try:
            url = f"{self.REALTIME_URL}?model={self.model}"
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1"
            }
            
            logger.info(f"Connecting to OpenAI Realtime API: {self.model}")
            
            self.ws = await websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=30,
                ping_timeout=10
            )
            
            self._connected = True
            logger.info("Connected to OpenAI Realtime API")
            
            # Configure session
            await self._configure_session()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Realtime API: {e}")
            self._connected = False
            return False
    
    async def _configure_session(self):
        """Configure the realtime session with voice and instructions."""
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": Config.SYSTEM_PROMPT,
                "voice": self.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1",
                    "language": "en"  # Force English transcription
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.4,  # Lower = more sensitive (was 0.5)
                    "prefix_padding_ms": 200,  # Reduced from 300ms
                    "silence_duration_ms": 300  # Reduced from 500ms - faster response!
                }
            }
        }
        
        await self.ws.send(json.dumps(session_config))
        logger.info(f"Session configured with voice: {self.voice}, language: English")
    
    async def send_audio(self, audio_base64: str):
        """
        Send audio chunk to Realtime API.
        
        Args:
            audio_base64: Base64-encoded PCM16 audio (24kHz, mono)
        """
        if not self._connected or not self.ws:
            logger.warning("Cannot send audio: not connected")
            return
            
        event = {
            "type": "input_audio_buffer.append",
            "audio": audio_base64
        }
        
        await self.ws.send(json.dumps(event))
    
    async def commit_audio(self):
        """Commit the audio buffer to trigger processing."""
        if not self._connected or not self.ws:
            return
            
        event = {"type": "input_audio_buffer.commit"}
        await self.ws.send(json.dumps(event))
    
    async def create_response(self):
        """Request the model to generate a response."""
        if not self._connected or not self.ws:
            return
            
        event = {"type": "response.create"}
        await self.ws.send(json.dumps(event))
        self._response_in_progress = True
    
    async def cancel_response(self):
        """Cancel the current response (for interruptions)."""
        if not self._connected or not self.ws:
            return
            
        event = {"type": "response.cancel"}
        await self.ws.send(json.dumps(event))
        self._response_in_progress = False
    
    async def receive_events(self) -> AsyncIterator[dict]:
        """
        Receive and yield events from the Realtime API.
        
        Yields:
            dict: Event data from OpenAI
        """
        if not self._connected or not self.ws:
            return
            
        try:
            async for message in self.ws:
                try:
                    event = json.loads(message)
                    event_type = event.get("type", "")
                    
                    # Log important events
                    if event_type not in ["response.audio.delta"]:
                        logger.debug(f"Received event: {event_type}")
                    
                    yield event
                    
                    # Track response completion
                    if event_type == "response.done":
                        self._response_in_progress = False
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse event: {e}")
                    
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"WebSocket connection closed: {e}")
            self._connected = False
            
        except Exception as e:
            logger.error(f"Error receiving events: {e}")
            self._connected = False
    
    async def disconnect(self):
        """Close the WebSocket connection."""
        if self.ws:
            try:
                await self.ws.close()
                logger.info("Disconnected from Realtime API")
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
            finally:
                self.ws = None
                self._connected = False
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to Realtime API."""
        return self._connected and self.ws is not None
    
    @property
    def is_responding(self) -> bool:
        """Check if a response is in progress."""
        return self._response_in_progress


class RealtimeEventHandler:
    """
    Handler for processing Realtime API events and bridging to client WebSocket.
    """
    
    def __init__(self, client_ws, realtime_service: RealtimeService):
        """
        Initialize event handler.
        
        Args:
            client_ws: FastAPI WebSocket connection to browser
            realtime_service: RealtimeService instance
        """
        self.client_ws = client_ws
        self.realtime = realtime_service
        self.transcription_text = ""
        self.response_text = ""
        
    async def handle_event(self, event: dict):
        """
        Process an event from OpenAI and forward to client.
        
        Args:
            event: Event dict from Realtime API
        """
        event_type = event.get("type", "")
        
        # Session events
        if event_type == "session.created":
            await self.client_ws.send_json({
                "type": "connected",
                "message": "Connected to AI Voice Agent (Realtime)"
            })
            
        elif event_type == "session.updated":
            logger.info("Session updated successfully")
            
        # Input audio events
        elif event_type == "input_audio_buffer.speech_started":
            # User started speaking - could cancel ongoing response
            if self.realtime.is_responding:
                try:
                    await self.realtime.cancel_response()
                    await self.client_ws.send_json({
                        "type": "interrupted",
                        "message": "Response interrupted by user"
                    })
                except Exception as e:
                    logger.debug(f"Cancel response failed (may not have active response): {e}")
                
        elif event_type == "input_audio_buffer.speech_stopped":
            # User stopped speaking
            logger.debug("Speech stopped, waiting for transcription")
            
        elif event_type == "input_audio_buffer.committed":
            logger.debug("Audio buffer committed")
            
        # Transcription events
        elif event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            if transcript:
                self.transcription_text = transcript
                await self.client_ws.send_json({
                    "type": "transcription",
                    "text": transcript,
                    "is_final": True
                })
                
        # Response events
        elif event_type == "response.audio.delta":
            # Stream audio chunk to client
            audio_delta = event.get("delta", "")
            if audio_delta:
                await self.client_ws.send_json({
                    "type": "response_audio",
                    "data": audio_delta,
                    "format": "pcm16"
                })
                
        elif event_type == "response.audio_transcript.delta":
            # Stream text transcript of response
            text_delta = event.get("delta", "")
            if text_delta:
                self.response_text += text_delta
                await self.client_ws.send_json({
                    "type": "response_text",
                    "text": text_delta,
                    "is_delta": True
                })
                
        elif event_type == "response.audio_transcript.done":
            # Full transcript available
            transcript = event.get("transcript", self.response_text)
            await self.client_ws.send_json({
                "type": "response_text",
                "text": transcript,
                "is_complete": True
            })
            self.response_text = ""
            
        elif event_type == "response.done":
            # Response complete
            await self.client_ws.send_json({
                "type": "response_complete"
            })
            
        # Error events
        elif event_type == "error":
            error_msg = event.get("error", {}).get("message", "Unknown error")
            logger.error(f"Realtime API error: {error_msg}")
            await self.client_ws.send_json({
                "type": "error",
                "code": "REALTIME_ERROR",
                "message": error_msg
            })
