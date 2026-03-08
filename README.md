# AI Voice Agent

A professional web-based AI voice agent with human-like speech, >90% accuracy, and low latency. Built with Python (FastAPI) backend and vanilla JavaScript frontend, powered entirely by OpenAI APIs (Whisper, GPT-4o, TTS).

## Features

- **Real-time Voice Conversation**: Speak naturally with AI using your microphone
- **High Accuracy**: >90% speech recognition accuracy using OpenAI Whisper
- **Low Latency**: <3 seconds from speech end to AI response start
- **Voice Activity Detection (VAD)**: Intelligent pause detection for natural conversations
- **Streaming Responses**: AI starts speaking while still generating the complete response
- **Conversation Summary**: Automatic summary generation at conversation end
- **Beautiful UI**: Modern, responsive web interface
- **Multiple Voices**: Choose from 6 different AI voices

## Tech Stack

### Backend
- **Framework**: FastAPI + WebSocket
- **Speech-to-Text**: OpenAI Whisper API
- **LLM**: GPT-4o (streaming)
- **Text-to-Speech**: OpenAI TTS
- **Audio Processing**: Pydub, NumPy

### Frontend
- **HTML/CSS/JavaScript** (vanilla, no framework)
- **WebSocket** for real-time communication
- **Web Audio API** for recording and VAD
- **MediaRecorder API** for audio capture

## Prerequisites

- Python 3.9 or higher
- OpenAI API key
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Microphone access

## Installation

### 1. Clone or Navigate to Project

```bash
cd "D:\Project\voice demo"
```

### 2. Set Up Backend

#### Create Virtual Environment

```bash
cd backend
python -m venv venv
```

#### Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Configure Environment Variables

Create a `.env` file in the `backend` directory:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-actual-openai-api-key-here
ENVIRONMENT=development
LOG_LEVEL=INFO
```

**Important**: Replace `sk-your-actual-openai-api-key-here` with your actual OpenAI API key from https://platform.openai.com/api-keys

## Usage

### Start the Server

From the `backend` directory with the virtual environment activated:

```bash
python main.py
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### Access the Application

Open your web browser and navigate to:

```
http://localhost:8000
```

### Using the Voice Agent

1. **Start Conversation**: Click the "Start Conversation" button
2. **Grant Microphone Permission**: Allow browser to access your microphone
3. **Speak Naturally**: Start speaking - the AI will detect when you pause
4. **Listen to Response**: The AI will respond with voice and text
5. **Continue Conversation**: Keep speaking - the system automatically detects turns
6. **End Conversation**: Click "Stop Conversation" to end and get a summary

## Configuration

### Backend Settings (backend/config.py)

You can modify various settings:

```python
# Audio Settings
SAMPLE_RATE = 16000  # Whisper optimal sample rate
VAD_SILENCE_THRESHOLD = 1.5  # Seconds of silence before user is done

# Model Configuration
WHISPER_MODEL = "whisper-1"
LLM_MODEL = "gpt-4o"  # or "gpt-4o-mini" for faster/cheaper
TTS_MODEL = "tts-1"  # or "tts-1-hd" for higher quality
TTS_VOICE = "alloy"  # Options: alloy, echo, fable, onyx, nova, shimmer
```

### Frontend Settings

Adjust settings in the UI:
- **Voice**: Choose from 6 different AI voices
- **Silence Threshold**: Adjust how long the system waits before processing your speech (0.5-3 seconds)

## Project Structure

```
voice demo/
├── backend/
│   ├── main.py                      # FastAPI app + WebSocket server
│   ├── config.py                    # Configuration
│   ├── requirements.txt             # Python dependencies
│   ├── .env                         # Environment variables (create this)
│   ├── services/
│   │   ├── whisper_service.py       # Speech-to-Text
│   │   ├── llm_service.py           # GPT-4o conversation
│   │   ├── tts_service.py           # Text-to-Speech
│   │   └── conversation_manager.py  # History & summary
│   └── utils/
│       └── audio_processing.py      # Audio utilities
│
├── frontend/
│   ├── index.html                   # Main UI
│   ├── css/
│   │   └── styles.css               # Styling
│   └── js/
│       ├── main.js                  # App controller
│       ├── websocket-client.js      # WebSocket communication
│       ├── audio-recorder.js        # Recording + VAD
│       └── audio-player.js          # Audio playback
│
└── README.md
```

## API Endpoints

### WebSocket Endpoint

```
ws://localhost:8000/ws/voice
```

**Client → Server Messages:**
```json
{
    "type": "audio_chunk",
    "data": "<base64-audio>",
    "format": "webm"
}

{
    "type": "end_conversation"
}
```

**Server → Client Messages:**
```json
{
    "type": "transcription",
    "text": "User said...",
    "is_final": true
}

{
    "type": "response_audio",
    "data": "<base64-mp3-audio>",
    "sequence": 0
}

{
    "type": "summary",
    "summary": {
        "overview": "...",
        "key_points": [...],
        "topics": [...],
        "sentiment": "positive"
    }
}
```

### HTTP Endpoints

- `GET /` - Serve frontend
- `GET /api/health` - Health check
- `GET /api/config` - Get configuration

## Performance Metrics

### Target Metrics

- **Latency**: <3 seconds (user stops speaking → AI starts speaking)
- **Accuracy**: >90% STT accuracy
- **Cost**: <$0.50 per 10-minute conversation

### Typical Latency Breakdown

```
User stops speaking → AI starts speaking

1. VAD detects silence:        0.2s
2. Audio upload (3s chunk):     0.2s
3. Whisper API (STT):          0.8s
4. GPT-4o first sentence:      1.0s
5. TTS API synthesis:          0.5s
6. Audio download + play:      0.2s
-------------------------------------------
Total:                         2.9s ✓
```

## Troubleshooting

### Microphone Not Working

1. Check browser permissions: Click the lock icon in address bar
2. Ensure microphone is not being used by another application
3. Try a different browser (Chrome recommended)

### WebSocket Connection Failed

1. Ensure backend server is running (`python main.py`)
2. Check that port 8000 is not in use by another application
3. Check firewall settings

### Poor Audio Quality

1. Check your microphone quality and distance
2. Reduce background noise
3. Speak clearly and at a moderate pace

### OpenAI API Errors

1. Verify your API key is correct in `.env`
2. Check your OpenAI account has available credits
3. Review rate limits at https://platform.openai.com/account/limits

### High Latency

1. Check your internet connection speed
2. Use `gpt-4o-mini` instead of `gpt-4o` (faster, cheaper)
3. Use `tts-1` instead of `tts-1-hd` (faster)
4. Reduce VAD silence threshold in settings

## Cost Estimation

Based on OpenAI pricing (as of 2024):

- **Whisper (STT)**: $0.006 per minute
- **GPT-4o**: ~$0.01 per minute (varies by conversation)
- **TTS**: ~$0.015 per minute

**Total**: ~$0.03-0.05 per minute = **$0.30-0.50 per 10-minute conversation**

## Development

### Running in Development Mode

The server automatically runs in development mode with hot reload:

```bash
python main.py
```

### Running in Production Mode

Set environment variable:

```bash
ENVIRONMENT=production python main.py
```

Or update `.env`:
```
ENVIRONMENT=production
```

## Security Considerations

### For Production Deployment

1. **HTTPS/WSS**: Use SSL/TLS certificates
2. **CORS**: Restrict allowed origins in `main.py`
3. **Rate Limiting**: Implement rate limiting to prevent abuse
4. **API Key Security**: Never expose API keys in frontend
5. **Input Validation**: Already implemented in backend

## License

This project is for educational and demonstration purposes.

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review backend logs for error messages
3. Check browser console for frontend errors

## Acknowledgments

Powered by:
- OpenAI Whisper (Speech-to-Text)
- OpenAI GPT-4o (Conversation AI)
- OpenAI TTS (Text-to-Speech)
- FastAPI (Backend framework)

---

**Built with ❤️ using OpenAI APIs**
