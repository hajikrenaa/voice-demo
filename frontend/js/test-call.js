/**
 * Test Call Client — in-browser test that reuses the EXACT same backend
 * pipeline as a live Vobiz call. No Vobiz credits are consumed.
 *
 * Flow:
 *   1. Open /ws/test-call?token=... (session token from localStorage).
 *   2. Send {type:"start", elevenlabs}.
 *   3. Capture mic as PCM16 24 kHz and stream via {type:"audio_chunk"}.
 *   4. Play PCM16 24 kHz from {type:"response_audio"}.
 *   5. Render transcripts live, handle clear_audio (interruption barge-in).
 */

class TestCallClient {
    constructor(options = {}) {
        this.ws = null;
        this.recorder = null;
        this.player = null;
        this.isActive = false;
        this.useElevenLabs = !!options.elevenlabs;

        // UI callbacks
        this.onStatus = options.onStatus || (() => {});
        this.onTranscript = options.onTranscript || (() => {});
        this.onEnded = options.onEnded || (() => {});
        this.onError = options.onError || ((e) => console.error(e));
    }

    async start() {
        if (this.isActive) return;

        const token = localStorage.getItem('authToken');
        if (!token) {
            this.onError(new Error('Not logged in'));
            return;
        }

        this.onStatus('connecting', 'Requesting microphone…');

        this.recorder = new AudioRecorder({
            pcmMode: true,
            silenceThreshold: 0.8,
            energyThreshold: 0.015,
            onPCMData: (b64) => this._handleMicData(b64),
            onSpeechStart: () => this._handleUserSpeechStart(),
            onError: (e) => this.onError(e),
        });

        try {
            await this.recorder.initializePCMMode();
        } catch (e) {
            this.onError(e);
            this._teardown();
            return;
        }

        this.player = new AudioPlayer({
            onPlaybackStart: () => this.onStatus('speaking', 'AI speaking'),
            onPlaybackEnd: () => {
                if (this.isActive) this.onStatus('listening', 'Listening');
            },
        });

        // Open WS
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const el = this.useElevenLabs ? 'true' : 'false';
        const url = `${proto}//${location.host}/ws/test-call`
            + `?token=${encodeURIComponent(token)}`
            + `&elevenlabs=${el}`;

        this.onStatus('connecting', 'Connecting…');

        try {
            this.ws = new WebSocket(url);
        } catch (e) {
            this.onError(e);
            this._teardown();
            return;
        }

        this.ws.onopen = () => {
            // Send start, then begin streaming mic
            this.ws.send(JSON.stringify({
                type: 'start',
                elevenlabs: this.useElevenLabs,
            }));
            this.isActive = true;
            try {
                this.recorder.startPCMStreaming();
            } catch (e) {
                this.onError(e);
                this.stop();
                return;
            }
            this.onStatus('listening', 'Connected — listening');
        };

        this.ws.onmessage = (ev) => this._handleServerMessage(ev.data);

        this.ws.onerror = (ev) => {
            console.error('Test call WS error', ev);
            this.onError(new Error('WebSocket error'));
        };

        this.ws.onclose = () => {
            if (this.isActive) {
                this.onStatus('ended', 'Call ended');
                this.isActive = false;
                this._teardown();
                this.onEnded();
            }
        };
    }

    async stop() {
        if (!this.isActive && !this.ws) return;
        this.isActive = false;
        try {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'stop' }));
            }
        } catch (_) { /* ignore */ }
        try {
            if (this.ws) this.ws.close();
        } catch (_) { /* ignore */ }
        this._teardown();
        this.onStatus('ended', 'Call ended');
        this.onEnded();
    }

    _handleMicData(b64) {
        if (!this.isActive) return;
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'audio_chunk',
                data: b64,
            }));
        }
    }

    _handleUserSpeechStart() {
        // The backend VAD also runs on the server; we don't need to send an
        // interrupt event. This is just for UI feedback.
        if (this.isActive) this.onStatus('listening', 'You are speaking…');
    }

    _handleServerMessage(raw) {
        let msg;
        try {
            msg = JSON.parse(raw);
        } catch (_) {
            return;
        }

        switch (msg.type) {
            case 'response_audio':
                if (this.player && msg.data) {
                    this.player.enqueuePCM(msg.data);
                }
                break;

            case 'clear_audio':
                // Server-side interruption: stop all queued/current AI audio
                if (this.player) this.player.stopPCM();
                this.onStatus('listening', 'Listening');
                break;

            case 'transcript':
                this.onTranscript(msg.role, msg.text, msg.final);
                break;

            case 'ended':
                // Server-initiated hangup (e.g. goodbye)
                this.onStatus('ended', 'Call ended by AI');
                this.stop();
                break;

            default:
                // Ignore unknown message types
                break;
        }
    }

    _teardown() {
        if (this.recorder) {
            try { this.recorder.cleanup(); } catch (_) { /* ignore */ }
            this.recorder = null;
        }
        if (this.player) {
            try { this.player.stopPCM(); } catch (_) { /* ignore */ }
            try { this.player.cleanup(); } catch (_) { /* ignore */ }
            this.player = null;
        }
        this.ws = null;
    }
}

window.TestCallClient = TestCallClient;
