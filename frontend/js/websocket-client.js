/**
 * WebSocket Client for AI Voice Agent
 * Handles real-time communication with the backend
 */

class WebSocketClient {
    constructor(options = {}) {
        this.ws = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000; // Start with 1 second
        this.messageHandlers = {};

        // Realtime API mode (ultra-low latency)
        this.realtimeMode = options.realtimeMode || false;
        this.voice = options.voice || 'alloy';
    }

    /**
     * Connect to WebSocket server
     * @param {boolean} useRealtimeAPI - If true, connect to Realtime API endpoint
     */
    connect(useRealtimeAPI = false) {
        // Update mode
        this.realtimeMode = useRealtimeAPI;

        return new Promise((resolve, reject) => {
            try {
                // Determine WebSocket URL
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const host = window.location.host || 'localhost:8000';

                // Select endpoint based on mode
                let wsUrl;
                if (this.realtimeMode) {
                    wsUrl = `${protocol}//${host}/ws/realtime?voice=${this.voice}`;
                    console.log('Connecting to Realtime API endpoint (ultra-low latency)');
                } else {
                    wsUrl = `${protocol}//${host}/ws/voice`;
                    console.log('Connecting to standard voice endpoint');
                }

                console.log(`WebSocket URL: ${wsUrl}`);

                this.ws = new WebSocket(wsUrl);

                // Connection opened
                this.ws.onopen = () => {
                    console.log('WebSocket connected');
                    this.isConnected = true;
                    this.reconnectAttempts = 0;
                    this.reconnectDelay = 1000;
                    resolve();
                };

                // Listen for messages
                this.ws.onmessage = (event) => {
                    try {
                        const message = JSON.parse(event.data);
                        this.handleMessage(message);
                    } catch (error) {
                        console.error('Failed to parse message:', error);
                    }
                };

                // Connection closed
                this.ws.onclose = () => {
                    console.log('WebSocket disconnected');
                    this.isConnected = false;
                    this.handleDisconnect();
                };

                // Connection error
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.isConnected = false;
                    reject(error);
                };

            } catch (error) {
                console.error('Failed to create WebSocket connection:', error);
                reject(error);
            }
        });
    }

    /**
     * Handle incoming messages from server
     */
    handleMessage(message) {
        const { type } = message;

        // Call registered handler for this message type
        if (this.messageHandlers[type]) {
            this.messageHandlers[type](message);
        } else {
            console.log('Unhandled message type:', type, message);
        }
    }

    /**
     * Register a handler for a specific message type
     */
    on(messageType, handler) {
        this.messageHandlers[messageType] = handler;
    }

    /**
     * Send a message to the server
     */
    send(message) {
        if (!this.isConnected || !this.ws) {
            console.error('WebSocket not connected');
            throw new Error('WebSocket not connected');
        }

        try {
            this.ws.send(JSON.stringify(message));
        } catch (error) {
            console.error('Failed to send message:', error);
            throw error;
        }
    }

    /**
     * Send audio chunk to server
     */
    sendAudioChunk(audioBase64, format = 'webm') {
        this.send({
            type: 'audio_chunk',
            data: audioBase64,
            format: format
        });
    }

    /**
     * Send interrupt signal to stop AI speech
     */
    interruptAI() {
        this.send({
            type: 'interrupt'
        });
    }

    /**
     * Commit audio buffer and request AI response (Realtime API mode)
     * Server VAD handles this automatically, but manual trigger available
     */
    commitAudio() {
        if (!this.realtimeMode) {
            console.warn('commitAudio is only for Realtime API mode');
            return;
        }
        this.send({
            type: 'commit_audio'
        });
    }

    /**
     * Send PCM16 audio for Realtime API (base64 encoded)
     */
    sendPCMAudio(audioBase64) {
        this.send({
            type: 'audio_chunk',
            data: audioBase64
        });
    }

    /**
     * Set voice for Realtime API
     */
    setVoice(voice) {
        this.voice = voice;
    }

    /**
     * Check if using Realtime API mode
     */
    isRealtimeMode() {
        return this.realtimeMode;
    }

    /**
     * Send end conversation signal
     */
    endConversation() {
        this.send({
            type: 'end_conversation'
        });
    }

    /**
     * Handle disconnection and attempt reconnect
     */
    handleDisconnect() {
        // Trigger disconnect handler
        if (this.messageHandlers['disconnected']) {
            this.messageHandlers['disconnected']();
        }

        // Attempt to reconnect
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);

            setTimeout(() => {
                this.connect().catch((error) => {
                    console.error('Reconnection failed:', error);
                });
            }, this.reconnectDelay);

            // Exponential backoff
            this.reconnectDelay = Math.min(this.reconnectDelay * 2, 30000); // Max 30 seconds
        } else {
            console.error('Max reconnection attempts reached');
            if (this.messageHandlers['reconnect_failed']) {
                this.messageHandlers['reconnect_failed']();
            }
        }
    }

    /**
     * Close the WebSocket connection
     */
    close() {
        if (this.ws) {
            this.isConnected = false;
            this.ws.close();
            this.ws = null;
        }
    }

    /**
     * Get connection status
     */
    getStatus() {
        return {
            isConnected: this.isConnected,
            reconnectAttempts: this.reconnectAttempts
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WebSocketClient;
}
