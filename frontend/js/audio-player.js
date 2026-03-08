/**
 * Audio Player
 * Handles queued playback of AI voice responses
 */

class AudioPlayer {
    constructor(options = {}) {
        this.queue = [];
        this.isPlaying = false;
        this.currentAudio = null;
        this.hasCalledPlaybackStart = false;

        // PCM16 playback (for Realtime API)
        this.audioContext = null;
        this.pcmQueue = [];
        this.isPCMPlaying = false;
        this.pcmSampleRate = 24000; // OpenAI Realtime API outputs 24kHz
        this.pcmNextStartTime = 0;

        // Callbacks
        this.onPlaybackStart = options.onPlaybackStart || (() => { });
        this.onPlaybackEnd = options.onPlaybackEnd || (() => { });
        this.onError = options.onError || ((error) => console.error(error));
    }

    /**
     * Add audio to the playback queue
     */
    enqueue(audioBase64, format = 'mp3') {
        // Convert Base64 to audio blob
        const audioBlob = this.base64ToBlob(audioBase64, format);
        const audioUrl = URL.createObjectURL(audioBlob);

        this.queue.push({
            url: audioUrl,
            blob: audioBlob,
            format: format
        });

        console.log(`Audio enqueued (queue length: ${this.queue.length})`);

        // Start playback if not already playing
        if (!this.isPlaying) {
            this.playNext();
        }
    }

    /**
     * Play the next audio in the queue
     */
    async playNext() {
        if (this.queue.length === 0) {
            this.isPlaying = false;
            // Only call onPlaybackEnd if we previously called onPlaybackStart
            if (this.hasCalledPlaybackStart) {
                this.hasCalledPlaybackStart = false;
                this.onPlaybackEnd();
            }
            return;
        }

        const audioItem = this.queue.shift();
        this.isPlaying = true;

        try {
            await this.playAudio(audioItem.url);

            // Clean up blob URL
            URL.revokeObjectURL(audioItem.url);

            // Play next audio
            this.playNext();

        } catch (error) {
            console.error('Error playing audio:', error);
            this.onError(error);

            // Continue with next audio despite error
            URL.revokeObjectURL(audioItem.url);
            this.playNext();
        }
    }

    /**
     * Play a single audio file
     */
    playAudio(audioUrl) {
        return new Promise((resolve, reject) => {
            this.currentAudio = new Audio(audioUrl);

            this.currentAudio.onloadedmetadata = () => {
                console.log(`Playing audio (duration: ${this.currentAudio.duration.toFixed(2)}s)`);
                // Only call onPlaybackStart once per playback session
                if (!this.hasCalledPlaybackStart) {
                    this.hasCalledPlaybackStart = true;
                    this.onPlaybackStart();
                }
            };

            this.currentAudio.onended = () => {
                console.log('Audio playback completed');
                this.currentAudio = null;
                resolve();
            };

            this.currentAudio.onerror = (error) => {
                console.error('Audio playback error:', error);
                this.currentAudio = null;
                reject(error);
            };

            // Start playback
            this.currentAudio.play().catch(reject);
        });
    }

    /**
     * Convert Base64 to Blob
     */
    base64ToBlob(base64, format) {
        // Determine MIME type
        const mimeType = format === 'mp3' ? 'audio/mpeg' :
            format === 'opus' ? 'audio/opus' :
                format === 'wav' ? 'audio/wav' :
                    'audio/mpeg';

        // Decode Base64
        const byteCharacters = atob(base64);
        const byteNumbers = new Array(byteCharacters.length);

        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }

        const byteArray = new Uint8Array(byteNumbers);
        return new Blob([byteArray], { type: mimeType });
    }

    /**
     * Stop current playback
     */
    stop() {
        if (this.currentAudio) {
            this.currentAudio.pause();
            this.currentAudio.currentTime = 0;
            this.currentAudio = null;
        }

        this.isPlaying = false;

        // Call onPlaybackEnd if we're stopping mid-playback
        if (this.hasCalledPlaybackStart) {
            this.hasCalledPlaybackStart = false;
            this.onPlaybackEnd();
        }

        console.log('Playback stopped');
    }

    /**
     * Clear the playback queue
     */
    clearQueue() {
        // Clean up any blob URLs in queue
        this.queue.forEach(item => {
            if (item.url) {
                URL.revokeObjectURL(item.url);
            }
        });

        this.queue = [];
        console.log('Playback queue cleared');
    }

    /**
     * Pause current playback
     */
    pause() {
        if (this.currentAudio) {
            this.currentAudio.pause();
            console.log('Playback paused');
        }
    }

    /**
     * Resume playback
     */
    resume() {
        if (this.currentAudio && this.currentAudio.paused) {
            this.currentAudio.play();
            console.log('Playback resumed');
        }
    }

    /**
     * Set playback volume
     */
    setVolume(volume) {
        if (volume < 0 || volume > 1) {
            throw new Error('Volume must be between 0 and 1');
        }

        if (this.currentAudio) {
            this.currentAudio.volume = volume;
        }

        console.log(`Volume set to: ${volume}`);
    }

    /**
     * Initialize PCM16 audio context for Realtime API playback
     */
    initializePCM() {
        if (!this.audioContext) {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.pcmSampleRate
            });
            console.log(`PCM AudioContext initialized at ${this.audioContext.sampleRate}Hz`);
        }
        return this.audioContext;
    }

    /**
     * Enqueue PCM16 audio for playback (from Realtime API)
     * @param {string} base64Audio - Base64 encoded PCM16 audio
     */
    enqueuePCM(base64Audio) {
        // Initialize audio context if needed
        this.initializePCM();

        // Decode base64 to Int16Array
        const binaryString = atob(base64Audio);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }

        // Convert to Int16Array
        const int16Data = new Int16Array(bytes.buffer);

        // Convert Int16 to Float32 for Web Audio API
        const float32Data = new Float32Array(int16Data.length);
        for (let i = 0; i < int16Data.length; i++) {
            float32Data[i] = int16Data[i] / 32768.0;
        }

        // Create audio buffer
        const audioBuffer = this.audioContext.createBuffer(
            1, // mono
            float32Data.length,
            this.pcmSampleRate
        );
        audioBuffer.getChannelData(0).set(float32Data);

        // Enqueue for playback
        this.pcmQueue.push(audioBuffer);

        // Start playback if not already playing
        if (!this.isPCMPlaying) {
            this.playNextPCM();
        }
    }

    /**
     * Play the next PCM audio buffer in queue
     */
    playNextPCM() {
        if (this.pcmQueue.length === 0) {
            this.isPCMPlaying = false;
            if (this.hasCalledPlaybackStart) {
                this.hasCalledPlaybackStart = false;
                this.onPlaybackEnd();
            }
            return;
        }

        this.isPCMPlaying = true;

        // Call onPlaybackStart once per playback session
        if (!this.hasCalledPlaybackStart) {
            this.hasCalledPlaybackStart = true;
            this.onPlaybackStart();
        }

        const audioBuffer = this.pcmQueue.shift();
        const source = this.audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.audioContext.destination);

        // Schedule for immediate or seamless playback
        const currentTime = this.audioContext.currentTime;
        const startTime = Math.max(currentTime, this.pcmNextStartTime);

        source.start(startTime);
        this.pcmNextStartTime = startTime + audioBuffer.duration;

        // Continue to next buffer when this one ends
        source.onended = () => {
            this.playNextPCM();
        };
    }

    /**
     * Stop PCM playback and clear queue
     */
    stopPCM() {
        this.pcmQueue = [];
        this.isPCMPlaying = false;
        this.pcmNextStartTime = 0;

        if (this.hasCalledPlaybackStart) {
            this.hasCalledPlaybackStart = false;
            this.onPlaybackEnd();
        }

        console.log('PCM playback stopped');
    }

    /**
     * Get current playback state
     */
    getState() {
        return {
            isPlaying: this.isPlaying || this.isPCMPlaying,
            queueLength: this.queue.length + this.pcmQueue.length,
            currentTime: this.currentAudio ? this.currentAudio.currentTime : 0,
            duration: this.currentAudio ? this.currentAudio.duration : 0
        };
    }

    /**
     * Clean up resources
     */
    cleanup() {
        this.stop();
        this.clearQueue();
        console.log('Audio player cleaned up');
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AudioPlayer;
}
