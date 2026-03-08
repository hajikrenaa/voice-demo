/**
 * Audio Recorder with Voice Activity Detection (VAD)
 * Handles browser-based audio recording and speech detection
 */

class AudioRecorder {
    constructor(options = {}) {
        this.mediaRecorder = null;
        this.audioContext = null;
        this.analyser = null;
        this.microphone = null;
        this.isRecording = false;
        this.isPaused = false;
        this.audioChunks = [];

        // VAD settings (optimized for faster response)
        this.silenceThreshold = options.silenceThreshold || 0.8; // seconds (reduced from 1.5s)
        this.energyThreshold = options.energyThreshold || 0.015; // 0-1 scale (lowered for sensitivity)
        this.lastSpeechTime = null;
        this.silenceTimer = null;

        // Advanced VAD settings (optimized for better listening)
        this.noiseFloor = 0.008; // Background noise level (calibrated dynamically)
        this.noiseGate = 2.5; // Speech must be 2.5x louder than noise (was 3x)
        this.minSpeechDuration = 150; // Minimum 150ms for short words (was 300ms)
        this.speechStartTime = null;
        this.noiseCalibrationSamples = [];
        this.maxNoiseCalibrationSamples = 15; // Faster calibration (was 30)
        this.isCalibrating = true;

        // PCM16 mode for Realtime API (ultra-low latency)
        this.pcmMode = options.pcmMode || false;
        this.scriptProcessor = null;
        this.targetSampleRate = 24000; // OpenAI Realtime API requires 24kHz
        this.pcmBufferSize = 4096;
        this.onPCMData = options.onPCMData || (() => { });

        // Callbacks
        this.onAudioData = options.onAudioData || (() => { });
        this.onSpeechStart = options.onSpeechStart || (() => { });
        this.onSpeechEnd = options.onSpeechEnd || (() => { });
        this.onVolumeChange = options.onVolumeChange || (() => { });
        this.onCalibrationComplete = options.onCalibrationComplete || (() => { });
        this.onError = options.onError || ((error) => console.error(error));

        // Recording settings
        this.mimeType = this.getSupportedMimeType();
    }

    /**
     * Get supported MIME type for recording
     */
    getSupportedMimeType() {
        const types = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/ogg;codecs=opus',
            'audio/mp4'
        ];

        for (const type of types) {
            if (MediaRecorder.isTypeSupported(type)) {
                console.log(`Using MIME type: ${type}`);
                return type;
            }
        }

        console.warn('No preferred MIME type supported, using default');
        return 'audio/webm';
    }

    /**
     * Request microphone permission and initialize
     */
    async initialize() {
        try {
            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: 16000 // Whisper optimal sample rate
                }
            });

            console.log('Microphone access granted');

            // Create audio context for VAD
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 2048;
            this.analyser.smoothingTimeConstant = 0.8;

            this.microphone = this.audioContext.createMediaStreamSource(stream);
            this.microphone.connect(this.analyser);

            // Create MediaRecorder
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: this.mimeType
            });

            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                    // Don't send chunks individually - wait for speech end
                    console.log(`Audio chunk buffered: ${event.data.size} bytes (total: ${this.audioChunks.length} chunks)`);
                }
            };

            this.mediaRecorder.onstop = () => {
                // Process and send the complete audio file
                if (this.audioChunks.length > 0 && !this.isPaused) {
                    console.log(`Recording stopped - processing ${this.audioChunks.length} chunks`);
                    this.processRecordedAudio();
                } else {
                    console.log('Recording stopped - no chunks or paused');
                }
                this.audioChunks = [];
            };

            return true;

        } catch (error) {
            console.error('Failed to initialize audio recorder:', error);
            this.onError(error);
            throw error;
        }
    }

    /**
     * Start recording with VAD
     */
    startRecording() {
        if (!this.mediaRecorder) {
            throw new Error('Recorder not initialized. Call initialize() first.');
        }

        if (this.isRecording) {
            console.warn('Already recording');
            return;
        }

        this.isRecording = true;
        this.isPaused = false;
        this.lastSpeechTime = null;
        this.isRecordingActive = false; // Track if MediaRecorder is actively recording

        // Start VAD monitoring first (but don't start MediaRecorder yet)
        this.startVADMonitoring();

        console.log('Recording session started - waiting for speech...');
    }

    /**
     * Start actual MediaRecorder (called when speech detected)
     */
    startMediaRecorder() {
        if (this.isRecordingActive) {
            return; // Already recording
        }

        // In PCM mode, MediaRecorder is not used - audio is streamed continuously
        if (this.pcmMode || !this.mediaRecorder) {
            this.isRecordingActive = true;
            return;
        }

        this.audioChunks = [];
        this.isRecordingActive = true;

        // Start WITHOUT time slicing - creates one complete WebM file
        this.mediaRecorder.start();
        console.log('MediaRecorder started');
    }

    /**
     * Stop MediaRecorder (called when silence detected)
     */
    stopMediaRecorder() {
        if (!this.isRecordingActive) {
            return;
        }

        // In PCM mode, MediaRecorder is not used
        if (this.pcmMode || !this.mediaRecorder) {
            this.isRecordingActive = false;
            return;
        }

        if (this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
            console.log('MediaRecorder stopped');
        }
        this.isRecordingActive = false;
    }

    /**
     * Clear buffered audio chunks
     */
    clearBuffer() {
        this.audioChunks = [];
        console.log('Audio buffer cleared');
    }

    /**
     * Stop recording
     */
    stopRecording() {
        if (!this.isRecording) {
            console.warn('Not currently recording');
            return;
        }

        this.isRecording = false;
        this.isPaused = false;

        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
        }

        this.stopVADMonitoring();

        console.log('Recording stopped');
    }

    /**
     * Pause recording (temporarily) - stops current recording without sending
     */
    pauseRecording() {
        if (!this.isRecording) {
            console.warn('Not currently recording');
            return;
        }

        this.isPaused = true;

        // Stop MediaRecorder if recording (will discard audio because isPaused=true)
        // In PCM mode, mediaRecorder is null - skip this step
        if (this.isRecordingActive && this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
            console.log('Recording stopped due to pause (audio discarded)');
        }

        // Clear speech state
        if (this.silenceTimer) {
            clearTimeout(this.silenceTimer);
            this.silenceTimer = null;
        }
        this.lastSpeechTime = null;
        this.isRecordingActive = false;
    }

    /**
     * Resume recording
     */
    resumeRecording() {
        if (!this.isRecording) {
            console.warn('Not currently recording');
            return;
        }

        this.isPaused = false;
        console.log('Recording resumed - ready for next speech');
    }

    /**
     * Start Voice Activity Detection monitoring with noise filtering
     */
    startVADMonitoring() {
        const bufferLength = this.analyser.frequencyBinCount;
        const timeDataArray = new Uint8Array(bufferLength);
        const frequencyDataArray = new Uint8Array(bufferLength);

        const checkAudio = () => {
            if (!this.isRecording) return;

            // Get audio data (both time and frequency domain)
            this.analyser.getByteTimeDomainData(timeDataArray);
            this.analyser.getByteFrequencyData(frequencyDataArray);

            // Calculate RMS energy level
            let sum = 0;
            for (let i = 0; i < bufferLength; i++) {
                const normalized = (timeDataArray[i] - 128) / 128;
                sum += normalized * normalized;
            }
            const rms = Math.sqrt(sum / bufferLength);

            // Calculate speech frequency energy (300Hz - 3000Hz range for human voice)
            // Frequency bins: bin = (frequency * bufferLength) / sampleRate
            const sampleRate = this.audioContext.sampleRate;
            const binWidth = sampleRate / (2 * bufferLength);
            const speechStartBin = Math.floor(200 / binWidth);  // Expanded lower range (was 300Hz)
            const speechEndBin = Math.floor(4000 / binWidth);   // Expanded upper range (was 3000Hz)

            let speechEnergy = 0;
            let speechBinCount = 0;
            for (let i = speechStartBin; i < speechEndBin && i < bufferLength; i++) {
                speechEnergy += frequencyDataArray[i] / 255.0; // Normalize to 0-1
                speechBinCount++;
            }
            speechEnergy = speechEnergy / speechBinCount;

            // Calibrate noise floor during first few seconds
            if (this.isCalibrating) {
                this.noiseCalibrationSamples.push(rms);

                if (this.noiseCalibrationSamples.length >= this.maxNoiseCalibrationSamples) {
                    // Calculate average noise floor
                    const avgNoise = this.noiseCalibrationSamples.reduce((a, b) => a + b, 0) / this.noiseCalibrationSamples.length;
                    this.noiseFloor = avgNoise * 1.5; // Set noise floor slightly above average
                    this.energyThreshold = this.noiseFloor * this.noiseGate; // Speech must be 3x louder
                    this.isCalibrating = false;
                    console.log(`Noise calibration complete. Noise floor: ${this.noiseFloor.toFixed(4)}, Threshold: ${this.energyThreshold.toFixed(4)}`);

                    // Notify that calibration is complete
                    this.onCalibrationComplete();
                }
            }

            // Notify volume change (for visualization)
            this.onVolumeChange(rms);

            // Advanced speech detection: require both energy AND speech frequency content
            const isSpeechEnergy = rms > this.energyThreshold;
            const hasSpeechFrequencies = speechEnergy > 0.08; // More sensitive (was 0.1)
            const isSpeech = isSpeechEnergy && hasSpeechFrequencies;

            // Check if speech is detected
            if (isSpeech && !this.isPaused) {
                const now = Date.now();

                // Track when potential speech started
                if (this.speechStartTime === null) {
                    this.speechStartTime = now;
                }

                // Only trigger speech detection after minimum duration
                const speechDuration = now - this.speechStartTime;
                if (speechDuration >= this.minSpeechDuration && this.lastSpeechTime === null) {
                    // Confirmed speech - start recording
                    console.log('Speech confirmed - starting recording');
                    this.startMediaRecorder();
                    this.onSpeechStart();
                    this.lastSpeechTime = now;
                } else if (this.lastSpeechTime !== null) {
                    // Continue existing speech
                    this.lastSpeechTime = now;
                }

                // Clear any existing silence timer
                if (this.silenceTimer) {
                    clearTimeout(this.silenceTimer);
                    this.silenceTimer = null;
                }
            } else {
                // No speech detected - reset speech start time if we haven't confirmed yet
                if (this.lastSpeechTime === null) {
                    this.speechStartTime = null;
                }

                // Silence detected - start timer if we were recording
                if (this.lastSpeechTime !== null && !this.silenceTimer) {
                    // Start silence timer
                    this.silenceTimer = setTimeout(() => {
                        // Silence threshold exceeded - stop recording and send
                        console.log('Silence detected - stopping recording');
                        this.stopMediaRecorder();
                        // Note: audio will be sent in onstop handler
                        this.onSpeechEnd();
                        this.lastSpeechTime = null;
                        this.speechStartTime = null;
                    }, this.silenceThreshold * 1000);
                }
            }

            // Continue monitoring
            requestAnimationFrame(checkAudio);
        };

        checkAudio();
    }

    /**
     * Stop VAD monitoring
     */
    stopVADMonitoring() {
        if (this.silenceTimer) {
            clearTimeout(this.silenceTimer);
            this.silenceTimer = null;
        }
        this.lastSpeechTime = null;
    }

    /**
     * Process single audio chunk and send immediately
     */
    processAudioChunk(audioBlob) {
        // Convert to Base64
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64Audio = reader.result.split(',')[1]; // Remove data:audio/webm;base64, prefix

            // Determine format from MIME type
            const format = this.mimeType.includes('webm') ? 'webm' :
                this.mimeType.includes('ogg') ? 'ogg' :
                    this.mimeType.includes('mp4') ? 'mp4' : 'webm';

            console.log(`Sending audio chunk: ${audioBlob.size} bytes, format: ${format}`);

            // Send to callback immediately
            this.onAudioData(base64Audio, format);
        };

        reader.readAsDataURL(audioBlob);
    }

    /**
     * Process recorded audio and send to callback
     */
    processRecordedAudio() {
        if (this.audioChunks.length === 0) {
            console.warn('No audio chunks to process');
            return;
        }

        // Combine chunks into a blob
        const audioBlob = new Blob(this.audioChunks, { type: this.mimeType });

        // Convert to Base64
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64Audio = reader.result.split(',')[1]; // Remove data:audio/webm;base64, prefix

            // Determine format from MIME type
            const format = this.mimeType.includes('webm') ? 'webm' :
                this.mimeType.includes('ogg') ? 'ogg' :
                    this.mimeType.includes('mp4') ? 'mp4' : 'webm';

            // Send to callback
            this.onAudioData(base64Audio, format);
        };

        reader.readAsDataURL(audioBlob);

        // Clear chunks
        this.audioChunks = [];
    }

    /**
     * Update silence threshold
     */
    setSilenceThreshold(seconds) {
        this.silenceThreshold = seconds;
        console.log(`Silence threshold updated to: ${seconds}s`);
    }

    /**
     * Update energy threshold
     */
    setEnergyThreshold(level) {
        this.energyThreshold = level;
        console.log(`Energy threshold updated to: ${level}`);
    }

    /**
     * Initialize PCM16 streaming for Realtime API (24kHz, mono, 16-bit)
     */
    async initializePCMMode() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: 24000 // Realtime API sample rate
                }
            });

            console.log('PCM mode: Microphone access granted');

            // Create audio context at 24kHz if possible, otherwise we'll resample
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 24000
            });

            console.log(`PCM mode: AudioContext sample rate: ${this.audioContext.sampleRate}Hz`);

            // Create analyser for VAD
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 2048;
            this.analyser.smoothingTimeConstant = 0.8;

            this.microphone = this.audioContext.createMediaStreamSource(stream);
            this.microphone.connect(this.analyser);

            // Use ScriptProcessorNode for raw PCM access (deprecated but widely supported)
            this.scriptProcessor = this.audioContext.createScriptProcessor(
                this.pcmBufferSize, 1, 1
            );

            this.scriptProcessor.onaudioprocess = (event) => {
                if (!this.isRecording || this.isPaused) return;

                const inputData = event.inputBuffer.getChannelData(0);

                // Resample if needed (browser may not support 24kHz)
                let pcmData;
                if (this.audioContext.sampleRate !== this.targetSampleRate) {
                    pcmData = this.resampleAudio(inputData, this.audioContext.sampleRate, this.targetSampleRate);
                } else {
                    pcmData = inputData;
                }

                // Convert float32 to int16 PCM
                const pcm16 = this.floatTo16BitPCM(pcmData);

                // Convert to base64 and send
                const base64 = this.arrayBufferToBase64(pcm16.buffer);
                this.onPCMData(base64);
            };

            this.microphone.connect(this.scriptProcessor);
            this.scriptProcessor.connect(this.audioContext.destination);

            // Store stream for cleanup
            this._pcmStream = stream;

            console.log('PCM mode initialized successfully');
            return true;

        } catch (error) {
            console.error('Failed to initialize PCM mode:', error);
            this.onError(error);
            throw error;
        }
    }

    /**
     * Start PCM16 streaming
     */
    startPCMStreaming() {
        if (!this.scriptProcessor) {
            throw new Error('PCM mode not initialized. Call initializePCMMode() first.');
        }

        this.isRecording = true;
        this.isPaused = false;

        // Start VAD monitoring
        this.startVADMonitoring();

        console.log('PCM streaming started');
    }

    /**
     * Stop PCM16 streaming
     */
    stopPCMStreaming() {
        this.isRecording = false;
        this.stopVADMonitoring();
        console.log('PCM streaming stopped');
    }

    /**
     * Convert Float32Array to Int16Array (PCM16)
     */
    floatTo16BitPCM(float32Array) {
        const int16Array = new Int16Array(float32Array.length);
        for (let i = 0; i < float32Array.length; i++) {
            // Clamp and convert
            const s = Math.max(-1, Math.min(1, float32Array[i]));
            int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        return int16Array;
    }

    /**
     * Resample audio from source to target sample rate
     */
    resampleAudio(audioData, sourceSampleRate, targetSampleRate) {
        if (sourceSampleRate === targetSampleRate) {
            return audioData;
        }

        const ratio = sourceSampleRate / targetSampleRate;
        const newLength = Math.round(audioData.length / ratio);
        const result = new Float32Array(newLength);

        for (let i = 0; i < newLength; i++) {
            const srcIndex = i * ratio;
            const srcIndexFloor = Math.floor(srcIndex);
            const srcIndexCeil = Math.min(srcIndexFloor + 1, audioData.length - 1);
            const t = srcIndex - srcIndexFloor;

            // Linear interpolation
            result[i] = audioData[srcIndexFloor] * (1 - t) + audioData[srcIndexCeil] * t;
        }

        return result;
    }

    /**
     * Convert ArrayBuffer to Base64
     */
    arrayBufferToBase64(buffer) {
        const bytes = new Uint8Array(buffer);
        let binary = '';
        for (let i = 0; i < bytes.length; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }

    /**
     * Clean up resources
     */
    cleanup() {
        this.stopRecording();

        if (this.mediaRecorder && this.mediaRecorder.stream) {
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }

        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }

        console.log('Audio recorder cleaned up');
    }

    /**
     * Get current recording state
     */
    getState() {
        return {
            isRecording: this.isRecording,
            isPaused: this.isPaused,
            isInitialized: this.mediaRecorder !== null,
            mimeType: this.mimeType
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AudioRecorder;
}
