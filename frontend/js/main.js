/**
 * Main Application Controller
 * Coordinates WebSocket, Audio Recorder, and Audio Player
 */

// Global state
let wsClient = null;
let audioRecorder = null;
let audioPlayer = null;
let isConversationActive = false;
let isAISpeaking = false;  // Track if AI is currently speaking
let useRealtimeAPI = true;  // Enable Realtime API for ultra-low latency (default: true)

// UI Elements
let btnStartStop, btnClear, btnCloseSummary, btnInterrupt;
let btnCall, btnHangup, phoneInput, callStatus, callStatusText;
let statusIndicator, statusDot, statusText;
let transcript, summaryPanel, summaryContent;
let volumeIndicator, loadingOverlay, errorToast, errorMessage;
let voiceSelect, vadThreshold, vadThresholdValue;
let currentCallSid = null;

/**
 * Initialize the application
 */
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Initializing AI Voice Agent...');

    // Get UI elements
    initializeUIElements();

    // Initialize services
    wsClient = new WebSocketClient();
    audioPlayer = new AudioPlayer({
        onPlaybackStart: handlePlaybackStart,
        onPlaybackEnd: handlePlaybackEnd,
        onError: handleError
    });

    // Set up WebSocket message handlers
    setupWebSocketHandlers();

    // Set up UI event handlers
    setupUIHandlers();

    console.log('Application initialized');
});

/**
 * Initialize UI element references
 */
function initializeUIElements() {
    btnStartStop = document.getElementById('btnStartStop');
    btnClear = document.getElementById('btnClear');
    btnCloseSummary = document.getElementById('btnCloseSummary');
    btnInterrupt = document.getElementById('btnInterrupt');

    statusIndicator = document.getElementById('statusIndicator');
    statusDot = document.getElementById('statusDot');
    statusText = document.getElementById('statusText');

    transcript = document.getElementById('transcript');
    summaryPanel = document.getElementById('summaryPanel');
    summaryContent = document.getElementById('summaryContent');

    volumeIndicator = document.getElementById('volumeIndicator');
    loadingOverlay = document.getElementById('loadingOverlay');
    errorToast = document.getElementById('errorToast');
    errorMessage = document.getElementById('errorMessage');

    voiceSelect = document.getElementById('voiceSelect');
    vadThreshold = document.getElementById('vadThreshold');
    vadThresholdValue = document.getElementById('vadThresholdValue');

    // Dialer elements
    btnCall = document.getElementById('btnCall');
    btnHangup = document.getElementById('btnHangup');
    phoneInput = document.getElementById('phoneInput');
    callStatus = document.getElementById('callStatus');
    callStatusText = document.getElementById('callStatusText');
}

/**
 * Set up WebSocket message handlers
 */
function setupWebSocketHandlers() {
    wsClient.on('connected', handleConnected);
    wsClient.on('transcription', handleTranscription);
    wsClient.on('response_text', handleResponseText);
    wsClient.on('response_audio', handleResponseAudio);
    wsClient.on('response_complete', handleResponseComplete);
    wsClient.on('interrupted', handleInterrupted);
    wsClient.on('summary', handleSummary);
    wsClient.on('error', handleServerError);
    wsClient.on('disconnected', handleDisconnected);
    wsClient.on('reconnect_failed', handleReconnectFailed);
}

/**
 * Set up UI event handlers
 */
function setupUIHandlers() {
    // Start/Stop button
    btnStartStop.addEventListener('click', toggleConversation);

    // Clear button
    btnClear.addEventListener('click', clearTranscript);

    // Interrupt button
    btnInterrupt.addEventListener('click', interruptAI);

    // Close summary button
    btnCloseSummary.addEventListener('click', () => {
        summaryPanel.classList.add('hidden');
    });

    // VAD threshold slider
    vadThreshold.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        vadThresholdValue.textContent = `${value}s`;
        if (audioRecorder) {
            audioRecorder.setSilenceThreshold(value);
        }
    });

    // Phone dialer
    btnCall.addEventListener('click', makeOutboundCall);
    btnHangup.addEventListener('click', hangupCall);

    // Allow Enter key to dial
    phoneInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            makeOutboundCall();
        }
    });

    // ElevenLabs toggle hint text
    const toggleElevenLabs = document.getElementById('toggleElevenLabs');
    const voiceHint = document.getElementById('voiceHint');
    if (toggleElevenLabs) {
        toggleElevenLabs.addEventListener('change', () => {
            voiceHint.textContent = toggleElevenLabs.checked
                ? '(Using ElevenLabs voice)'
                : '(Using OpenAI built-in voice)';
        });
    }
}

/**
 * Toggle conversation on/off
 */
async function toggleConversation() {
    if (!isConversationActive) {
        await startConversation();
    } else {
        await stopConversation();
    }
}

/**
 * Start a new conversation
 */
async function startConversation() {
    try {
        showLoading('Connecting...');

        // Get selected voice
        const selectedVoice = voiceSelect ? voiceSelect.value : 'alloy';
        wsClient.setVoice(selectedVoice);

        // Connect to WebSocket (use Realtime API for ultra-low latency)
        await wsClient.connect(useRealtimeAPI);

        // Initialize audio recorder with appropriate mode
        if (useRealtimeAPI) {
            // Realtime API mode - use PCM16 streaming
            audioRecorder = new AudioRecorder({
                pcmMode: true,
                silenceThreshold: parseFloat(vadThreshold.value) || 0.8,
                energyThreshold: 0.015,
                onPCMData: handlePCMData,  // PCM audio callback
                onSpeechStart: handleSpeechStart,
                onSpeechEnd: handleSpeechEnd,
                onVolumeChange: handleVolumeChange,
                onCalibrationComplete: handleCalibrationComplete,
                onError: handleError
            });
            await audioRecorder.initializePCMMode();
            audioRecorder.startPCMStreaming();
            console.log('Using Realtime API (ultra-low latency mode)');
        } else {
            // Standard mode - use MediaRecorder
            audioRecorder = new AudioRecorder({
                silenceThreshold: parseFloat(vadThreshold.value) || 0.8,
                energyThreshold: 0.015,
                onAudioData: handleAudioData,
                onSpeechStart: handleSpeechStart,
                onSpeechEnd: handleSpeechEnd,
                onVolumeChange: handleVolumeChange,
                onCalibrationComplete: handleCalibrationComplete,
                onError: handleError
            });
            await audioRecorder.initialize();
            audioRecorder.startRecording();
            console.log('Using standard voice endpoint');
        }

        // Update UI
        isConversationActive = true;
        updateUIForActiveConversation();

        hideLoading();
        const modeLabel = useRealtimeAPI ? 'Realtime API' : 'Standard';
        updateStatus('connected', `Calibrating... (${modeLabel})`);

        console.log('Conversation started');

    } catch (error) {
        console.error('Failed to start conversation:', error);
        hideLoading();
        showError('Failed to start conversation. Please check your microphone permissions.');
        isConversationActive = false;
    }
}

/**
 * Stop the conversation
 */
async function stopConversation() {
    try {
        // Stop recording
        if (audioRecorder) {
            audioRecorder.stopRecording();
            audioRecorder.cleanup();
            audioRecorder = null;
        }

        // Send end conversation signal
        if (wsClient && wsClient.isConnected) {
            wsClient.endConversation();
        }

        // Update UI
        isConversationActive = false;
        updateUIForInactiveConversation();
        updateStatus('idle', 'Conversation ended');

        console.log('Conversation stopped');

    } catch (error) {
        console.error('Error stopping conversation:', error);
        showError('Error stopping conversation');
    }
}

/**
 * Handle audio data from recorder
 */
function handleAudioData(audioBase64, format) {
    // Block sending if AI is speaking
    if (isAISpeaking) {
        console.log('Skipping audio chunk - AI is speaking');
        return;
    }

    if (wsClient && wsClient.isConnected) {
        wsClient.sendAudioChunk(audioBase64, format);
        console.log('Audio chunk sent to server');
    }
}

/**
 * Handle PCM audio data from recorder (Realtime API mode)
 */
function handlePCMData(audioBase64) {
    // Block sending if AI is speaking
    if (isAISpeaking) {
        return;  // Silent skip for PCM streaming
    }

    if (wsClient && wsClient.isConnected) {
        wsClient.sendPCMAudio(audioBase64);
    }
}

/**
 * Handle speech start
 */
function handleSpeechStart() {
    console.log('Speech started');

    // If AI is speaking, interrupt it
    if (isAISpeaking && audioPlayer) {
        console.log('User interrupted AI - stopping playback');
        audioPlayer.stop();
        audioPlayer.clearQueue();
        isAISpeaking = false;

        // Notify server to stop generating
        if (wsClient && wsClient.isConnected) {
            wsClient.interruptAI();
        }
    }

    updateStatus('recording', 'You are speaking...');
}

/**
 * Handle speech end
 */
function handleSpeechEnd() {
    console.log('Speech ended');
    updateStatus('processing', 'Processing...');
}

/**
 * Handle volume change (for visualization)
 */
function handleVolumeChange(volume) {
    if (volume > 0.02) {
        volumeIndicator.classList.add('active');
    } else {
        volumeIndicator.classList.remove('active');
    }
}

/**
 * Handle calibration complete
 */
function handleCalibrationComplete() {
    console.log('Noise calibration complete - ready for speech');
    if (isConversationActive && !isAISpeaking) {
        updateStatus('connected', 'Listening...');
    }
}

/**
 * Handle WebSocket connected
 */
function handleConnected(message) {
    console.log('Connected to server:', message);
}

/**
 * Handle transcription from server
 */
function handleTranscription(message) {
    const { text } = message;
    console.log('Transcription received:', text);

    addMessageToTranscript('user', text);
    updateStatus('processing', 'AI is thinking...');
}

/**
 * Handle response text from server
 */
function handleResponseText(message) {
    const { text, sequence } = message;
    console.log('Response text received:', text);

    if (sequence === 0) {
        // First sentence - create new message
        addMessageToTranscript('assistant', text);
    } else {
        // Append to existing message
        appendToLastAssistantMessage(text);
    }
}

/**
 * Handle response audio from server
 */
function handleResponseAudio(message) {
    const { data, sequence, format } = message;
    console.log(`Response audio received (format: ${format || 'mp3'})`);

    // Check if PCM16 format (from Realtime API)
    if (format === 'pcm16') {
        audioPlayer.enqueuePCM(data);
    } else {
        // Standard MP3 format
        audioPlayer.enqueue(data, format || 'mp3');
    }
    updateStatus('speaking', 'AI is speaking...');
}

/**
 * Handle response complete (Realtime API)
 */
function handleResponseComplete() {
    console.log('AI response complete');
}

/**
 * Handle AI interrupted by user
 */
function handleInterrupted(message) {
    console.log('AI was interrupted:', message);
    if (audioPlayer) {
        audioPlayer.stopPCM();
        audioPlayer.stop();
        audioPlayer.clearQueue();
    }
    isAISpeaking = false;
}

/**
 * Handle summary from server
 */
function handleSummary(message) {
    const { summary } = message;
    console.log('Summary received:', summary);

    displaySummary(summary);
}

/**
 * Handle server error
 */
function handleServerError(message) {
    const { code, message: errorMsg } = message;
    console.error('Server error:', code, errorMsg);
    showError(errorMsg || 'Server error occurred');
}

/**
 * Handle disconnection
 */
function handleDisconnected() {
    console.warn('Disconnected from server');
    updateStatus('idle', 'Disconnected - Reconnecting...');
}

/**
 * Handle reconnect failed
 */
function handleReconnectFailed() {
    console.error('Reconnection failed');
    showError('Failed to reconnect to server. Please refresh the page.');
    isConversationActive = false;
    updateUIForInactiveConversation();
}

/**
 * Interrupt AI speech
 */
function interruptAI() {
    console.log('User manually interrupted AI');

    if (audioPlayer) {
        audioPlayer.stop();
        audioPlayer.clearQueue();
    }

    isAISpeaking = false;
    btnInterrupt.classList.add('hidden');

    // Resume recording
    if (audioRecorder && isConversationActive) {
        audioRecorder.resumeRecording();
        updateStatus('connected', 'Listening...');
    }

    // Notify server
    if (wsClient && wsClient.isConnected) {
        wsClient.interruptAI();
    }
}

/**
 * Handle playback start
 */
function handlePlaybackStart() {
    isAISpeaking = true;  // Set flag to block audio sending

    // Show interrupt button
    btnInterrupt.classList.remove('hidden');

    // Pause recording to avoid feedback loop
    if (audioRecorder) {
        audioRecorder.pauseRecording();
    }
    updateStatus('speaking', 'AI is speaking...');
    console.log('AI started speaking - recording paused');
}

/**
 * Handle playback end
 */
function handlePlaybackEnd() {
    isAISpeaking = false;  // Clear flag to allow audio sending

    // Hide interrupt button
    btnInterrupt.classList.add('hidden');

    // Resume recording after AI finishes speaking
    if (audioRecorder && isConversationActive) {
        audioRecorder.resumeRecording();
        updateStatus('connected', 'Listening...');
        console.log('AI finished speaking - recording resumed');
    }
}

/**
 * Handle errors
 */
function handleError(error) {
    console.error('Error:', error);
    showError(error.message || 'An error occurred');
}

/**
 * Add message to transcript
 */
function addMessageToTranscript(role, text) {
    // Remove welcome message if present
    const welcomeMsg = transcript.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${role}`;

    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'message-bubble';
    bubbleDiv.textContent = text;

    messageDiv.appendChild(bubbleDiv);
    transcript.appendChild(messageDiv);

    // Scroll to bottom
    transcript.scrollTop = transcript.scrollHeight;
}

/**
 * Append text to last assistant message
 */
function appendToLastAssistantMessage(text) {
    const messages = transcript.querySelectorAll('.message-assistant');
    if (messages.length > 0) {
        const lastMessage = messages[messages.length - 1];
        const bubble = lastMessage.querySelector('.message-bubble');
        bubble.textContent += ' ' + text;

        // Scroll to bottom
        transcript.scrollTop = transcript.scrollHeight;
    }
}

/**
 * Clear transcript
 */
function clearTranscript() {
    transcript.innerHTML = `
        <div class="welcome-message">
            <p>Click "Start Conversation" to begin speaking with the AI.</p>
            <p class="help-text">Make sure your microphone is ready!</p>
        </div>
    `;
}

/**
 * Display summary
 */
function displaySummary(summary) {
    let summaryHTML = '';

    // Overview
    if (summary.overview) {
        summaryHTML += `
            <div class="summary-section">
                <h3>Overview</h3>
                <p>${summary.overview}</p>
            </div>
        `;
    }

    // Key Points
    if (summary.key_points && summary.key_points.length > 0) {
        summaryHTML += `
            <div class="summary-section">
                <h3>Key Points</h3>
                <ul>
                    ${summary.key_points.map(point => `<li>${point}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    // Topics
    if (summary.topics && summary.topics.length > 0) {
        summaryHTML += `
            <div class="summary-section">
                <h3>Topics Discussed</h3>
                <p>${summary.topics.join(', ')}</p>
            </div>
        `;
    }

    // Action Items
    if (summary.action_items && summary.action_items.length > 0) {
        summaryHTML += `
            <div class="summary-section">
                <h3>Action Items</h3>
                <ul>
                    ${summary.action_items.map(item => `<li>${item}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    // Metadata
    if (summary.metadata) {
        summaryHTML += `
            <div class="summary-section">
                <h3>Conversation Details</h3>
                <p>Duration: ${Math.floor(summary.metadata.duration_seconds / 60)}m ${Math.floor(summary.metadata.duration_seconds % 60)}s</p>
                <p>Exchanges: ${summary.metadata.turn_count}</p>
                <p>Sentiment: ${summary.sentiment || 'Neutral'}</p>
            </div>
        `;
    }

    summaryContent.innerHTML = summaryHTML;
    summaryPanel.classList.remove('hidden');
}

/**
 * Update status indicator
 */
function updateStatus(state, text) {
    statusIndicator.className = `status-indicator ${state}`;
    statusText.textContent = text;
}

/**
 * Update UI for active conversation
 */
function updateUIForActiveConversation() {
    const btnIcon = document.getElementById('btnIcon');
    const btnText = document.getElementById('btnText');

    btnStartStop.classList.add('active');
    btnIcon.textContent = '⏹️';
    btnText.textContent = 'Stop Conversation';
}

/**
 * Update UI for inactive conversation
 */
function updateUIForInactiveConversation() {
    const btnIcon = document.getElementById('btnIcon');
    const btnText = document.getElementById('btnText');

    btnStartStop.classList.remove('active');
    btnIcon.textContent = '🎤';
    btnText.textContent = 'Start Conversation';

    volumeIndicator.classList.remove('active');
}

/**
 * Show loading overlay
 */
function showLoading(text = 'Processing...') {
    const loadingText = document.querySelector('.loading-text');
    if (loadingText) {
        loadingText.textContent = text;
    }
    loadingOverlay.classList.remove('hidden');
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    loadingOverlay.classList.add('hidden');
}

/**
 * Show error toast
 */
function showError(message) {
    errorMessage.textContent = message;
    errorToast.classList.remove('hidden');

    // Auto-hide after 5 seconds
    setTimeout(() => {
        errorToast.classList.add('hidden');
    }, 5000);
}

/**
 * Make an outbound call via Twilio
 */
async function makeOutboundCall() {
    const phoneNumber = phoneInput.value.trim();

    if (!phoneNumber) {
        showError('Please enter a phone number');
        phoneInput.focus();
        return;
    }

    // Basic phone validation - must start with + and have digits
    const cleaned = phoneNumber.replace(/[\s\-\(\)]/g, '');
    if (!/^\+?\d{7,15}$/.test(cleaned)) {
        showError('Enter a valid phone number with country code (e.g. +1234567890)');
        return;
    }

    const dialNumber = cleaned.startsWith('+') ? cleaned : '+' + cleaned;
    const useElevenLabs = document.getElementById('toggleElevenLabs')?.checked || false;

    try {
        btnCall.disabled = true;
        btnCall.classList.add('hidden');
        btnHangup.classList.remove('hidden');
        callStatus.classList.remove('hidden');
        callStatus.classList.remove('error');
        callStatusText.textContent = `Calling ${dialNumber}...`;

        const response = await fetch('/twilio/outbound-call', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ to: dialNumber, elevenlabs: useElevenLabs })
        });

        const data = await response.json();

        if (data.success) {
            currentCallSid = data.call_sid;
            callStatusText.textContent = `Connected - Call SID: ${data.call_sid.substring(0, 12)}...`;
            addMessageToTranscript('assistant', `Calling ${dialNumber}... AI agent is handling the call.`);
        } else {
            throw new Error(data.error || 'Call failed');
        }

    } catch (error) {
        console.error('Outbound call failed:', error);
        callStatus.classList.add('error');
        callStatusText.textContent = `Call failed: ${error.message}`;
        showError('Failed to make call: ' + error.message);
        resetDialer();
    }
}

/**
 * Hang up current call
 */
async function hangupCall() {
    if (!currentCallSid) {
        resetDialer();
        return;
    }

    try {
        callStatusText.textContent = 'Hanging up...';

        const response = await fetch('/twilio/hangup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ call_sid: currentCallSid })
        });

        addMessageToTranscript('assistant', 'Call ended.');
    } catch (error) {
        console.error('Hangup failed:', error);
    } finally {
        resetDialer();
    }
}

/**
 * Reset dialer UI to idle state
 */
function resetDialer() {
    btnCall.disabled = false;
    btnCall.classList.remove('hidden');
    btnHangup.classList.add('hidden');
    callStatus.classList.add('hidden');
    currentCallSid = null;
}

/**
 * Clean up on page unload
 */
window.addEventListener('beforeunload', () => {
    if (audioRecorder) {
        audioRecorder.cleanup();
    }
    if (audioPlayer) {
        audioPlayer.cleanup();
    }
    if (wsClient) {
        wsClient.close();
    }
});
