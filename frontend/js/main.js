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
let authToken = localStorage.getItem('authToken') || null;

// UI Elements
let btnStartStop, btnClear, btnCloseSummary, btnInterrupt;
let btnCall, btnHangup, phoneInput, callStatus, callStatusText;
let statusIndicator, statusDot, statusText;
let transcript, summaryPanel, summaryContent;
let volumeIndicator, loadingOverlay, errorToast, errorMessage;
let voiceSelect, vadThreshold, vadThresholdValue;
let currentCallSid = null;

/**
 * Authenticated fetch wrapper — adds Authorization header automatically
 */
function authFetch(url, options = {}) {
    if (!options.headers) options.headers = {};
    if (authToken) {
        options.headers['Authorization'] = 'Bearer ' + authToken;
    }
    return fetch(url, options);
}

/**
 * Initialize the application
 */
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Initializing AI Voice Agent...');

    // Set up login form handler
    setupLoginHandler();

    // Check if already authenticated
    if (authToken) {
        try {
            const res = await authFetch('/api/auth/check');
            if (res.ok) {
                showApp();
            } else {
                authToken = null;
                localStorage.removeItem('authToken');
                showLogin();
                return;
            }
        } catch (e) {
            showLogin();
            return;
        }
    } else {
        showLogin();
        return;
    }

    // Initialize app after auth
    initApp();
});

/**
 * Show login overlay, hide app
 */
function showLogin() {
    document.getElementById('loginOverlay').classList.remove('hidden');
    document.getElementById('appContainer').classList.add('hidden');
}

/**
 * Show app, hide login overlay
 */
function showApp() {
    document.getElementById('loginOverlay').classList.add('hidden');
    document.getElementById('appContainer').classList.remove('hidden');
}

/**
 * Set up login form submission
 */
function setupLoginHandler() {
    const form = document.getElementById('loginForm');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const username = document.getElementById('loginUsername').value.trim();
        const password = document.getElementById('loginPassword').value;
        const errorDiv = document.getElementById('loginError');
        const btnLogin = document.getElementById('btnLogin');

        if (!username || !password) return;

        btnLogin.disabled = true;
        btnLogin.textContent = 'Signing in...';
        errorDiv.classList.add('hidden');

        try {
            const res = await fetch('/api/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });
            const data = await res.json();

            if (data.success && data.token) {
                authToken = data.token;
                localStorage.setItem('authToken', authToken);
                showApp();
                initApp();
            } else {
                errorDiv.textContent = data.error || 'Invalid username or password';
                errorDiv.classList.remove('hidden');
            }
        } catch (err) {
            errorDiv.textContent = 'Connection error. Please try again.';
            errorDiv.classList.remove('hidden');
        } finally {
            btnLogin.disabled = false;
            btnLogin.textContent = 'Sign In';
        }
    });
}

/**
 * Initialize the main app (called after successful auth)
 */
let _appInitialized = false;
function initApp() {
    // Prevent duplicate initialization (e.g. re-login without page reload)
    if (_appInitialized) {
        // Clean up old instances
        if (wsClient) { wsClient.close(); }
        if (audioPlayer) { audioPlayer.cleanup(); }
    }

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

    // Set up UI event handlers (only once)
    if (!_appInitialized) {
        setupUIHandlers();
    }

    _appInitialized = true;
    console.log('Application initialized');
}

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

    // --- Agent Config: Questions add/remove ---
    document.getElementById('btnAddQuestion').addEventListener('click', addQuestion);
    document.getElementById('questionsList').addEventListener('click', (e) => {
        if (e.target.classList.contains('btn-remove-q')) {
            const item = e.target.closest('.question-item');
            if (item && document.querySelectorAll('.question-item').length > 1) {
                item.remove();
            }
        }
    });

    // --- Script save / activate / deactivate ---
    document.getElementById('btnSaveScript').addEventListener('click', saveScript);
    document.getElementById('btnActivateScript').addEventListener('click', activateScript);
    document.getElementById('btnDeactivateScript').addEventListener('click', deactivateScript);

    // --- Logout ---
    document.getElementById('btnLogout').addEventListener('click', async () => {
        try {
            await authFetch('/api/logout', { method: 'POST' });
        } catch (e) { /* ignore */ }
        authToken = null;
        localStorage.removeItem('authToken');
        showLogin();
    });

    // --- API Settings ---
    document.getElementById('btnSaveSettings').addEventListener('click', saveApiSettings);
    // Load settings when the panel is opened
    document.getElementById('apiSettingsPanel').addEventListener('toggle', (e) => {
        if (e.target.open) loadApiSettings();
    });

    // Load saved scripts and check active status on page load
    loadSavedScripts();
    checkScriptStatus();
}

/**
 * Toggle password field visibility
 */
function toggleKeyVisibility(btn) {
    const input = btn.parentElement.querySelector('input');
    if (input.type === 'password') {
        input.type = 'text';
        btn.textContent = 'Hide';
    } else {
        input.type = 'password';
        btn.textContent = 'Show';
    }
}

/**
 * Load API settings from server into the form
 */
async function loadApiSettings() {
    try {
        const res = await authFetch('/api/settings');
        if (!res.ok) return;
        const data = await res.json();
        const settings = data.settings || {};

        // Fill each input that has a data-key attribute
        document.querySelectorAll('.api-key-input[data-key]').forEach(input => {
            const key = input.dataset.key;
            if (settings[key]) {
                input.value = settings[key].value || '';
                input.placeholder = settings[key].masked ? 'Enter new value to update' : '';
            }
        });
    } catch (e) {
        console.error('Failed to load settings:', e);
    }
}

/**
 * Save API settings to server
 */
async function saveApiSettings() {
    const btn = document.getElementById('btnSaveSettings');
    const statusEl = document.getElementById('saveStatus');
    const updates = {};

    // Collect all non-empty inputs
    document.querySelectorAll('.api-key-input[data-key]').forEach(input => {
        const key = input.dataset.key;
        const val = input.value.trim();
        // Only send if value is present and not a masked placeholder
        if (val && !val.includes('****')) {
            updates[key] = val;
        }
    });

    if (Object.keys(updates).length === 0) {
        statusEl.textContent = 'No changes to save';
        statusEl.className = 'save-status error';
        statusEl.classList.remove('hidden');
        setTimeout(() => statusEl.classList.add('hidden'), 3000);
        return;
    }

    btn.disabled = true;
    btn.textContent = 'Saving...';
    statusEl.classList.add('hidden');

    try {
        const res = await authFetch('/api/settings', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ settings: updates })
        });
        const data = await res.json();

        if (data.success) {
            statusEl.textContent = `Saved! Updated: ${data.updated.join(', ')}`;
            statusEl.className = 'save-status';
            statusEl.classList.remove('hidden');
            // Reload to show masked values
            await loadApiSettings();
        } else {
            statusEl.textContent = data.error || 'Save failed';
            statusEl.className = 'save-status error';
            statusEl.classList.remove('hidden');
        }
    } catch (e) {
        statusEl.textContent = 'Connection error';
        statusEl.className = 'save-status error';
        statusEl.classList.remove('hidden');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Save All Changes';
        setTimeout(() => statusEl.classList.add('hidden'), 5000);
    }
}

/**
 * Add a new blank question input
 */
function addQuestion() {
    const list = document.getElementById('questionsList');
    const item = document.createElement('div');
    item.className = 'question-item';
    item.innerHTML = `
        <input type="text" class="question-input" placeholder="Enter a question..." />
        <button class="btn-remove-q" title="Remove">&times;</button>
    `;
    list.appendChild(item);
    item.querySelector('input').focus();
}

/**
 * Collect script config from the UI
 */
function getScriptConfig() {
    const welcome = document.getElementById('welcomeMessage').value.trim();
    const goal = document.getElementById('agentGoal').value.trim();
    const behaviour = document.getElementById('agentBehaviour').value.trim();

    const questions = [];
    document.querySelectorAll('.question-input').forEach(input => {
        const q = input.value.trim();
        if (q) questions.push(q);
    });

    return { welcome, questions, goal, behaviour };
}

/**
 * Activate the current script on the server
 */
async function activateScript() {
    const script = getScriptConfig();
    if (!script.welcome && script.questions.length === 0) {
        showError('Please add a welcome message or at least one question');
        return;
    }
    try {
        const res = await authFetch('/api/script/activate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(script)
        });
        const data = await res.json();
        if (data.success) {
            updateScriptStatusUI(true);
        }
    } catch (e) {
        showError('Failed to activate script');
    }
}

/**
 * Deactivate the script on the server
 */
async function deactivateScript() {
    try {
        const res = await authFetch('/api/script/deactivate', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
            updateScriptStatusUI(false);
        }
    } catch (e) {
        showError('Failed to deactivate script');
    }
}

/**
 * Check script status on page load
 */
async function checkScriptStatus() {
    try {
        const res = await authFetch('/api/script/status');
        const data = await res.json();
        updateScriptStatusUI(data.active);
        if (data.active && data.script) {
            _activeScriptId = data.script.id || null;
        }
    } catch (e) {
        // ignore
    }
}

/**
 * Update the script status UI
 */
function updateScriptStatusUI(active) {
    const bar = document.getElementById('scriptStatusBar');
    const label = document.getElementById('scriptStatusLabel');
    const btnActivate = document.getElementById('btnActivateScript');
    const btnDeactivate = document.getElementById('btnDeactivateScript');

    if (active) {
        bar.classList.add('active');
        label.textContent = 'Script is ACTIVE — calls will use this script';
        btnActivate.classList.add('hidden');
        btnDeactivate.classList.remove('hidden');
    } else {
        bar.classList.remove('active');
        label.textContent = 'No script active — using default prompt';
        btnActivate.classList.remove('hidden');
        btnDeactivate.classList.add('hidden');
    }
}

// ==========================================
// Saved Scripts Management
// ==========================================

let _activeScriptId = null;

/**
 * Save current script to server
 */
async function saveScript() {
    const script = getScriptConfig();
    if (!script.welcome && script.questions.length === 0) {
        showError('Add a welcome message or questions before saving');
        return;
    }
    const name = prompt('Script name:', script.welcome.substring(0, 40) || 'My Script');
    if (!name) return;

    try {
        const res = await authFetch('/api/scripts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ...script, name })
        });
        const data = await res.json();
        if (data.success) {
            loadSavedScripts();
        }
    } catch (e) {
        showError('Failed to save script');
    }
}

/**
 * Load all saved scripts from server
 */
async function loadSavedScripts() {
    try {
        const res = await authFetch('/api/scripts');
        const scripts = await res.json();
        renderSavedScripts(scripts);
    } catch (e) {
        // ignore
    }
}

/**
 * Render saved scripts list
 */
function renderSavedScripts(scripts) {
    const list = document.getElementById('savedScriptsList');
    const msg = document.getElementById('noScriptsMsg');

    if (!scripts || scripts.length === 0) {
        list.innerHTML = '<p class="no-scripts-msg" id="noScriptsMsg">No saved scripts yet. Create one above and click Save.</p>';
        return;
    }

    list.innerHTML = scripts.map(s => `
        <div class="saved-script-card ${s.id === _activeScriptId ? 'active-card' : ''}" data-id="${s.id}">
            <div class="saved-script-info" onclick="loadScriptToEditor('${s.id}')">
                <div class="saved-script-name">${escapeHtml(s.name)}</div>
                <div class="saved-script-meta">${s.questions ? s.questions.length : 0} questions</div>
            </div>
            <div class="saved-script-actions">
                <button class="btn-load-script" onclick="loadScriptToEditor('${s.id}')">Load</button>
                <button class="btn-use-script" onclick="activateSavedScript('${s.id}')">Use</button>
                <button class="btn-delete-script" onclick="deleteSavedScript('${s.id}')">Delete</button>
            </div>
        </div>
    `).join('');
}

/**
 * Load a saved script into the editor fields
 */
async function loadScriptToEditor(scriptId) {
    try {
        const res = await authFetch('/api/scripts');
        const scripts = await res.json();
        const s = scripts.find(x => x.id === scriptId);
        if (!s) return;

        document.getElementById('welcomeMessage').value = s.welcome || '';
        document.getElementById('agentGoal').value = s.goal || '';
        document.getElementById('agentBehaviour').value = s.behaviour || '';

        // Rebuild questions list
        const qList = document.getElementById('questionsList');
        qList.innerHTML = '';
        const questions = s.questions && s.questions.length > 0 ? s.questions : [''];
        questions.forEach(q => {
            const item = document.createElement('div');
            item.className = 'question-item';
            item.innerHTML = `
                <input type="text" class="question-input" value="${escapeHtml(q)}" />
                <button class="btn-remove-q" title="Remove">&times;</button>
            `;
            qList.appendChild(item);
        });
    } catch (e) {
        showError('Failed to load script');
    }
}

/**
 * Activate a saved script directly (load + activate in one click)
 */
async function activateSavedScript(scriptId) {
    try {
        const res = await authFetch('/api/scripts');
        const scripts = await res.json();
        const s = scripts.find(x => x.id === scriptId);
        if (!s) return;

        const payload = {
            id: s.id,
            welcome: s.welcome,
            questions: s.questions,
            goal: s.goal,
            behaviour: s.behaviour
        };

        const actRes = await authFetch('/api/script/activate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await actRes.json();
        if (data.success) {
            _activeScriptId = scriptId;
            updateScriptStatusUI(true);
            loadSavedScripts();
            // Also load into editor
            await loadScriptToEditor(scriptId);
        }
    } catch (e) {
        showError('Failed to activate script');
    }
}

/**
 * Delete a saved script
 */
async function deleteSavedScript(scriptId) {
    if (!confirm('Delete this script?')) return;
    try {
        await authFetch(`/api/scripts/${scriptId}`, { method: 'DELETE' });
        if (_activeScriptId === scriptId) {
            _activeScriptId = null;
            updateScriptStatusUI(false);
        }
        loadSavedScripts();
    } catch (e) {
        showError('Failed to delete script');
    }
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text || '';
    return div.innerHTML;
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

        // Send end conversation signal, then close the WebSocket
        if (wsClient && wsClient.isConnected) {
            try {
                wsClient.endConversation();
            } catch (e) {
                // Ignore send errors on already-closing socket
            }
            // Close the WebSocket to prevent ghost connections and reconnect loops
            wsClient.reconnectAttempts = wsClient.maxReconnectAttempts; // prevent auto-reconnect
            wsClient.close();
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
                <p>${escapeHtml(summary.overview)}</p>
            </div>
        `;
    }

    // Key Points
    if (summary.key_points && summary.key_points.length > 0) {
        summaryHTML += `
            <div class="summary-section">
                <h3>Key Points</h3>
                <ul>
                    ${summary.key_points.map(point => `<li>${escapeHtml(point)}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    // Topics
    if (summary.topics && summary.topics.length > 0) {
        summaryHTML += `
            <div class="summary-section">
                <h3>Topics Discussed</h3>
                <p>${summary.topics.map(t => escapeHtml(t)).join(', ')}</p>
            </div>
        `;
    }

    // Action Items
    if (summary.action_items && summary.action_items.length > 0) {
        summaryHTML += `
            <div class="summary-section">
                <h3>Action Items</h3>
                <ul>
                    ${summary.action_items.map(item => `<li>${escapeHtml(item)}</li>`).join('')}
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
                <p>Sentiment: ${escapeHtml(summary.sentiment || 'Neutral')}</p>
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

        const response = await authFetch('/twilio/outbound-call', {
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

        const response = await authFetch('/twilio/hangup', {
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
