document.addEventListener('DOMContentLoaded', () => {
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const clearBtn = document.getElementById('clearBtn');
    const heatmapBtn = document.getElementById('heatmapBtn');
    const logsBtn = document.getElementById('logsBtn');
    const statusDiv = document.getElementById('status');
    const timerDiv = document.getElementById('timer');
    
    // Settings
    const heatmapTypeSelect = document.getElementById('heatmapTypeSelect');
    const gazeCursorToggle = document.getElementById('gazeCursorToggle');
    
    // Status Dot / Connection
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    
    // Counters
    const mouseCounter = document.getElementById('mouseCounter');
    const gazeCounter = document.getElementById('gazeCounter');
    
    // Tabs
    const tabGazeBtn = document.getElementById('tabGazeBtn');
    const tabMouseBtn = document.getElementById('tabMouseBtn');
    const gazeInterestsDiv = document.getElementById('gazeInterests');
    const mouseInterestsDiv = document.getElementById('mouseInterests');
    
    // Gaze Interest elements
    const topGazeInterestHero = document.getElementById('topGazeInterestHero');
    const heroGazeElement = document.getElementById('heroGazeElement');
    const heroGazeTime = document.getElementById('heroGazeTime');
    const gazeInterestsContainer = document.getElementById('gazeInterestsContainer');
    
    // Mouse Interest elements
    const topMouseInterestHero = document.getElementById('topMouseInterestHero');
    const heroMouseElement = document.getElementById('heroMouseElement');
    const heroMouseTime = document.getElementById('heroMouseTime');
    const mouseInterestsContainer = document.getElementById('mouseInterestsContainer');
    
    const logContainer = document.getElementById('logContainer');

    // Tab Switching Logic
    tabGazeBtn.addEventListener('click', () => {
        tabGazeBtn.classList.add('active');
        tabMouseBtn.classList.remove('active');
        gazeInterestsDiv.classList.add('active');
        mouseInterestsDiv.classList.remove('active');
    });

    tabMouseBtn.addEventListener('click', () => {
        tabMouseBtn.classList.add('active');
        tabGazeBtn.classList.remove('active');
        mouseInterestsDiv.classList.add('active');
        gazeInterestsDiv.classList.remove('active');
    });

    // Settings Event Listeners
    heatmapTypeSelect.addEventListener('change', () => {
        chrome.runtime.sendMessage({
            action: "setHeatmapType",
            heatmapType: heatmapTypeSelect.value
        });
    });

    gazeCursorToggle.addEventListener('change', () => {
        chrome.runtime.sendMessage({
            action: "setGazeCursorVisibility",
            visible: gazeCursorToggle.checked
        });
    });

    // Listen for WebSocket connection updates from background script
    chrome.runtime.onMessage.addListener((request) => {
        if (request.action === "connectionUpdate") {
            updateConnectionUI(request.isGazeConnected);
        }
    });

    // Poll status on interval
    setInterval(checkStatus, 1000);
    checkStatus();

    let timerInterval = null;

    function startTimerDisplay(startTs) {
        if (timerInterval) clearInterval(timerInterval);
        updateTimerDisplay(startTs);
        timerInterval = setInterval(() => {
            updateTimerDisplay(startTs);
        }, 1000);
    }

    function stopTimerDisplay() {
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }
    }

    function updateTimerDisplay(startTs, stopTs = null) {
        if (!startTs) return;
        const endTs = stopTs || Date.now();
        const diff = Math.floor((endTs - startTs) / 1000);
        const mins = Math.floor(diff / 60).toString().padStart(2, '0');
        const secs = (diff % 60).toString().padStart(2, '0');
        timerDiv.textContent = `Session Time: ${mins}:${secs}`;
    }

    function checkStatus() {
        chrome.runtime.sendMessage({ action: "getStatus" }, (response) => {
            if (chrome.runtime.lastError) {
                statusDiv.textContent = "Error: Background script offline";
                return;
            }
            if (response) {
                updateUI(response);
                updateConnectionUI(response.isGazeConnected);
                
                // Sync settings state
                if (response.heatmapType) {
                    heatmapTypeSelect.value = response.heatmapType;
                }
                gazeCursorToggle.checked = response.showGazeCursor !== false;

                // Sync data counters
                mouseCounter.textContent = `Mouse: ${response.mouseDataCount || 0} pts`;
                gazeCounter.textContent = `Gaze: ${response.gazeDataCount || 0} pts`;

                // Render interests
                renderGazeInterests(response.topGazeInterests);
                renderMouseInterests(response.topInterests);

                // Sync timer
                if (response.isTracking && response.startTime) {
                    if (!timerInterval) {
                        startTimerDisplay(response.startTime);
                    }
                } else if (!response.isTracking) {
                    stopTimerDisplay();
                    if (response.startTime) {
                        updateTimerDisplay(response.startTime, response.stopTime);
                    } else {
                        timerDiv.textContent = "Session Time: 00:00";
                    }
                }
            }
        });
    }

    startBtn.addEventListener('click', () => {
        statusDiv.textContent = "Status: Starting...";
        startTimerDisplay(Date.now());
        chrome.runtime.sendMessage({ action: "startTracking" }, (response) => {
            if (response) statusDiv.textContent = "Status: Tracking...";
        });
    });

    stopBtn.addEventListener('click', () => {
        stopTimerDisplay();
        chrome.runtime.sendMessage({ action: "stopTracking" }, (response) => {
            if (response) statusDiv.textContent = "Status: Stopped";
        });
    });

    clearBtn.addEventListener('click', () => {
        chrome.runtime.sendMessage({ action: "clearData" }, (response) => {
            if (response) {
                statusDiv.textContent = "Status: Data Cleared";
                logContainer.style.display = 'none';
                logContainer.innerHTML = '';
                
                topGazeInterestHero.style.display = 'none';
                gazeInterestsContainer.style.display = 'none';
                topMouseInterestHero.style.display = 'none';
                mouseInterestsContainer.style.display = 'none';
                
                timerDiv.textContent = "Session Time: 00:00";
                mouseCounter.textContent = "Mouse: 0 pts";
                gazeCounter.textContent = "Gaze: 0 pts";
            }
        });
    });

    heatmapBtn.addEventListener('click', () => {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            if (tabs[0]) {
                chrome.tabs.sendMessage(tabs[0].id, { action: "toggleHeatmap" });
            }
        });
    });

    logsBtn.addEventListener('click', () => {
        chrome.runtime.sendMessage({ action: "getLogs" }, (response) => {
            if (response && response.logs) {
                renderLogs(response.logs);
            } else {
                logContainer.style.display = 'block';
                logContainer.textContent = "No logs.";
            }
        });
    });

    function updateUI(status) {
        if (status.isTracking) {
            statusDiv.textContent = "Status: Tracking Gaze & Mouse";
            statusDiv.style.color = "green";
        } else {
            statusDiv.textContent = "Status: Stopped";
            statusDiv.style.color = "#666";
        }
    }

    function updateConnectionUI(isConnected) {
        if (isConnected) {
            statusDot.className = "status-dot connected";
            statusText.textContent = "Gaze Engine: Connected";
            statusText.style.color = "#2e7d32";
        } else {
            statusDot.className = "status-dot disconnected";
            statusText.textContent = "Gaze Engine: Offline";
            statusText.style.color = "#c62828";
        }
    }

    function renderLogs(logs) {
        logContainer.style.display = 'block';
        if (logs.length === 0) {
            logContainer.textContent = "No clicks recorded.";
            return;
        }

        let html = '<ul style="padding-left: 0; margin: 0; list-style-type: none;">';
        logs.slice().reverse().forEach(log => {
            const gazeOffsetStr = (log.gazeX && log.gazeY) ? 
                `<br><span style="color:#d32f2f; font-size:9px;">Gaze: (${log.gazeX}, ${log.gazeY}) | Click: (${log.x}, ${log.y})</span>` : '';
                
            html += `<li style="margin-bottom: 8px; border-bottom: 1px solid #eee; padding-bottom: 4px;">
                        <span style="color: #666; font-size: 10px;">${log.timestamp}</span><br>
                        <b>${log.element}</b> <br>
                        <span style="color: #333;">"${log.text}"</span>
                        ${gazeOffsetStr}
                        <div style="font-size:9px; color:#aaa; overflow:hidden; white-space:nowrap; text-overflow:ellipsis;">${log.url}</div>
                     </li>`;
        });
        html += '</ul>';
        logContainer.innerHTML = html;
    }

    function formatDuration(ms) {
        return Math.round(ms / 1000) + 's';
    }

    function renderGazeInterests(interests) {
        if (!interests || interests.length === 0) {
            topGazeInterestHero.style.display = 'none';
            gazeInterestsContainer.style.display = 'none';
            return;
        }

        // Hero
        const top = interests[0];
        topGazeInterestHero.style.display = 'block';
        heroGazeElement.textContent = top.element;
        heroGazeTime.textContent = formatDuration(top.duration);

        // List
        if (interests.length > 1) {
            gazeInterestsContainer.style.display = 'block';
            let html = '<div style="margin-top:5px;">';
            interests.slice(1).forEach((item, index) => {
                html += `<div class="interest-item">
                            ${index + 2}. <b>${item.element}</b> <span style="float:right; color: #555;">${formatDuration(item.duration)}</span>
                         </div>`;
            });
            html += '</div>';
            gazeInterestsContainer.innerHTML = html;
        } else {
            gazeInterestsContainer.style.display = 'none';
        }
    }

    function renderMouseInterests(interests) {
        if (!interests || interests.length === 0) {
            topMouseInterestHero.style.display = 'none';
            mouseInterestsContainer.style.display = 'none';
            return;
        }

        // Hero
        const top = interests[0];
        topMouseInterestHero.style.display = 'block';
        heroMouseElement.textContent = top.element;
        heroMouseTime.textContent = formatDuration(top.duration);

        // List
        if (interests.length > 1) {
            mouseInterestsContainer.style.display = 'block';
            let html = '<div style="margin-top:5px;">';
            interests.slice(1).forEach((item, index) => {
                html += `<div class="interest-item">
                            ${index + 2}. <b>${item.element}</b> <span style="float:right; color: #555;">${formatDuration(item.duration)}</span>
                         </div>`;
            });
            html += '</div>';
            mouseInterestsContainer.innerHTML = html;
        } else {
            mouseInterestsContainer.style.display = 'none';
        }
    }
});
