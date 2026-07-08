// Background service worker
console.log("Mouse Tracker background script loaded.");

// Persistent State
let state = {
    isTracking: false,
    trailData: [],    // For visual path
    heatmapData: [],  // For heatmap intensity
    clickData: [],
    dwellData: {}, // Map of elementSelector -> duration (ms)
    
    // Gaze State
    gazeTrailData: [],
    gazeHeatmapData: [],
    gazeDwellData: {},
    showGazeCursor: true,
    heatmapType: "combined"
};

// WebSocket connection to Python Gaze server
let gazeWS = null;
let isGazeConnected = false;

function connectGazeWebSocket() {
    if (gazeWS) {
        try { gazeWS.close(); } catch(e) {}
    }
    
    console.log("Attempting to connect to Gaze Python server...");
    gazeWS = new WebSocket("ws://localhost:8765");
    
    gazeWS.onopen = () => {
        console.log("Connected to Gaze Python server");
        isGazeConnected = true;
        broadcastConnectionStatus();
    };
    
    gazeWS.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === "gaze") {
                handleGazeCoordinate(data.x, data.y);
            }
        } catch(e) {
            console.error("Error parsing gaze message:", e);
        }
    };
    
    gazeWS.onclose = () => {
        console.log("Disconnected from Gaze Python server");
        isGazeConnected = false;
        broadcastConnectionStatus();
        if (state.isTracking) {
            // Retry connection in 2 seconds
            setTimeout(() => {
                if (state.isTracking) connectGazeWebSocket();
            }, 2000);
        }
    };
    
    gazeWS.onerror = (err) => {
        console.error("Gaze WS error:", err);
    };
}

function disconnectGazeWebSocket() {
    if (gazeWS) {
        try { gazeWS.close(); } catch(e) {}
        gazeWS = null;
    }
    isGazeConnected = false;
    broadcastConnectionStatus();
    console.log("Disconnected from Gaze Python server (manual)");
}

function handleGazeCoordinate(x, y) {
    // Broadcast the coordinate in real-time to the active tab's content script
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs && tabs[0]) {
            chrome.tabs.sendMessage(tabs[0].id, {
                action: "gazeUpdate",
                x,
                y,
                showGazeCursor: state.showGazeCursor
            }).catch(() => {
                // Content script not loaded in this tab (e.g. chrome:// tabs)
            });
        }
    });
}

function broadcastConnectionStatus() {
    chrome.runtime.sendMessage({
        action: "connectionUpdate",
        isGazeConnected
    }).catch(() => {});
}

// Load state from storage on startup
let stateLoaded = false;
const stateReady = new Promise((resolve) => {
    chrome.storage.local.get(['mouseTrackerState'], (result) => {
        if (result.mouseTrackerState) {
            console.log("State restored from storage");
            state = result.mouseTrackerState;
            
            // Clean up any missing properties if upgrading state
            if (state.gazeTrailData === undefined) state.gazeTrailData = [];
            if (state.gazeHeatmapData === undefined) state.gazeHeatmapData = [];
            if (state.gazeDwellData === undefined) state.gazeDwellData = {};
            if (state.showGazeCursor === undefined) state.showGazeCursor = true;
            if (state.heatmapType === undefined) state.heatmapType = "combined";

            if (state.isTracking) {
                connectGazeWebSocket();
            }
        }
        stateLoaded = true;
        resolve();
    });
});

// Save state to storage (debounced)
let saveTimeout;
function saveState() {
    if (saveTimeout) clearTimeout(saveTimeout);
    saveTimeout = setTimeout(() => {
        chrome.storage.local.set({ mouseTrackerState: state });
    }, 1000); // Save at most once per second
}

// Listen for messages from content scripts and popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    // Wait for state to be ready before processing ANY message
    stateReady.then(() => {
        handleMessage(request, sender, sendResponse);
    });
    return true; // Keep message channel open for async response
});

function handleMessage(request, sender, sendResponse) {
    switch (request.action) {
        case "startTracking":
            state.isTracking = true;
            state.startTime = Date.now();
            state.stopTime = null;
            console.log("Tracking started (background)");
            connectGazeWebSocket();
            saveState();
            broadcastState();
            sendResponse({ status: "tracking", startTime: state.startTime });
            break;

        case "stopTracking":
            state.isTracking = false;
            state.stopTime = Date.now();
            console.log("Tracking stopped (background)");
            disconnectGazeWebSocket();
            saveState();
            broadcastState();
            sendResponse({ status: "stopped" });
            break;

        case "clearData":
            state.trailData = [];
            state.heatmapData = [];
            state.clickData = [];
            state.dwellData = {};
            state.gazeTrailData = [];
            state.gazeHeatmapData = [];
            state.gazeDwellData = {};
            state.startTime = null; // Clear time
            chrome.storage.local.remove('mouseTrackerState');
            console.log("Data cleared (background)");
            broadcastState();
            sendResponse({ status: "cleared" });
            break;

        case "logData":
            if (state.isTracking) {
                let changed = false;
                if (request.trail) {
                    state.trailData.push(...request.trail);
                    changed = true;
                }
                if (request.heatmap) {
                    state.heatmapData.push(...request.heatmap);
                    changed = true;
                }
                if (request.click) {
                    // Record both click position and Gaze position if available
                    state.clickData.push(...(Array.isArray(request.click) ? request.click : [request.click]));
                    changed = true;
                }
                if (request.dwell) {
                    const dwellItems = Array.isArray(request.dwell) ? request.dwell : [request.dwell];
                    dwellItems.forEach(item => {
                        const { element, duration } = item;
                        if (!state.dwellData[element]) {
                            state.dwellData[element] = 0;
                        }
                        state.dwellData[element] += duration;
                    });
                    changed = true;
                }
                
                // Gaze logging
                if (request.gazeTrail) {
                    state.gazeTrailData.push(...request.gazeTrail);
                    changed = true;
                }
                if (request.gazeHeatmap) {
                    state.gazeHeatmapData.push(...request.gazeHeatmap);
                    changed = true;
                }
                if (request.gazeDwell) {
                    const gazeDwellItems = Array.isArray(request.gazeDwell) ? request.gazeDwell : [request.gazeDwell];
                    gazeDwellItems.forEach(item => {
                        const { element, duration } = item;
                        if (!state.gazeDwellData[element]) {
                            state.gazeDwellData[element] = 0;
                        }
                        state.gazeDwellData[element] += duration;
                    });
                    changed = true;
                }

                if (changed) saveState();
            }
            sendResponse({ status: "logged" });
            break;

        case "getStatus":
        case "getLogs":
            sendResponse({
                isTracking: state.isTracking,
                startTime: state.startTime,
                stopTime: state.stopTime,
                hasData: state.trailData.length > 0 || state.gazeTrailData.length > 0 || state.clickData.length > 0,
                logs: state.clickData,
                mouseDataCount: state.trailData.length + state.heatmapData.length,
                gazeDataCount: state.gazeTrailData.length + state.gazeHeatmapData.length,
                isGazeConnected: isGazeConnected,
                showGazeCursor: state.showGazeCursor,
                heatmapType: state.heatmapType,
                topInterests: getTopInterests(),
                topGazeInterests: getTopGazeInterests()
            });
            break;

        case "getAllData":
            sendResponse({
                trailData: state.trailData,
                heatmapData: state.heatmapData,
                clickData: state.clickData,
                gazeTrailData: state.gazeTrailData,
                gazeHeatmapData: state.gazeHeatmapData,
                heatmapType: state.heatmapType
            });
            break;

        case "setGazeCursorVisibility":
            state.showGazeCursor = request.visible;
            saveState();
            sendResponse({ status: "updated", showGazeCursor: state.showGazeCursor });
            break;

        case "setHeatmapType":
            state.heatmapType = request.heatmapType;
            saveState();
            // Send update to content scripts if heatmap is active
            chrome.tabs.query({}, (tabs) => {
                tabs.forEach(tab => {
                    chrome.tabs.sendMessage(tab.id, {
                        action: "heatmapTypeUpdate",
                        heatmapType: state.heatmapType
                    }).catch(() => {});
                });
            });
            sendResponse({ status: "updated", heatmapType: state.heatmapType });
            break;
    }
}

function getTopInterests() {
    const entries = Object.entries(state.dwellData).map(([element, duration]) => ({
        element,
        duration,
        score: duration
    }));
    entries.sort((a, b) => b.duration - a.duration);
    return entries.slice(0, 5);
}

function getTopGazeInterests() {
    const entries = Object.entries(state.gazeDwellData).map(([element, duration]) => ({
        element,
        duration,
        score: duration
    }));
    entries.sort((a, b) => b.duration - a.duration);
    return entries.slice(0, 5);
}

function broadcastState() {
    chrome.tabs.query({}, (tabs) => {
        tabs.forEach(tab => {
            chrome.tabs.sendMessage(tab.id, {
                action: "statusUpdate",
                isTracking: state.isTracking
            }).catch(() => {});
        });
    });
}
