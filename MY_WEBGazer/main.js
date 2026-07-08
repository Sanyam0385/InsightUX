// WebGazer - main.js
const { app, BrowserWindow, session } = require('electron');
const path = require('path');

function createWindow() {
  // Disable CSP to allow injecting scripts and loading resources on external domains
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
    let responseHeaders = { ...details.responseHeaders };
    delete responseHeaders['content-security-policy'];
    delete responseHeaders['Content-Security-Policy'];
    callback({ responseHeaders });
  });

  // Automatically grant permission request for webcam (needed for WebGazer)
  session.defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
    if (permission === 'media') {
      callback(true);
    } else {
      callback(false);
    }
  });

  // Create the main browser window.
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    title: "WebGazer Analysis Browser",
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false, 
      webviewTag: true, // Enable the webview tag for guest content
    }
  });

  // Load the browser shell interface
  win.loadFile(path.join(__dirname, 'index.html'));

  // Open the DevTools automatically (essential for your development phase)
  win.webContents.openDevTools();
}

app.whenReady().then(createWindow);

// Quit when all windows are closed.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
