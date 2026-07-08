# Setup and Run Instructions (InsightUX Gaze Tracking)

This guide provides all the commands and steps required to set up and run this project locally on a Windows machine.

---

## Prerequisites
*   **Operating System**: Windows 10/11
*   **Hardware**: A functional webcam.
*   **Software / Language Versions**: 
    *   **Python**: `3.11.9` (highly recommended; ONNXRuntime and MediaPipe have strict compatibility limits).
    *   **Node.js**: `v16+` (for Electron application).
    *   **Electron**: `v30.0.0` (specified in Electron configuration).

---

## Step 1: Python Virtual Environment Setup & Installation

Open PowerShell or Command Prompt in the project's root directory:

1. **Create a virtual environment:**
   ```powershell
   python -m venv venv
   ```

2. **Activate the virtual environment:**
   *   **PowerShell:**
       ```powershell
       .\venv\Scripts\Activate.ps1
       ```
   *   **Command Prompt (cmd):**
       ```cmd
       .\venv\Scripts\activate.bat
       ```

3. **Install Dependencies:**
   All required packages (including UI/Webview dependencies) are version-frozen in `requirements.txt`:
   ```powershell
   pip install -r requirements.txt
   ```


---

## Step 2: Running Calibration (Required First)
Before using any of the real-time trackers, you must calibrate the CNN and RBF mappings to your face, screen, and camera:

```powershell
python calibrate_FINAL_CURRENT.py
```
*   **Instructions**: Sit centered in front of the screen. Look directly at the red/green dots as they appear across your screen. Do not move your head; let only your eyes follow the dots.
*   This will save the calibration profile into `calibration.pkl` and `baseline_pose.pkl`.

---

## Step 3: Run the Application Targets

You have multiple options depending on how you want to run the project.

### Option A: Running the Electron Browser App (Recommended Interface)
This runs the full **InsightUX Browser** graphical interface, which automatically boots up the Python gaze server backend and visualizes real-time gaze trails, records web sessions, and performs poster layouts analysis.

1. Calibrate first (Step 2).
2. Open a terminal and navigate to the Electron folder:
   ```powershell
   cd MY_WEBGazer
   ```
3. Install node packages:
   ```powershell
   npm install
   ```
4. Start the application:
   ```powershell
   npm start
   ```
5. In the UI, click **Start Tracking** to launch the camera feed window and live overlay.

### Option B: Standalone Web Overlay (Python Only)
Runs a transparent screen overlay showing your current eye gaze zone estimates (3x3 grid) relative to your desktop monitor.

1. Calibrate first (Step 2).
2. Run from the root directory:
   ```powershell
   python main_webcam_pipeline.py
   ```
3. *Controls*: Press `Esc` or `q` to close, or `r` to reset the cursor smoothing anchor.

### Option C: Python-Webview Hosted Session (Automated Web Session Logger)
Hosts a webpage in a local window, auto-injects tracking listeners, and logs the DOM elements viewed/scrolled.

1. Calibrate first (Step 2).
2. Run from the root directory and pass a URL (or run it without arguments to get prompted):
   ```powershell
   python website_segmentation_FINAL_CURRENT.py https://wikipedia.org
   ```
3. Look at your captured gaze logs in:
   *   `sessions/live/gaze_log.jsonl` (raw coordinates)
   *   `sessions/live/dom_log.jsonl` (elements/AOIs viewed)

### Option D: Standalone Gaze Server (WebSocket Server)
If you want to stream gaze data from the camera to a custom browser extension or external applications:

1. Calibrate first (Step 2).
2. Run from the root directory:
   ```powershell
   python gaze_server.py
   ```
3. This runs the tracker and streams JSON coordinates over a WebSocket server at `ws://localhost:8765`.

---

## Diagnostic & Validation Scripts (Optional)

*   **Accuracy Validation:** Runs 9 check points on the screen to measure tracking error in pixels.
    ```powershell
    python validate_FINAL_CURRENT.py
    ```
*   **Test on custom images:** Runs gaze direction calculations on a directory of images.
    ```powershell
    python main_image.py
    ```
*   **Test on a custom video:** Processes an offline video file `test.mp4`.
    ```powershell
    python main_video.py
    ```
