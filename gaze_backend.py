import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: gaze_backend.py [--calibrate | --validate | --track] [args]")
        sys.exit(1)
        
    mode = sys.argv[1]
    
    if mode == "--calibrate":
        print("[gaze_backend] Launching Calibration Mode...")
        import calibrate_FINAL_CURRENT
        calibrate_FINAL_CURRENT.main()
    elif mode == "--validate":
        print("[gaze_backend] Launching Validation Mode...")
        import validate_FINAL_CURRENT
        validate_FINAL_CURRENT.main()
    elif mode == "--track":
        print("[gaze_backend] Launching Live Tracking Mode...")
        import gaze_server
        # Shift sys.argv so that gaze_server sees its argument (sessionDir) at index 1
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        gaze_server.main()
    else:
        print(f"[gaze_backend] Unknown mode: {mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
