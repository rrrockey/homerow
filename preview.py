import sys
import time

import cv2

camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# AVFoundation is the macOS camera backend (needed for Continuity Camera).
cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    sys.exit(f"Could not open camera {camera_index}")

# Ask for 1080p @ 60. The camera may ignore this, so we print what we got.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)

print(
    f"Camera reports: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
    f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps"
)

frames = 0
window_start = time.perf_counter()
measured_fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Measured FPS = frames actually delivered per second, averaged over 1s.
    frames += 1
    now = time.perf_counter()
    if now - window_start >= 1.0:
        measured_fps = frames / (now - window_start)
        frames = 0
        window_start = now

    cv2.putText(frame, f"{measured_fps:.1f} fps", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imshow("homerow preview", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
