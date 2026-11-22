"""Stage 3: calibrate the keyboard.

Click the 4 outer corners of the key block, in this order:
    1. top-left      (top-left of the ` key)
    2. top-right     (top-right of the delete key)
    3. bottom-right  (bottom-right of the right shift key)
    4. bottom-left   (bottom-left of the left shift key)

Then the key grid is drawn over the live video. If it doesn't line up, press r
and click again. Press s to save calibration.json, q to quit.

Usage:  python calibrate.py [camera_index]
"""
import json
import sys
import time

import cv2
import numpy as np

import layout
from tracker import KeyboardTracker

camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
WINDOW = "homerow calibrate"
clicks = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
        clicks.append((x, y))


def compute_homography():
    # Maps layout units -> image pixels, using the 4 clicked corners.
    src = np.array(layout.BLOCK_CORNERS, dtype=np.float32)
    dst = np.array(clicks, dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def project(H, points):
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def draw_grid(frame, H):
    for label, key in layout.KEYS.items():
        poly = project(H, key["corners"]).astype(np.int32)
        cv2.polylines(frame, [poly], True, (0, 255, 255), 1)
        cx, cy = project(H, [key["center"]])[0]
        cv2.putText(frame, label[:1] if len(label) > 1 else label,
                    (int(cx) - 6, int(cy) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    sys.exit(f"Could not open camera {camera_index}")

cv2.namedWindow(WINDOW)
cv2.setMouseCallback(WINDOW, on_mouse)
last_good = time.perf_counter()
tracker = None

while True:
    ok, frame = cap.read()
    if not ok:
        if time.perf_counter() - last_good > 5:
            sys.exit("No frames from camera for 5s, giving up")
        continue
    last_good = time.perf_counter()

    if len(clicks) == 4:
        if tracker is None:
            # Remember this clean frame (before anything is drawn on it).
            tracker = KeyboardTracker(frame, compute_homography())
        H, inliers = tracker.update(frame)
        draw_grid(frame, H)
        status = f"tracking ({inliers} matches)  s = save, r = redo, q = quit"
    else:
        names = ["top-left", "top-right", "bottom-right", "bottom-left"]
        status = f"click corner {len(clicks) + 1}/4: {names[len(clicks)]}"

    for p in clicks:
        cv2.circle(frame, p, 6, (0, 0, 255), -1)
    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)
    cv2.imshow(WINDOW, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("r"):
        clicks.clear()
        tracker = None
    if key == ord("s") and tracker is not None:
        # Save the calibration and its reference frame; later stages rebuild
        # the tracker from these.
        with open("calibration.json", "w") as f:
            json.dump({"homography": tracker.H_calib.tolist()}, f)
        cv2.imwrite("calibration_ref.png", tracker.ref_frame)
        print("Saved calibration.json and calibration_ref.png")

cap.release()
cv2.destroyAllWindows()
