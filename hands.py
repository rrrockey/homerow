"""Stage 2: run MediaPipe hand tracking on live video and draw the landmarks.

Usage:  python hands.py [camera_index]
Press q to quit.
"""
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# Landmark indices of the five fingertips (thumb, index, middle, ring, pinky).
FINGERTIPS = {4: "thumb", 8: "index", 12: "middle", 16: "ring", 20: "pinky"}

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,  # tracks across frames, needs timestamps
    num_hands=2,
)
landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    sys.exit(f"Could not open camera {camera_index}")

start = time.perf_counter()
last_good = start  # Continuity Camera can take a moment to deliver frames

while True:
    ok, frame = cap.read()
    if not ok:
        if time.perf_counter() - last_good > 5:
            print("No frames from camera for 5s, giving up")
            break
        continue
    last_good = time.perf_counter()

    # MediaPipe wants RGB; OpenCV gives BGR.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int((time.perf_counter() - start) * 1000)
    result = landmarker.detect_for_video(image, timestamp_ms)

    h, w = frame.shape[:2]
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        # Landmarks are normalized 0..1; multiply by frame size to get pixels.
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        for c in vision.HandLandmarksConnections.HAND_CONNECTIONS:
            cv2.line(frame, points[c.start], points[c.end], (255, 255, 255), 2)
        for i, p in enumerate(points):
            if i in FINGERTIPS:
                cv2.circle(frame, p, 10, (0, 255, 0), -1)
                cv2.putText(frame, FINGERTIPS[i], (p[0] + 12, p[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.circle(frame, p, 4, (0, 0, 255), -1)

        label = handedness[0].category_name
        cv2.putText(frame, label, (points[0][0], points[0][1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)

    cv2.imshow("homerow hands", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
