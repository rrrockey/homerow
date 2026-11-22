"""Stage 5: offline hand tracking for a recorded session.

Reads recordings/<session>/video.mp4, runs MediaPipe on every frame and writes
recordings/<session>/hands.csv, one row per detected hand per frame:

    frame, t, hand, score, x0, y0, z0, ... x20, y20, z20

x/y are pixels in the full-size video frame, the same coordinates the keyboard
calibration uses. z is MediaPipe's relative depth (smaller = closer to camera).
t is the frame's timestamp from frames.csv, on the same clock as keys.csv.

Usage:  python track_hands.py recordings/<session>     (q in the window stops early)
"""
import csv
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

FINGERTIPS = (4, 8, 12, 16, 20)  # thumb, index, middle, ring, pinky
PREVIEW_W = 960


def make_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO,  # tracks hands from frame to frame
        num_hands=2,
    )
    return vision.HandLandmarker.create_from_options(options)


def header():
    cols = ["frame", "t", "hand", "score"]
    for i in range(21):
        cols += [f"x{i}", f"y{i}", f"z{i}"]
    return cols


def draw(frame, points):
    for c in vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, points[c.start], points[c.end], (255, 255, 255), 2)
    for i in FINGERTIPS:
        cv2.circle(frame, points[i], 8, (0, 255, 0), -1)


def track(folder):
    frame_times = [float(r["t"]) for r in csv.DictReader(open(folder / "frames.csv"))]
    video = cv2.VideoCapture(str(folder / "video.mp4"))
    landmarker = make_landmarker()

    hands_per_frame = [0, 0, 0]  # how many frames had 0, 1, 2 hands
    last_ms = -1
    start = time.perf_counter()

    with open(folder / "hands.csv", "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(header())

        for i, t in enumerate(frame_times):
            ok, frame = video.read()
            if not ok:
                print(f"Video ended early: frame {i} of {len(frame_times)}")
                break

            # VIDEO mode needs strictly increasing timestamps in ms. Using the
            # real capture times (not i * 33ms) keeps the tracker honest about
            # how much time passed between frames.
            ms = max(int(round((t - frame_times[0]) * 1000)), last_ms + 1)
            last_ms = ms
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)

            h, w = frame.shape[:2]
            hands_per_frame[len(result.hand_landmarks)] += 1
            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                best = handedness[0]
                row = [i, t, best.category_name, round(best.score, 3)]
                for lm in landmarks:
                    row += [round(lm.x * w, 1), round(lm.y * h, 1), round(lm.z, 4)]
                out.writerow(row)
                draw(frame, [(int(lm.x * w), int(lm.y * h)) for lm in landmarks])

            preview = cv2.resize(frame, (PREVIEW_W, int(PREVIEW_W * h / w)))
            cv2.imshow("homerow track hands", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped early")
                break
            if i % 100 == 0:
                print(f"  frame {i}/{len(frame_times)}")

    total = sum(hands_per_frame)
    seconds = time.perf_counter() - start
    print(f"Wrote {folder / 'hands.csv'}")
    print(f"{total} frames in {seconds:.0f}s ({total / seconds:.1f} frames/s)")
    print(f"frames with 2 hands: {hands_per_frame[2] / total:.0%}, "
          f"1 hand: {hands_per_frame[1] / total:.0%}, 0 hands: {hands_per_frame[0] / total:.0%}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python track_hands.py recordings/<session>")
    track(Path(sys.argv[1]))
