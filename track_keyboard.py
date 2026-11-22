"""Stage 6a: where the keyboard is in every frame of a recorded session.

Runs the KeyboardTracker over video.mp4 and writes keyboard.csv:

    frame, inliers, h0 ... h8

h0..h8 is that frame's homography (layout units -> pixels), row by row.
inliers is how many feature matches agreed; if too few, the tracker held its
last good position for that frame.

Usage:  python track_keyboard.py recordings/<session>
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

import sessiondata
from tracker import MIN_INLIERS, KeyboardTracker


def track(folder):
    ref = cv2.imread(str(folder / "calibration_ref.png"))
    tracker = KeyboardTracker(ref, sessiondata.calibration(folder))
    video = cv2.VideoCapture(str(folder / "video.mp4"))

    inliers = []
    with open(folder / "keyboard.csv", "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["frame", "inliers"] + [f"h{i}" for i in range(9)])
        i = 0
        while True:
            ok, frame = video.read()
            if not ok:
                break
            H, n = tracker.update(frame)
            out.writerow([i, n] + [f"{v:.8g}" for v in H.flatten()])
            inliers.append(n)
            i += 1
            if i % 200 == 0:
                print(f"  frame {i}")

    inliers = np.array(inliers)
    print(f"Wrote {folder / 'keyboard.csv'} ({len(inliers)} frames)")
    print(f"inliers: median {int(np.median(inliers))}, min {inliers.min()}; "
          f"frames held at last position: {(inliers < MIN_INLIERS).sum()}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python track_keyboard.py recordings/<session>")
    track(Path(sys.argv[1]))
