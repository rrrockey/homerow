"""Open the camera and read frames, the same way for every stage.

Calibration is stored in pixel coordinates, so calibrating and recording must
use identical camera settings. Keeping the setup here guarantees that.
"""
import sys
import time

import cv2


def open_camera(index):
    # AVFoundation is the macOS camera backend (needed for Continuity Camera).
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {index}")
    # Ask for 1080p @ 60. The camera may give less; preview.py shows what we get.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)
    return cap


def read_frame(cap, timeout=5.0):
    """Next frame. Continuity Camera can be slow to start, so retry before giving up."""
    start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if ok:
            return frame
        if time.perf_counter() - start > timeout:
            sys.exit(f"No frames from camera for {timeout:.0f}s, giving up")
