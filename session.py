"""Stage 4c: run one recorded typing session.

    1. Calibrate the keyboard (click 4 corners, press s).
    2. 3-second countdown (the passage is already visible; get your hands ready).
    3. Record: every camera frame and every key event is stamped with the same
       clock (time.perf_counter()), while the typing test runs on screen.
    4. Stop when the test is finished (or Esc) and save everything to
       recordings/<timestamp>/.

Usage:  python session.py [camera_index]
"""
import csv
import json
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from calibrate import calibrate, save_calibration
from camera import open_camera, read_frame
from keylog import KeyLogger
from typingtest import BG, FONT, TypingTest
from typingtest import W as PANEL_W

WINDOW = "homerow"
PREVIEW_W = 640          # the camera preview is small; the saved video is full size
COUNTDOWN_SECONDS = 3
TAIL_SECONDS = 1.0       # keep recording briefly after the last character


def compose(panel, frame):
    """Typing test on top, camera preview centered underneath."""
    h, w = frame.shape[:2]
    preview = cv2.resize(frame, (PREVIEW_W, int(PREVIEW_W * h / w)))
    top = panel.shape[0]
    canvas = np.full((top + preview.shape[0], PANEL_W, 3), BG, np.uint8)
    canvas[:top] = panel
    x = (PANEL_W - PREVIEW_W) // 2
    canvas[top:, x:x + PREVIEW_W] = preview
    return canvas


def countdown(cap, test):
    """Show the passage and camera while counting down. Returns the last frame."""
    start = time.perf_counter()
    while True:
        frame = read_frame(cap)
        remaining = COUNTDOWN_SECONDS - (time.perf_counter() - start)
        if remaining <= 0:
            return frame
        canvas = compose(test.render(0), frame)
        cv2.putText(canvas, f"Starting in {int(remaining) + 1}", (PANEL_W // 2 - 150, 380),
                    FONT, 1.5, (0, 200, 255), 3)
        cv2.imshow(WINDOW, canvas)
        cv2.waitKey(1)


class FrameWriter:
    """Encodes video on a background thread, so the capture loop never waits on it.

    Frames go into a queue and the thread writes them in order. OpenCV releases
    Python's GIL while encoding, so the two really do run in parallel.
    """

    def __init__(self, path, size, fps=30):
        self._writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while (frame := self._queue.get()) is not None:
            self._writer.write(frame)

    def write(self, frame):
        self._queue.put(frame)

    def close(self):
        """Finish writing every queued frame, then close the file."""
        self._queue.put(None)
        self._thread.join()
        self._writer.release()


def record(cap, test, first_frame, folder, logger):
    """Run the test while recording. Returns (frame_times, key_events)."""
    h, w = first_frame.shape[:2]
    writer = FrameWriter(folder / "video.mp4", (w, h))

    frame_times = []
    seen = len(logger.events)  # ignore anything typed before recording began
    aborted = False
    t_finished = None
    while True:
        frame = read_frame(cap)
        t = time.perf_counter()  # stamp the frame as soon as we have it
        writer.write(frame)      # the clean frame; nothing is drawn on it
        frame_times.append(t)

        while seen < len(logger.events):
            key_t, name, action = logger.events[seen]
            seen += 1
            if name == "esc" and action == "down":
                aborted = True
            test.handle(name, action, key_t)

        if test.finished and t_finished is None:
            t_finished = t
        if aborted or (t_finished is not None and t - t_finished > TAIL_SECONDS):
            break

        cv2.imshow(WINDOW, compose(test.render(t), frame))
        cv2.waitKey(1)  # lets OpenCV redraw; its own key input is ignored

    writer.close()
    key_events =[e for e in logger.events if e[0] >= frame_times[0]]
    return frame_times, key_events


def save(folder, test, frame_times, key_events, tracker):
    with open(folder / "frames.csv", "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["frame", "t"])
        out.writerows(enumerate(frame_times))
    with open(folder / "keys.csv", "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["t", "key", "action"])
        out.writerows(key_events)
    with open(folder / "test.json", "w") as f:
        json.dump({
            "target": test.target,
            "typed": "".join(c or "" for c in test.typed),
            "finished": test.finished,
            "wpm": test.wpm(frame_times[-1]),
            "accuracy": test.accuracy(),
        }, f, indent=2)
    save_calibration(tracker, str(folder))


if __name__ == "__main__":
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = open_camera(camera_index)

    tracker = calibrate(cap, WINDOW)
    if tracker is None:
        sys.exit("Calibration cancelled")

    # Start the keylogger first: it takes about a second to come up, and we
    # don't want to miss the first keys.
    logger = KeyLogger()
    logger.start()

    test = TypingTest()
    first_frame = countdown(cap, test)

    folder = Path("recordings") / datetime.now().strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True)
    frame_times, key_events = record(cap, test, first_frame, folder, logger)
    logger.stop()
    save(folder, test, frame_times, key_events, tracker)

    duration = frame_times[-1] - frame_times[0]
    print(f"Saved {folder}")
    print(f"{len(frame_times)} frames in {duration:.1f}s = "
          f"{len(frame_times) / duration:.1f} fps, {len(key_events)} key events")
    print(f"{test.wpm(frame_times[-1]):.0f} wpm, {test.accuracy():.0f}% accuracy")

    cap.release()
    cv2.destroyAllWindows()
