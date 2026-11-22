"""Stage 6b: measure the delay between the keylog and the video.

Frames are stamped when they reach Python, not when the camera captured them,
so the moment of a keypress appears in the video at (key time + offset).

We don't know the offset, so we try a range of them. For each one, we look at
every character keypress, take the frame at (key time + offset), and measure
how far the nearest fingertip is from the pressed key. At the right offset the
fingers are actually on the keys, so that distance is smallest.

Distances are in key widths: fingertips are mapped from pixels back onto the
flat keyboard layout with the inverse of that frame's homography.

Writes sync.json with the best offset.

Usage:  python sync_offset.py recordings/<session>
        (needs hands.csv and keyboard.csv: run track_hands.py and track_keyboard.py first)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import layout
import sessiondata

FINGERTIPS = (4, 8, 12, 16, 20)
OFFSETS = np.arange(-0.100, 0.401, 0.010)  # seconds


def fingertips_on_keyboard(frame_hands, H):
    """All fingertips in a frame, in layout units (key widths)."""
    pts = np.array([hand[i, :2] for hand in frame_hands.values() for i in FINGERTIPS],
                   dtype=np.float32)
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)


def nearest_frame(times, t):
    i = int(np.searchsorted(times, t))
    if i == 0:
        return 0
    if i == len(times):
        return len(times) - 1
    return i if times[i] - t < t - times[i - 1] else i - 1


def median_distance(offset, presses, times, hands, Hs):
    dists = []
    for t, key in presses:
        f = nearest_frame(times, t + offset)
        if f not in hands or f not in Hs:
            continue
        tips = fingertips_on_keyboard(hands[f], Hs[f])
        center = np.array(layout.KEYS[key]["center"])
        dists.append(np.linalg.norm(tips - center, axis=1).min())
    return float(np.median(dists)), len(dists)


def main(folder):
    times = sessiondata.frame_times(folder)
    hands = sessiondata.hands(folder)
    Hs = sessiondata.homographies(folder)
    # Character keys only: big keys (space, shift) are pressed off-center.
    presses = [(t, k) for t, k, a in sessiondata.key_events(folder)
               if a == "down" and len(k) == 1 and k in layout.KEYS]
    print(f"{len(presses)} character key presses")

    results = [(off, *median_distance(off, presses, times, hands, Hs)) for off in OFFSETS]
    best = min(results, key=lambda r: r[1])
    worst = max(r[1] for r in results)

    print("offset   median distance from pressed key to nearest fingertip")
    for off, d, n in results:
        bar = "#" * int(40 * d / worst)
        mark = "  <- best" if off == best[0] else ""
        print(f"{off * 1000:5.0f}ms  {d:.3f} keys  {bar}{mark}")

    with open(folder / "sync.json", "w") as f:
        json.dump({"offset": round(best[0], 3), "median_distance_keys": round(best[1], 3),
                   "presses_used": best[2]}, f, indent=2)
    print(f"Best offset {best[0] * 1000:.0f} ms (median {best[1]:.2f} keys). Wrote sync.json")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python sync_offset.py recordings/<session>")
    main(Path(sys.argv[1]))
