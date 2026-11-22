"""Load the files of a recorded session (recordings/<session>/) into Python."""
import csv
import json
from collections import defaultdict

import numpy as np


def frame_times(folder):
    """Timestamp of every video frame, as a numpy array (index = frame number)."""
    return np.array([float(r["t"]) for r in csv.DictReader(open(folder / "frames.csv"))])


def key_events(folder):
    """[(t, key, action), ...] from keys.csv."""
    return [(float(r["t"]), r["key"], r["action"])
            for r in csv.DictReader(open(folder / "keys.csv"))]


def hands(folder):
    """{frame: {"Left"/"Right": 21x3 array of (x px, y px, z)}} from hands.csv."""
    out = defaultdict(dict)
    for r in csv.DictReader(open(folder / "hands.csv")):
        points = np.array([[float(r[f"{a}{i}"]) for a in "xyz"] for i in range(21)])
        out[int(r["frame"])][r["hand"]] = points
    return out


def homographies(folder):
    """{frame: 3x3 H mapping layout units -> pixels} from keyboard.csv."""
    return {int(r["frame"]): np.array([float(r[f"h{i}"]) for i in range(9)]).reshape(3, 3)
            for r in csv.DictReader(open(folder / "keyboard.csv"))}


def calibration(folder):
    return np.array(json.load(open(folder / "calibration.json"))["homography"])
