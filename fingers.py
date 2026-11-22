"""The touch-typing chart: which finger should press each key.

Standard home-row method: fingers rest on A S D F / J K L ; and each one owns
the column of keys above and below its home key. Index fingers also cover the
middle columns (T G B, Y H N); pinkies cover everything at the outer edges.
"""

CHART = {
    ("Left", "pinky"): ["`", "1", "q", "a", "z", "tab", "caps", "shift_l"],
    ("Left", "ring"): ["2", "w", "s", "x"],
    ("Left", "middle"): ["3", "e", "d", "c"],
    ("Left", "index"): ["4", "5", "r", "t", "f", "g", "v", "b"],
    ("Right", "index"): ["6", "7", "y", "u", "h", "j", "n", "m"],
    ("Right", "middle"): ["8", "i", "k", ","],
    ("Right", "ring"): ["9", "o", "l", "."],
    ("Right", "pinky"): ["0", "-", "=", "p", "[", "]", "\\", ";", "'", "/",
                         "delete", "return", "shift_r"],
}

# key -> (hand, finger), e.g. "a" -> ("Left", "pinky")
EXPECTED = {key: owner for owner, keys in CHART.items() for key in keys}

# MediaPipe landmark index of each fingertip. Thumbs are left out on purpose:
# they rest on the space bar, right below the bottom row, and would otherwise
# often be "nearest" to keys like B or N without ever pressing them.
TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}

ORDER = [("Left", f) for f in ("pinky", "ring", "middle", "index")] + \
        [("Right", f) for f in ("index", "middle", "ring", "pinky")]


def name(owner):
    hand, finger = owner
    return f"{hand[0]}-{finger}"  # e.g. "L-pinky"
