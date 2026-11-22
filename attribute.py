"""Stage 7: which finger pressed each key, and was it the right one?

For every character keypress:
  1. take the frame at (key time + sync offset), when the key is going down
  2. map the 8 non-thumb fingertips onto the flat keyboard layout
  3. the fingertip nearest the pressed key's center pressed it
  4. compare with the touch-typing chart (fingers.py)

Writes presses.csv (one row per press) and prints a summary.
`margin` is how much farther the runner-up fingertip was, in key widths:
small margins are the presses where attribution is least sure.

Usage:  python attribute.py recordings/<session>
        (needs hands.csv, keyboard.csv and sync.json)
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

import layout
import sessiondata
from fingers import EXPECTED, ORDER, TIPS, name
from sync_offset import nearest_frame


def fingertips_on_keyboard(frame_hands, H):
    """[((hand, finger), (x, y) in key units), ...] for the 8 non-thumb fingers."""
    owners, pts = [], []
    for hand, points in frame_hands.items():
        for finger, i in TIPS.items():
            owners.append((hand, finger))
            pts.append(points[i, :2])
    pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    on_kb = cv2.perspectiveTransform(pts, np.linalg.inv(H)).reshape(-1, 2)
    return list(zip(owners, on_kb))


def attribute(folder):
    offset = json.load(open(folder / "sync.json"))["offset"]
    times = sessiondata.frame_times(folder)
    hands = sessiondata.hands(folder)
    Hs = sessiondata.homographies(folder)
    presses = [(t, k) for t, k, a in sessiondata.key_events(folder)
               if a == "down" and len(k) == 1 and k in layout.KEYS]

    rows = []
    for t, key in presses:
        f = nearest_frame(times, t + offset)
        if f not in hands or f not in Hs:
            continue  # no hands detected in that frame
        center = np.array(layout.KEYS[key]["center"])
        ranked = sorted(fingertips_on_keyboard(hands[f], Hs[f]),
                        key=lambda c: np.linalg.norm(c[1] - center))
        used = ranked[0][0]
        d1 = np.linalg.norm(ranked[0][1] - center)
        d2 = np.linalg.norm(ranked[1][1] - center)
        rows.append({
            "t": t, "key": key, "frame": f,
            "finger": name(used), "expected": name(EXPECTED[key]),
            "correct": used == EXPECTED[key],
            "distance": round(float(d1), 3), "margin": round(float(d2 - d1), 3),
        })

    with open(folder / "presses.csv", "w", newline="") as fh:
        out = csv.DictWriter(fh, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)

    summarize(rows, len(presses))
    print(f"Wrote {folder / 'presses.csv'}")


def summarize(rows, total):
    n = len(rows)
    print(f"Attributed {n} of {total} character presses "
          f"({sum(r['correct'] for r in rows) / n:.0%} with the correct finger)\n")

    used = Counter(r["finger"] for r in rows)
    expected = Counter(r["expected"] for r in rows)
    print("finger      used   should be")
    for owner in ORDER:
        f = name(owner)
        print(f"{f:10s} {used[f] / n:5.0%}   {expected[f] / n:5.0%}")

    by_key = defaultdict(list)
    for r in rows:
        by_key[r["key"]].append(r)
    print("\nkeys most often pressed with the wrong finger (3+ presses):")
    print("key  presses  wrong  usually typed with   should be")
    worst = sorted(((sum(not r["correct"] for r in rs) / len(rs), k, rs)
                    for k, rs in by_key.items() if len(rs) >= 3), reverse=True)
    for wrong, key, rs in worst[:10]:
        if wrong == 0:
            break
        usual = Counter(r["finger"] for r in rs).most_common(1)[0][0]
        print(f" {key}   {len(rs):6d}  {wrong:5.0%}  {usual:18s}  {rs[0]['expected']}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python attribute.py recordings/<session>")
    attribute(Path(sys.argv[1]))
