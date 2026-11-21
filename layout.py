"""A flat model of the MacBook's main four key rows, measured in key units.

1 unit = the width of a standard letter key. Every row is 14.5 units wide, so the
whole block is a 14.5 x 4 rectangle. Coordinates have the origin at the top-left
corner, x to the right, y downward (same orientation as the image).

Calibration finds the transform that maps this ideal rectangle onto the camera
image, which lets us project any key's position into the video.
"""

# (label, width in units) per row, top row first.
ROWS = [
    [("`", 1)] + [(c, 1) for c in "1234567890-="] + [("delete", 1.5)],
    [("tab", 1.5)] + [(c, 1) for c in "qwertyuiop[]"] + [("\\", 1)],
    [("caps", 1.75)] + [(c, 1) for c in "asdfghjkl;'"] + [("return", 1.75)],
    [("shift_l", 2.25)] + [(c, 1) for c in "zxcvbnm,./"] + [("shift_r", 2.25)],
]

WIDTH = 14.5
HEIGHT = float(len(ROWS))


def _build():
    keys = {}
    for row_index, row in enumerate(ROWS):
        x = 0.0
        for label, width in row:
            keys[label] = {
                "center": (x + width / 2, row_index + 0.5),
                "corners": [  # top-left, top-right, bottom-right, bottom-left
                    (x, row_index),
                    (x + width, row_index),
                    (x + width, row_index + 1),
                    (x, row_index + 1),
                ],
            }
            x += width
        assert abs(x - WIDTH) < 1e-9, f"row {row_index} is {x} units wide"
    return keys


KEYS = _build()

# The four corners of the whole block, in the order the user clicks them.
BLOCK_CORNERS = [(0, 0), (WIDTH, 0), (WIDTH, HEIGHT), (0, HEIGHT)]
