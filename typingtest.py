"""Stage 4b: a Monkeytype-style typing test drawn in an OpenCV window.

The test never reads the keyboard itself. It is fed the same (time, key, action)
events that KeyLogger records, so what you see on screen and what gets logged
can never disagree.

Run it on its own to try it:  python typingtest.py     (Esc quits)
"""
import random
import re
import time

import cv2
import numpy as np

from keylog import SHIFTED, KeyLogger

WORDS = """the be to of and a in that have it for not on with he as you do at this
but his by from they we say her she or an will my one all would there their what
so up out if about who get which go me when make can like time no just him know
take people into year your good some could them see other than then now look only
come its over think also back after use two how our work first well way even new
want because any these give day most us are is was were been has had said did made
find long down side been called water little very still world great where much
through before right too mean old same tell boy follow came show also around form
three small set put end does another large must big even such because turn here
why ask went men read need land different home move try kind hand picture again
change off play spell air away animal house point page letter mother answer found
study never last city tree cross farm hard start might story saw far sea draw left
late run while press close night real life few north open seem together next white""".split()

# Typing a shifted symbol: the reverse of keylog.SHIFTED (e.g. '1' -> '!').
UNSHIFTED_TO_SHIFTED = {v: k for k, v in SHIFTED.items()}

W, H = 1100, 400
MARGIN = 50
LINE_HEIGHT = 60
FONT = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 0.9
THICK = 2

BG = (30, 30, 30)
UNTYPED = (130, 130, 130)
CORRECT = (240, 240, 240)
WRONG = (60, 60, 230)  # BGR red
CURSOR = (0, 200, 255)


def make_passage(num_words=40):
    """Sentences of 6-10 random words: capital at the start, comma, full stop.

    The capitals matter: they are what exercise the shift keys later.
    """
    words = []
    while len(words) < num_words:
        sentence = random.sample(WORDS, random.randint(6, 10))
        sentence[0] = sentence[0].capitalize()
        if random.random() < 0.5:
            sentence[3] += ","
        sentence[-1] += "."
        words += sentence
    return " ".join(words)


class TypingTest:
    def __init__(self, target=None):
        self.target = target or make_passage()
        self.typed = [None] * len(self.target)  # what was typed at each position
        self.cursor = 0
        self.shift = False
        self.keystrokes = 0          # every character keypress, even later-corrected ones
        self.correct_keystrokes = 0
        self.t_start = None          # set on the first keypress
        self.t_end = None            # set when the last character is typed
        self.positions = self._layout()

    @property
    def finished(self):
        return self.cursor == len(self.target)

    def _layout(self):
        """Pixel position (x, baseline y, width) of every character, wrapped by word."""
        positions = []
        x, y = MARGIN, 140
        for token in re.findall(r"\S+ ?", self.target):  # a word plus its trailing space
            widths = [cv2.getTextSize(c, FONT, SCALE, THICK)[0][0] for c in token]
            word_width = sum(widths[:len(token.rstrip())])
            if x + word_width > W - MARGIN and x > MARGIN:
                x, y = MARGIN, y + LINE_HEIGHT
            for width in widths:
                positions.append((x, y, width))
                x += width
        return positions

    def _char_for(self, name):
        """The character a key produces, taking the shift state into account."""
        if name == "space":
            return " "
        if len(name) != 1:
            return None  # arrows, ctrl, esc... don't type anything
        if self.shift:
            return name.upper() if name.isalpha() else UNSHIFTED_TO_SHIFTED.get(name, name)
        return name

    def handle(self, name, action, t):
        """Feed one key event from KeyLogger."""
        if name in ("shift_l", "shift_r"):
            self.shift = action == "down"
            return
        if action != "down" or self.finished:
            return

        if name == "delete":
            if self.cursor > 0:
                self.cursor -= 1
                self.typed[self.cursor] = None
            return

        char = self._char_for(name)
        if char is None:
            return
        if self.t_start is None:
            self.t_start = t  # the clock starts on the first keypress, like Monkeytype
        self.keystrokes += 1
        self.correct_keystrokes += char == self.target[self.cursor]
        self.typed[self.cursor] = char
        self.cursor += 1
        if self.finished:
            self.t_end = t

    def elapsed(self, now):
        if self.t_start is None:
            return 0.0
        return (self.t_end or now) - self.t_start

    def wpm(self, now):
        seconds = self.elapsed(now)
        if seconds < 1:  # too early for a meaningful number
            return 0.0
        correct = sum(t == c for t, c in zip(self.typed, self.target))
        return (correct / 5) / (seconds / 60)  # a "word" is 5 characters

    def accuracy(self):
        if not self.keystrokes:
            return 100.0
        return 100 * self.correct_keystrokes / self.keystrokes

    def render(self, now):
        img = np.full((H, W, 3), BG, np.uint8)
        stats = (f"{self.wpm(now):.0f} wpm    {self.accuracy():.0f}% acc    "
                 f"{self.elapsed(now):.0f}s")
        if self.finished:
            stats += "    Done!"
        cv2.putText(img, stats, (MARGIN, 70), FONT, 1.0, CURSOR, 2)

        for i, char in enumerate(self.target):
            x, y, width = self.positions[i]
            typed = self.typed[i]
            if typed is None:
                color = UNTYPED
            elif typed == char:
                color = CORRECT
            else:
                color = WRONG
            if char == " ":
                if color == WRONG:  # a wrong character where a space belonged
                    cv2.line(img, (x, y + 4), (x + width, y + 4), WRONG, 3)
            else:
                cv2.putText(img, char, (x, y), FONT, SCALE, color, THICK)

        if not self.finished:
            x, y, width = self.positions[self.cursor]
            cv2.line(img, (x, y + 12), (x + width, y + 12), CURSOR, 3)
        return img


if __name__ == "__main__":
    logger = KeyLogger()
    test = TypingTest()
    logger.start()

    seen = 0
    running = True
    while running:
        while seen < len(logger.events):
            t, name, action = logger.events[seen]
            seen += 1
            if name == "esc" and action == "down":
                running = False
            test.handle(name, action, t)
        cv2.imshow("homerow typing test", test.render(time.perf_counter()))
        cv2.waitKey(15)  # lets OpenCV redraw the window; we ignore its own key input

    logger.stop()
    cv2.destroyAllWindows()
