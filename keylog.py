"""Stage 4a: log every key press and release with a timestamp.

Timestamps use time.perf_counter(), the same clock the recorder will stamp video
frames with, so a keypress can later be matched to the frame it happened on.

Run it on its own to test:  python keylog.py     (press Esc to stop)
"""
import time

from pynput import keyboard

# Special keys, named to match the labels in layout.py.
SPECIAL = {
    keyboard.Key.shift: "shift_l",
    keyboard.Key.shift_r: "shift_r",
    keyboard.Key.enter: "return",
    keyboard.Key.backspace: "delete",
    keyboard.Key.tab: "tab",
    keyboard.Key.caps_lock: "caps",
    keyboard.Key.space: "space",
}

# A shifted symbol is the same physical key as its unshifted one (US layout).
SHIFTED = dict(zip('!@#$%^&*()_+{}|:"<>?~', "1234567890-=[]\\;',./`"))


def key_name(key):
    """Turn a pynput key into the physical key's name, e.g. 'A' -> 'a', '!' -> '1'."""
    if key in SPECIAL:
        return SPECIAL[key]
    if isinstance(key, keyboard.KeyCode) and key.char:
        c = key.char.lower()
        return SHIFTED.get(key.char, c)
    return str(key).replace("Key.", "")


class KeyLogger:
    def __init__(self):
        self.events = []  # (time, key_name, "down" | "up")
        self._listener = keyboard.Listener(
            on_press=lambda k: self._add(k, "down"),
            on_release=lambda k: self._add(k, "up"),
        )

    def _add(self, key, action):
        # Stamp immediately: the callback runs on pynput's own thread.
        self.events.append((time.perf_counter(), key_name(key), action))

    def start(self):
        self._listener.start()

    def stop(self):
        self._listener.stop()


if __name__ == "__main__":
    logger = KeyLogger()
    logger.start()
    print("Logging. Type something, then press Esc.")
    seen = 0
    while True:
        time.sleep(0.05)
        while seen < len(logger.events):
            t, name, action = logger.events[seen]
            print(f"{t:12.4f}  {action:4}  {name}")
            seen += 1
        if logger.events and logger.events[-1][1:] == ("esc", "down"):
            break
    logger.stop()
