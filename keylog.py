"""Stage 4a: log every key press and release with a timestamp.

Timestamps use time.perf_counter(), the same clock the recorder will stamp video
frames with, so a keypress can later be matched to the frame it happened on.

Run it on its own to test:  python keylog.py     (press Esc to stop)
"""
import multiprocessing
import queue
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


def _listen(out):
    """Runs in the child process: forward every key event to the queue `out`."""
    def add(key, action):
        # Stamp immediately. perf_counter() is one system-wide clock on macOS,
        # so these timestamps match the ones the parent puts on video frames.
        out.put((time.perf_counter(), key_name(key), action))

    listener = keyboard.Listener(
        on_press=lambda k: add(k, "down"),
        on_release=lambda k: add(k, "up"),
    )
    listener.start()
    listener.wait()
    out.put(None)  # tell the parent the listener is up and running
    listener.join()


class KeyLogger:
    """Captures key events in a separate process.

    pynput's listener asks macOS to translate keycodes from a background thread.
    macOS aborts the whole process (SIGTRAP) when that happens in a process that
    also has a window open, as the session does with OpenCV. In its own process
    it is safe.
    """

    def __init__(self):
        ctx = multiprocessing.get_context("spawn")
        self._queue = ctx.Queue()
        self._process = ctx.Process(target=_listen, args=(self._queue,), daemon=True)
        self._events = []

    @property
    def events(self):
        """All (time, key_name, "down" | "up") events so far, oldest first."""
        while True:
            try:
                self._events.append(self._queue.get_nowait())
            except queue.Empty:
                return self._events

    def start(self):
        """Start capturing. Blocks until the listener is ready (about a second),
        so call it before you need the first key."""
        self._process.start()
        self._queue.get(timeout=15)  # the "ready" signal

    def stop(self):
        self.events  # collect anything already sent
        self._process.terminate()


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
