"""In-process runtime control hooks for robot-client runners."""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable


@dataclasses.dataclass
class RuntimeControl:
    """Thread-safe pause/resume/stop control shared with a runner.

    This is intentionally tiny: callers can keep the robot client alive and
    trigger interruptions without file polling or restarting the Python process.
    """

    pause_event: threading.Event = dataclasses.field(default_factory=threading.Event)
    resume_event: threading.Event = dataclasses.field(default_factory=threading.Event)
    stop_event: threading.Event = dataclasses.field(default_factory=threading.Event)
    paused_event: threading.Event = dataclasses.field(default_factory=threading.Event)
    on_pause: Callable[[], None] | None = None

    def request_pause(self) -> None:
        self.resume_event.clear()
        self.paused_event.clear()
        self.pause_event.set()

    def resume(self) -> None:
        self.pause_event.clear()
        self.resume_event.set()

    def request_stop(self) -> None:
        self.stop_event.set()
        self.resume_event.set()

    def wait_until_paused(self, timeout: float | None = None) -> bool:
        return self.paused_event.wait(timeout)
