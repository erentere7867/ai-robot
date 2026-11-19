"""The demo's driver loop: what the page asks for reaches the robot, and a failure never takes the page down."""

from __future__ import annotations

import threading
import time

from robot.eval.suite import office
from robot.interfaces.demo import Holder, drive
from robot.interfaces.live import Live
from robot.sim.world import SimRobot


def wait_for(check, timeout_s: float = 3.0) -> bool:  # noqa: ANN001
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(0.02)
    return False


def test_a_room_that_fails_to_load_shows_an_error_and_the_page_keeps_running():
    live, holder = Live(), Holder(SimRobot(office()))
    before = holder.sim

    def new_scene(name: str) -> None:
        raise FileNotFoundError("[Errno 2] No such file or directory: 'habitat/bin/python'")

    stop = threading.Event()
    loop = threading.Thread(target=drive, args=(live, None, holder, new_scene), kwargs={"stop_event": stop},
                            daemon=True)
    loop.start()
    try:
        live.inbox.put(("scene", "3D apartment"))
        assert wait_for(lambda: live.state.status == "failed")
        assert loop.is_alive()
        assert holder.sim is before
        assert any("3D apartment" in p and "habitat/bin/python" in p for p in live.state.problems)

        live.inbox.put(("stop", ""))
        assert wait_for(lambda: live.state.status == "stopped")
    finally:
        stop.set()
        loop.join(2.0)
