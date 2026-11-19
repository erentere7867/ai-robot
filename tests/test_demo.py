"""The demo: what the page asks for reaches the robot, a room that fails to load never takes the page down, the room
picker shows the room the robot is in (or the one loading), and a room swap clears what the last room left on the
page."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from types import SimpleNamespace

import pytest

from robot.eval.suite import office
from robot.interfaces import demo, voice
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


@pytest.mark.parametrize("loads", [False, True])
def test_the_room_picker_shows_the_room_loading_then_the_room_loaded(loads):
    live, holder = Live(), Holder(SimRobot(office()))
    live.update(scene="office", scenes=list(demo.SCENES))
    server = live.serve(port=0)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    release = threading.Event()

    def new_scene(name: str) -> None:
        release.wait(3.0)
        if not loads:
            raise FileNotFoundError("[Errno 2] No such file or directory: 'habitat/bin/python'")
        live.update(scene=name, status="idle")

    def state() -> dict:
        with urllib.request.urlopen(f"{url}/state") as response:
            return json.load(response)

    stop = threading.Event()
    loop = threading.Thread(target=drive, args=(live, None, holder, new_scene), kwargs={"stop_event": stop},
                            daemon=True)
    loop.start()
    try:
        pick = urllib.request.Request(f"{url}/scene", data=json.dumps({"text": "3D apartment"}).encode(),
                                      headers={"Content-Type": "application/json"})
        urllib.request.urlopen(pick).close()
        assert state()["scene_loading"] == "3D apartment"

        release.set()
        assert wait_for(lambda: live.state.status == ("idle" if loads else "failed"))
        assert wait_for(lambda: state()["scene_loading"] == "")
        assert state()["scene"] == ("3D apartment" if loads else "office")
    finally:
        stop.set()
        loop.join(2.0)
        server.shutdown()


@pytest.mark.parametrize("said,chime", [([], ""), (["I could not find the vase."], "uh_oh")])
def test_swapping_rooms_clears_reply_and_chime(monkeypatch, said, chime):
    live = Live()
    live.update(said=said, chime=chime)
    robot = SimRobot(office())
    forgotten = []
    monkeypatch.setattr("sys.argv", ["demo", "--scene", "living room"])
    monkeypatch.setattr(demo, "Live", lambda: live)
    monkeypatch.setattr(live, "serve", lambda port: None)
    monkeypatch.setattr(voice, "attach", lambda live: None)
    monkeypatch.setattr(demo, "Brain", lambda url: SimpleNamespace(healthy=lambda: True))
    monkeypatch.setattr(demo, "Agent", lambda *args, **kwargs: SimpleNamespace(forget=lambda: forgotten.append(True)))
    monkeypatch.setattr(demo, "make_robot", lambda *args: robot)
    monkeypatch.setattr(demo, "show_cameras", lambda *args: None)
    monkeypatch.setattr(demo.threading.Thread, "start", lambda self: None)

    def swap(live, agent, holder, new_scene):
        new_scene("two rooms")
        assert holder.sim is robot
        assert live.state.scene == "two rooms"
        assert live.state.status == "idle"
        assert live.state.said == []
        assert live.state.chime == ""
        assert forgotten == [True]

    monkeypatch.setattr(demo, "drive", swap)
    demo.main()
