from types import SimpleNamespace

import pytest

from robot.interfaces import demo, voice
from robot.interfaces.live import Live


@pytest.mark.parametrize("said,chime", [([], ""), (["I could not find the vase."], "uh_oh")])
def test_swapping_rooms_clears_reply_and_chime(monkeypatch, said, chime):
    live = Live()
    live.update(said=said, chime=chime)
    robot = object()
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
