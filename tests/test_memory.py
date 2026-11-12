"""Long-term memory: what the robot saw, was told and was shown survives a restart (robot/mind/memory.py)."""

import time

from robot.agent.api import Pose
from robot.mind.memory import Memory
from robot.perception.tracker import Detection, Tracker


def _seen(tracker: Tracker, label: str, x: float, y: float, color: str = "", **kw) -> None:
    now = time.monotonic()
    for k in range(2):
        tracker.update([Detection(label, x, y, 0.1, color, score=0.8, **kw)], now + k * 0.1)


def test_what_was_seen_told_and_shown_comes_back_after_a_restart(tmp_path):
    before = Tracker()
    _seen(before, "keys", 3.8, -2.0, "silver")
    _seen(before, "person", 0.0, 1.0)  # people move: not remembered
    gone = before.objects()[0]
    _seen(before, "cup", 1.0, 1.0, "red")
    before.verify(next(t for t in before.objects() if t.label == "cup"), None)  # the vision model saw nothing there
    Memory(tmp_path / "home.db").save(before, ["the keys go in the bowl"], {"start": Pose(0, 0, 0),
                                                                            "kitchen door": Pose(2.0, 0.5, 90.0)})

    after, memory = Tracker(), Memory(tmp_path / "home.db")  # a new process: nothing in it yet
    assert memory.restore(after) == 1
    (keys,) = after.objects()
    assert (keys.label, keys.color, keys.id) == ("keys", "silver", gone.id)
    assert (keys.x, keys.y) == (3.8, -2.0) and time.monotonic() - keys.last_seen < 5  # its age came back too
    assert memory.notes() == ["the keys go in the bowl"] and memory.places() == {"kitchen door": (2.0, 0.5, 90.0)}
    _seen(after, "ball", -1.0, 2.0)  # new things get new ids
    assert len({t.id for t in after.objects()}) == 2


def test_a_thing_found_gone_is_not_remembered_as_there(tmp_path):
    tracker, memory = Tracker(), Memory(tmp_path / "home.db")
    _seen(tracker, "keys", 3.8, -2.0)
    for _ in range(3):
        tracker.missed(tracker.objects()[0])  # looked at its place clearly three times: not there
    memory.save(tracker, [], {})
    restored = Tracker()
    memory.restore(restored)
    assert not restored.objects()[0].present
