"""Tool-argument parsing: every common way a model writes a number is read as that number."""

from __future__ import annotations

import pytest

from robot.agent.tools import _num


@pytest.mark.parametrize(("text", "expected"), [
    ("2", 2.0), ("-2", -2.0), ("1.5", 1.5), (" 3 m ", 3.0), ("90 degrees", 90.0), ("2s", 2.0),
    (".5", 0.5), ("-.5", -0.5), ("+1", 1.0), ("1.", 1.0), ("1e2", 100.0), ("2.5E-1", 0.25), ("+.5 m", 0.5),
])
def test_every_common_number_form_is_read(text: str, expected: float) -> None:
    assert _num(text) == expected


@pytest.mark.parametrize("text", ["", ".", "-", "e2", "1e", "1,5", "1.2.3", "inf", "nan", "1e999", "two", "5 km"])
def test_anything_else_is_refused(text: str) -> None:
    with pytest.raises(ValueError):
        _num(text)


def test_true_and_false_are_not_numbers() -> None:
    with pytest.raises(ValueError):
        _num(True)


def test_honest_move_turn_and_go_toward_partial_outcomes() -> None:
    from robot.agent.tools import Tools
    from robot.eval.suite import living_room
    from robot.sim.world import SimRobot

    robot = SimRobot(living_room())
    tools = Tools(robot)

    # Short move reports partial (ok=False) with measured distance
    robot.move = lambda metres, direction="forward": None  # type: ignore[method-assign]
    res = tools.execute("move", {"metres": 1.0, "direction": "forward"})
    assert not res.ok and res.status == "partial" and "0.00 m of 1 m" in res.text

    # Short turn reports partial (ok=False) with measured degrees
    robot.turn = lambda degrees, direction="left": None  # type: ignore[method-assign]
    res = tools.execute("turn", {"degrees": 90, "direction": "left"})
    assert not res.ok and res.status == "partial" and "0 of 90 degrees" in res.text

    # Short go_toward reports partial (ok=False)
    robot.go_to = lambda *a, **kw: None  # type: ignore[method-assign]
    res = tools.execute("go_toward", {"direction": "front", "metres": 2.0})
    assert not res.ok and res.status == "partial" and "ended short" in res.text


def test_tools_uses_robot_store_when_memory_not_passed(tmp_path) -> None:
    from robot.agent.tools import Tools
    from robot.eval.suite import living_room
    from robot.mind.memory import CirculatingMemory
    from robot.sim.world import SimRobot

    db = tmp_path / "store.db"
    store = CirculatingMemory(db)
    robot = SimRobot(living_room())
    robot.store = store  # type: ignore[attr-defined]
    tools = Tools(robot)
    assert tools.memory is store
    res = tools.execute("remember", {"fact": "umbrella is in the closet"})
    assert res.ok
    store.close()

    reopened = CirculatingMemory(db)
    assert any("umbrella" in f for f, _ in reopened.query_facts("where is my umbrella"))
    reopened.close()


def test_tools_remember_single_write_path_and_recall_dedupes_raw_facts(tmp_path) -> None:
    from robot.agent.tools import Tools
    from robot.eval.suite import living_room
    from robot.mind.memory import CirculatingMemory
    from robot.sim.world import SimRobot

    store = CirculatingMemory(tmp_path / "single.db")
    writes: list[str] = []
    orig = store.remember_fact

    def tracked(text: str, room: str = "", now_wall: float | None = None, *, refresh: bool = True) -> None:
        writes.append(text)
        orig(text, room=room, now_wall=now_wall, refresh=refresh)

    store.remember_fact = tracked  # type: ignore[method-assign]

    class StoreBackedRobot(SimRobot):
        persists_facts = True

        def __init__(self) -> None:
            super().__init__(living_room())
            self.store = store

        def remember(self, fact: str) -> None:
            super().remember(fact)
            self.store.remember_fact(fact, room="living room")

    robot = StoreBackedRobot()
    tools = Tools(robot, memory=store)
    assert tools.execute("remember", {"fact": "spare keys are on the couch"}).ok
    assert tools.execute("remember", {"fact": "spare keys are on the dining table"}).ok
    assert writes == ["spare keys are on the couch", "spare keys are on the dining table"]

    recalled = tools.execute("recall", {"query": "where are my spare keys?"})
    assert recalled.ok
    lines = [line for line in recalled.text.splitlines() if "spare keys" in line]
    assert lines == [
        "spare keys are on the couch (earlier)",
        "spare keys are on the dining table (most recent)",
    ]
    store.close()


def test_go_to_side_accounts_for_default_02m_gap() -> None:
    from robot.agent.api import Object, Pose
    from robot.agent.tools import ROBOT_RADIUS, Tools

    class SideStopRobot:
        def __init__(self) -> None:
            self._pose = Pose(0.0, 0.0, 0.0)
            # gap = distance - radius - ROBOT_RADIUS = 1.25 - 0.4 - 0.15 = 0.70 m (within 0.2 + 0.6)
            self._obj = Object(1, "table", (1.25, 0.0), 1.25, 0.0, "ahead", 0.4, "", 0.0)
            self.places: dict[str, Pose] = {}

        def pose(self) -> Pose:
            return self._pose

        def obj(self, oid: int) -> Object:
            assert oid == 1
            return self._obj

        def where(self, *_a, **_kw):  # noqa: ANN002, ANN003, ANN201
            return [self._obj]

        def go_to(self, _t, **_kw) -> None:  # noqa: ANN001, ANN003
            self._pose = Pose(0.5, 0.0, 0.0)

        def face(self, _t) -> None:  # noqa: ANN001
            pass

        def sees_now(self, _o: Object) -> bool:
            return True

        def user(self) -> None:
            return None

    assert abs((1.25 - 0.4 - ROBOT_RADIUS) - 0.7) < 1e-6
    res = Tools(SideStopRobot()).execute("go_to_side", {"target": "1", "side": "left"})
    assert res.ok and res.status == "reached"

