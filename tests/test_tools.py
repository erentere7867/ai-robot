"""Tool-argument parsing: every common way a model writes a number is read as that number."""

from __future__ import annotations

import pytest

from robot.agent.api import Pose
from robot.agent.tools import Tools, _num
from robot.eval.suite import living_room, two_rooms
from robot.sim.world import SimRobot


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


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_numbers_are_refused(value: float) -> None:
    tools = Tools(SimRobot(living_room()))
    result = tools.execute("move", {"metres": value, "direction": "forward"})
    assert not result.ok and "finite number" in result.text
    assert not tools.robot.trace.events


@pytest.mark.parametrize("steps", [["   "], [{"text": "a step"}], "first; second", ["step"] * 31, ["step"] * 30])
def test_invalid_todo_cannot_overwrite_progress(steps: object) -> None:
    tools = Tools(SimRobot(living_room()))
    assert tools.execute("todo", {"steps": ["first", "second"]}).ok
    assert tools.execute("todo_done", {"step": 1}).ok
    assert not tools.execute("todo", {"steps": steps}).ok
    assert not tools.execute("todo_done", {"step": 2.5}).ok
    assert tools.todo.open_steps() == ["second"]


def test_empty_todo_revision_clears_only_unfinished_work() -> None:
    tools = Tools(SimRobot(living_room()))
    assert tools.execute("todo", {"steps": [" first ", "second"]}).ok
    assert tools.execute("todo_done", {"step": 1}).ok
    result = tools.execute("todo", {"steps": []})
    assert result.ok and result.text == "To-do list:\n1. [x] first"
    assert tools.todo.open_steps() == []


def test_non_string_target_cannot_trigger_a_search() -> None:
    tools = Tools(SimRobot(living_room()))
    result = tools.execute("go_to", {"target": ["chair"]})
    assert not result.ok and "non-empty string" in result.text
    assert not tools.robot.trace.events


def test_honest_move_turn_and_go_toward_partial_outcomes() -> None:
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
    assert not res.ok and res.status == "partial" and "Did not move: no open floor front" in res.text


def test_successful_motion_progress_reports_measured_distance_and_turn(monkeypatch) -> None:
    from robot.agent.tools import Tools
    from robot.eval.suite import living_room
    from robot.sim.world import SimRobot

    robot = SimRobot(living_room())
    tools = Tools(robot)
    move, turn = robot.move, robot.turn
    monkeypatch.setattr(robot, "move", lambda metres, direction: move(0.83 * metres, direction))
    monkeypatch.setattr(robot, "turn", lambda degrees, direction: turn(degrees - 15, direction))

    moved = tools.execute("move", {"metres": 1, "direction": "forward"})
    turned = tools.execute("turn", {"degrees": 90, "direction": "left"})

    assert moved.ok and "This request so far: moved 0.82 m forward." in moved.text
    assert turned.ok and "moved 0.82 m forward; turned 75 degrees left; 75 degrees of turning in all." in turned.text


def test_tools_uses_robot_store_when_memory_not_passed(tmp_path) -> None:
    from robot.mind.memory import CirculatingMemory

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


@pytest.mark.parametrize(("door_open", "landmark", "scope", "approach"), [
    (True, "kitchen", "kitchen", "Approached the kitchen"),
    (False, "kitchen", "living", "Could not reach the kitchen"),
    (True, "unmapped shelf", "living", "Could not locate 'unmapped shelf'"),
])
def test_search_reports_landmark_outcome_and_scope(door_open, landmark, scope, approach, monkeypatch):
    robot = SimRobot(two_rooms(door_open=door_open))
    called = []

    def search(what, area, *, toward):
        called.append((what, area, robot.room()))
        return None

    monkeypatch.setattr(robot, "search", search)
    result = Tools(robot).execute("search", {"what": "towel", "near": landmark})
    assert result.ok and called == [("towel", "here", scope)]
    assert f"No towel found in the views checked. Search scope: {scope}." in result.text
    assert approach in result.text


def test_search_does_not_redispatch_a_body_error(monkeypatch):
    robot = SimRobot(two_rooms())
    directions = []

    def search(what, area, *, toward=""):
        directions.append(toward)
        raise TypeError("bad camera sample")

    monkeypatch.setattr(robot, "search", search)
    result = Tools(robot).execute("search", {"what": "towel", "toward": "front"})
    assert not result.ok and "bad camera sample" in result.text
    assert directions == ["front"]


def test_search_does_not_count_travel_before_the_search_as_coverage(monkeypatch):
    robot = SimRobot(two_rooms())
    tools = Tools(robot)
    robot.places["far counter"] = Pose(4.0, 0.0, 0.0)
    robot.go_to("far counter")
    robot.go_to("start")
    monkeypatch.setattr(robot, "search", lambda *args, **kwargs: None)
    result = tools.execute("search", {"what": "towel", "area": "everywhere"})
    assert result.ok and "Areas not visited closely yet: far counter." in result.text


def test_repositioning_does_not_repeat_a_search_from_the_same_landmark(monkeypatch):
    robot = SimRobot(two_rooms())
    tools = Tools(robot)
    monkeypatch.setattr(robot, "search", lambda *args, **kwargs: None)
    args = {"what": "towel", "near": "kitchen"}
    assert tools.execute("search", args).ok
    robot.go_to(Pose(-1.3, 0.0, 0.0))
    before = robot.pose()
    drives = sum(e.name == "go_to" for e in robot.trace.events)
    result = tools.execute("search", args)
    assert not result.ok and "already searched from that landmark" in result.text
    assert robot.pose() == before and sum(e.name == "go_to" for e in robot.trace.events) == drives
    assert tools.execute("search", {"what": "towel"}).ok


@pytest.mark.parametrize("anchor_retained", [True, False])
def test_search_coverage_uses_the_new_trace_after_rollover(anchor_retained, monkeypatch):
    robot = SimRobot(two_rooms())
    robot.places.update(counter=Pose(4.0, 0.0, 0.0), cabinet=Pose(-2.5, 2.4, 0.0))
    previous = [(0.0, -2.5, 0.0, 0.0), (1.0, -2.5, 0.0, 0.0)]
    robot.trace.poses = previous

    def search(*args, **kwargs):
        robot.trace.poses = previous[-1:] if anchor_retained else []
        robot.trace.poses.extend([(2.0, 4.0, 0.0, 0.0), (3.0, -2.5, 0.0, 0.0)])
        return None

    monkeypatch.setattr(robot, "search", search)
    result = Tools(robot).execute("search", {"what": "towel", "area": "everywhere"})
    assert result.ok and "No towel found in the views checked." in result.text
    assert "counter" not in result.text
    assert ("Areas not visited closely yet: cabinet." in result.text) == anchor_retained


def test_plan_revision_keeps_completed_steps_and_blocking_is_not_completion():
    tools = Tools(SimRobot(two_rooms()))
    assert tools.execute("todo", {"steps": ["Reach the doorway", "Enter the room"]}).ok
    assert tools.execute("todo_done", {"step": 1}).ok
    assert tools.execute("todo_blocked", {"step": 2, "reason": "door shut"}).ok
    assert tools.todo.open_steps() == ["Enter the room (blocked: door shut)"]
    assert not tools.execute("todo_blocked", {"step": 1, "reason": "door shut"}).ok
    result = tools.execute("todo", {"steps": ["Ask the user to open the door", "Enter the room"]})
    assert result.ok and result.text.splitlines()[1:] == [
        "1. [x] Reach the doorway", "2. [ ] Ask the user to open the door", "3. [ ] Enter the room",
    ]
    assert tools.todo.done_steps() == ["Reach the doorway"]


def test_turn_span_is_the_net_rotation_not_every_wobble():
    import math as m
    from types import SimpleNamespace

    from robot.agent.api import Pose
    from robot.agent.tools import _turn_span
    # a 90 degree turn whose samples wobble by 5 degrees on the way, and a two-turn spin
    wobbly = [m.radians(d) for d in (20, 15, 40, 35, 60, 55, 90)]
    robot = SimpleNamespace(trace=SimpleNamespace(poses=[(0.0, 0.0, 0.0, y) for y in wobbly]))
    assert abs(_turn_span(robot, 0, Pose(0, 0, 0.0), Pose(0, 0, 90.0), 90) - 90) < 1e-6
    spin = [m.radians(d) for d in range(30, 721, 30)]
    robot = SimpleNamespace(trace=SimpleNamespace(poses=[(0.0, 0.0, 0.0, y) for y in spin]))
    assert abs(_turn_span(robot, 0, Pose(0, 0, 0.0), Pose(0, 0, 0.0), 720) - 720) < 1e-6


def test_a_negative_size_is_explained_with_the_call_to_make_instead():
    res = Tools(SimRobot(living_room())).execute("move", {"metres": -1, "direction": "backward"})
    assert "metres is a size and is never negative. Call it again with metres=1" in res.text


def test_going_to_the_thing_you_are_at_says_you_did_not_move():
    tools = Tools(SimRobot(living_room()))
    assert "Reached box" in tools.execute("go_to", {"target": "box"}).text
    again = tools.execute("go_to", {"target": "box"})
    assert again.ok and again.status == "already_there" and "Already at box" in again.text
