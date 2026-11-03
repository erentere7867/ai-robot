"""Drives and the autonomy controller: when (and when not) the robot wakes its model on its own."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from robot.behavior.attention import AttentionManager
from robot.behavior.autonomy import AutonomyController
from robot.behavior.drives import DriveSystem
from robot.behavior.fsm import InteractionFSM
from robot.body.hal import BodyPose, SensorReadings
from robot.cognition.agent import TurnResult
from robot.core.bus import EventBus
from robot.core.clock import ReplayClock
from robot.core.config import AutonomyConfig


# ---- drives ------------------------------------------------------------------------------------------
def test_drives_rise_over_time_and_cap():
    drives = DriveSystem()
    start = {k: d.level for k, d in drives.drives.items()}
    drives.step(60.0)
    assert drives.drives["curiosity"].level == pytest.approx(start["curiosity"] + 0.06)
    assert drives.drives["social"].level == pytest.approx(0.04)
    assert drives.drives["play"].level == pytest.approx(start["play"] + 0.03)
    assert drives.drives["energy"].level == 0.0  # driven by the battery, not by time
    assert drives.minutes_since_human == pytest.approx(1.0)
    drives.step(3600.0)
    assert all(d.level <= 1.0 for d in drives.drives.values())
    fast = DriveSystem(rate_scale=10.0)
    fast.step(60.0)
    assert fast.drives["social"].level == pytest.approx(0.4)
    drives.step(-50.0)  # a clock glitch never lowers anything
    assert drives.drives["social"].level == 1.0


def test_tools_satisfy_their_drives_and_talking_resets_loneliness():
    drives = DriveSystem()
    drives.step(1200.0)
    curiosity, play = drives.drives["curiosity"].level, drives.drives["play"].level
    log = [{"tool": "look_around", "ok": True}, {"tool": "express", "ok": True}, {"tool": "explore", "ok": False}]
    drives.on_turn(TurnResult(say="", tool_log=log), autonomous=True)
    assert drives.drives["curiosity"].level == pytest.approx(curiosity - 0.6)
    assert drives.drives["play"].level == pytest.approx(play - 0.5)
    assert drives.drives["social"].level > 0
    drives.on_turn(TurnResult(say="hi"), autonomous=False)
    assert drives.drives["social"].level == 0.0 and drives.minutes_since_human == 0.0


def test_urgent_picks_the_most_pressing_drive_and_low_battery_wins():
    drives = DriveSystem()
    assert drives.urgent() is None
    drives.drives["curiosity"].level = 0.95
    drives.drives["social"].level = 0.85
    name, feeling = drives.urgent()
    assert name == "curiosity" and "You feel curious" in feeling
    drives.drives["social"].level = 1.0
    drives.drives["curiosity"].level = 0.8
    assert drives.urgent()[0] == "social"
    drives.step(0.0, battery_pct=22.0)  # energy 0.7, only just over its 0.6 threshold
    name, feeling = drives.urgent()
    assert name == "energy" and "battery is low (22%)" in feeling and "charging dock" in feeling
    snap = drives.snapshot()
    assert snap["battery_pct"] == 22.0 and set(snap["drives"]) == {"curiosity", "social", "play", "energy"}


# ---- autonomy controller ---------------------------------------------------------------------------------
class FakeArbiter:
    def __init__(self) -> None:
        self.busy = False
        self.follow_up = False
        self.fsm = SimpleNamespace(in_follow_up_window=lambda: self.follow_up)
        self.turns: list[str] = []
        self.plans: list = []
        self.next_wake_s: float | None = None

    async def handle_user_turn(self, text, *, autonomous=False, plan=None, **_):
        assert autonomous
        self.turns.append(text)
        self.plans.append(plan)
        return TurnResult(say="", next_wake_s=self.next_wake_s)


class FakeBody:
    def __init__(self) -> None:
        self.sensor_values = SensorReadings(battery_pct=100.0)

    def pose(self) -> BodyPose:
        return BodyPose()

    def sensors(self) -> SensorReadings:
        return self.sensor_values


class Rig:
    def __init__(self, **config) -> None:
        self.clock = ReplayClock()
        self.bus = EventBus(self.clock)
        self.arbiter = FakeArbiter()
        self.body = FakeBody()
        self.audio = SimpleNamespace(is_speaking=False, _utterance_task=None)
        self.drives = DriveSystem()
        fsm = InteractionFSM(clock=self.clock)
        self.ctl = AutonomyController(
            bus=self.bus, arbiter=self.arbiter, body=self.body,
            config=AutonomyConfig(**{"idle_after_s": 20.0, "think_interval_s": 45.0, **config}),
            attention=AttentionManager(self.bus, fsm), audio=self.audio, drives=self.drives)
        self.ctl.bind()

    async def start(self) -> None:
        await self.ctl.start()

    async def tick(self, seconds: float) -> None:
        self.clock.advance(seconds)
        await self.ctl.step()
        if self.ctl._task is not None:
            await self.ctl._task

    def make_curious(self) -> None:
        self.drives.drives["curiosity"].level = 0.9


async def test_no_background_turn_while_drives_are_content():
    rig = Rig()
    await rig.start()
    for _ in range(10):
        await rig.tick(10.0)  # 100 s: curiosity rises from 0.55 but stays below 0.7
    assert rig.arbiter.turns == [] and rig.ctl.reason == "content; no reason to think"


async def test_urgent_drive_wakes_the_model_once_idle():
    rig = Rig()
    await rig.start()
    rig.make_curious()
    await rig.tick(5.0)
    assert rig.arbiter.turns == [] and rig.ctl.reason == "giving the human time to talk"
    await rig.tick(20.0)
    assert len(rig.arbiter.turns) == 1 and "You feel curious" in rig.arbiter.turns[0]
    assert rig.ctl.last_trigger == "drive:curiosity"
    await rig.tick(10.0)
    assert len(rig.arbiter.turns) == 1  # at most one decision per think interval
    await rig.tick(40.0)
    assert len(rig.arbiter.turns) == 2


@pytest.mark.parametrize("blocker", ["busy", "paused", "disabled", "speaking", "utterance", "follow_up", "estop",
                                     "picked_up", "stopped"])
async def test_never_wakes_while_busy_paused_speaking_or_unsafe(blocker):
    rig = Rig()
    await rig.start()
    rig.make_curious()
    if blocker == "busy":
        rig.arbiter.busy = True
    elif blocker == "paused":
        await rig.ctl.set_mode(paused=True)
    elif blocker == "disabled":
        await rig.ctl.set_mode(enabled=False)
    elif blocker == "speaking":
        rig.audio.is_speaking = True
    elif blocker == "utterance":
        rig.audio._utterance_task = asyncio.get_running_loop().create_future()
    elif blocker == "follow_up":
        rig.arbiter.follow_up = True
    elif blocker == "estop":
        rig.body.sensor_values.estop_active = True
    elif blocker == "picked_up":
        rig.body.sensor_values.picked_up = True
    elif blocker == "stopped":
        await rig.ctl.stop()
    for _ in range(5):
        await rig.tick(60.0)
    assert rig.arbiter.turns == []


async def test_user_stop_pauses_until_resumed():
    rig = Rig()
    await rig.start()
    rig.make_curious()
    await rig.bus.emit("action.cancel", "test", {"reason": "user_stop_command"})
    assert rig.ctl.paused
    await rig.tick(120.0)
    assert rig.arbiter.turns == []
    await rig.ctl.set_mode(paused=False)
    await rig.tick(1.0)
    assert len(rig.arbiter.turns) == 1


async def test_rules_policy_acts_on_drives_without_the_model():
    rig = Rig(policy="rules")
    await rig.start()
    rig.make_curious()
    await rig.tick(25.0)
    assert rig.arbiter.plans == [[("explore", {})]]
    rig.drives.drives["curiosity"].level = 0.0
    rig.drives.drives["play"].level = 0.95
    await rig.tick(50.0)
    assert rig.arbiter.plans[-1][0][0] == "express"


async def test_model_policy_wakes_the_model_for_drives():
    rig = Rig(policy="model")
    await rig.start()
    rig.make_curious()
    await rig.tick(25.0)
    assert rig.arbiter.plans == [None]


