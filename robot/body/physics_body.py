"""Rigid-body MuJoCo simulation of companion_r1 with motor/servo actuators, odometry and sensor noise.

`PhysicsBody` is an optional drop-in replacement for the kinematic `SimBody`. Commands drive the wheel
velocity and head servo actuators of `config/companion_r1_physics.xml`; a 100 Hz "firmware" loop inside
the 2 ms physics step closes the loop on the odometry estimate (quantised wheel encoders + noisy IMU
yaw), as motor firmware would on hardware. `pose()` is that estimate; `true_pose()` is MuJoCo ground
truth. Chassis contact or stalled wheels raise a "bumper" reflex. Movable sandbox objects are free
bodies that the robot can push; their new positions are written back with `sandbox.move_object()`.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from robot.body.hal import BodyCapabilities, BodyCommand, BodyPose, CommandResult, SensorReadings
from robot.body.sim_body import SimBody, _normalize_angle_deg
from robot.core.bus import EventBus
from robot.core.clock import RealClock
from robot.core.config import PROJECT_ROOT
from robot.core.events import EventPriority
from robot.core.service import ServiceHealth

logger = logging.getLogger(__name__)

PHYSICS_XML_PATH = PROJECT_ROOT / "config" / "companion_r1_physics.xml"
_WHEELS = ("fl", "fr", "rl", "rr")
_RAY_GROUPS = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)  # group 3 = the robot's own geoms
_TOF_FAN_DEG = (-12.0, -6.0, 0.0, 6.0, 12.0)
_TOF_MAX_MM = 2500.0
_PARK_XY_M = (4.0, 4.0)  # outside the walls; the floor plane is infinite


@dataclass(frozen=True)
class SensorNoise:
    """1-sigma sensor errors. Systematic terms are drawn once per body from its seed."""

    wheel_scale: float = 0.01  # relative wheel-radius calibration error, per side
    gyro_bias_deg_s: float = 0.05
    gyro_noise_deg_s: float = 0.2  # white noise per 100 Hz IMU sample
    tof_mm: float = 3.0

    @classmethod
    def off(cls) -> SensorNoise:
        return cls(0.0, 0.0, 0.0, 0.0)


@dataclass
class _BaseMotion:
    """Drive/spin set point tracked by the firmware loop in the odometry frame."""

    kind: str  # "drive" (mm) | "spin" (deg)
    sign: float
    target: float
    speed: float
    x0: float
    y0: float
    heading0: float
    yaw0: float
    timeout_s: float
    rate: float = 0.0
    rate_i: float = 0.0
    arrived: bool = False
    settle_s: float = 0.0
    elapsed_s: float = 0.0
    done: bool = False
    timed_out: bool = False


@dataclass
class _PhysObject:
    """Physics body mirroring one SimulationSandbox object."""

    body: str
    free: bool
    rest_z: float
    dynamic: bool
    signature: tuple | None = None
    in_use: bool = False
    sandbox_xy: tuple[float, float] | None = None
    physics_xy: tuple[float, float] | None = None
    qadr: int = -1
    vadr: int = -1
    mocap: int = -1


def _signature(obj: dict[str, Any]) -> tuple:
    return (obj["shape"], obj["width_mm"], obj["depth_mm"], obj["height_mm"], bool(obj.get("movable", True)))


class PhysicsBody(SimBody):
    """MuJoCo rigid-body companion_r1 with skid-steer wheel motors, head servo, odometry and bumper."""

    name = "physics_body"

    LINEAR_ACCEL_MM_S2 = 800.0
    LINEAR_DECEL_MM_S2 = 600.0
    ANGULAR_ACCEL_DEG_S2 = 360.0
    CREEP_MM_S = 15.0
    CREEP_DEG_S = 8.0
    DRIVE_TOLERANCE_MM = 0.8
    SPIN_TOLERANCE_DEG = 0.15
    HEADING_GAIN = 4.0  # (deg/s) per deg of heading error
    CROSS_TRACK_GAIN_DEG_MM = 0.15
    RATE_P = 0.5
    RATE_I = 3.0
    TURN_SLIP_FACTOR = 2.4  # effective/nominal track width of the skid-steer base (feed-forward only)
    HEAD_TOLERANCE_DEG = 0.5
    BUMPER_FORCE_N = 6.0
    BUMPER_HOLD_S = 0.04
    STALL_HOLD_S = 0.3
    FIRMWARE_DT_S = 0.01
    MAX_CATCHUP_S = 0.5
    IDLE_PERIOD_S = 0.05
    OBJECT_SYNC_MM = 5.0

    def __init__(
        self,
        bus: EventBus,
        capabilities: BodyCapabilities | None = None,
        *,
        step_dt_s: float = 0.05,
        sandbox: Any | None = None,
        seed: int = 0,
        battery_idle_pct_per_min: float = 0.5,
        battery_pct_per_m: float = 1.0,
        charge_pct_per_min: float = 20.0,
        noise: SensorNoise | None = None,
        xml_path: Path | str | None = None,
    ) -> None:
        self._data: mujoco.MjData | None = None
        super().__init__(bus, capabilities, step_dt_s=step_dt_s)
        for key, value in (
            ("battery_idle_pct_per_min", battery_idle_pct_per_min),
            ("battery_pct_per_m", battery_pct_per_m),
            ("charge_pct_per_min", charge_pct_per_min),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        self.battery_idle_pct_per_min = float(battery_idle_pct_per_min)
        self.battery_pct_per_m = float(battery_pct_per_m)
        self.charge_pct_per_min = float(charge_pct_per_min)
        self.noise = noise or SensorNoise()
        self._rng = np.random.default_rng(seed)
        self._wheel_scale = tuple(1.0 + self._rng.normal(0.0, self.noise.wheel_scale) for _ in range(2))
        self._gyro_bias = float(self._rng.normal(0.0, self.noise.gyro_bias_deg_s))
        self._r_nominal_m = self._caps.wheel_diameter_mm / 2000.0
        self._half_track_m = self._caps.track_width_mm * self.TURN_SLIP_FACTOR / 2000.0

        # MuJoCo cannot open non-ASCII paths on Windows, so always parse from a string.
        xml = Path(xml_path or PHYSICS_XML_PATH).read_text(encoding="utf-8")
        self._spec = mujoco.MjSpec.from_string(xml)
        self._model = self._spec.compile()
        self._data = mujoco.MjData(self._model)
        self._xml_objects: dict[str, _PhysObject] = {}
        self._dyn_objects: dict[str, _PhysObject] = {}
        self._dyn_serial = 0
        for i in range(self._model.nbody):
            body = self._model.body(i)
            if body.name.startswith("obj_"):
                free = int(self._model.body_jntnum[i]) > 0
                z = float(self._model.body_pos[i][2])
                self._xml_objects[body.name[4:]] = _PhysObject(body.name, free, z, dynamic=False, in_use=True)
        self._index_model()
        self._fw_every = max(1, round(self.FIRMWARE_DT_S / self._model.opt.timestep))
        self._contact_force = np.zeros(6)
        self._ray_geom = np.zeros(1, dtype=np.int32)

        self._motion: _BaseMotion | None = None
        self._fw_v = 0.0  # speed-control set point ramped by the firmware (mm/s, deg/s)
        self._fw_w = 0.0
        self._fw_w_i = 0.0
        self._fw_blocked = False
        self._fw_reflex: tuple[str, str] | None = None
        self._halt_serial = 0
        self._head_setpoint = 0.0
        self._head_target = 0.0
        self._head_speed = self._caps.joints["head_tilt"].max_speed_deg_s
        self._head_settle_s = 0.0
        self._bump_s = 0.0
        self._stall_s = 0.0
        self._phase = 0
        self._time_debt = 0.0
        self._sim_time = 0.0
        self._awake_until = 0.0
        self._advancing = False
        self._perf_wall_s = 0.0
        self._perf_sim_s = 0.0
        self._idle_task: asyncio.Task | None = None
        self._sandbox: Any | None = None
        self._battery_travel_mm = 0.0

        # Let the tyres settle onto the floor before odometry is zeroed.
        mujoco.mj_step(self._model, self._data, nstep=round(0.3 / self._model.opt.timestep))
        self._reset_odometry()
        self._clock_ref = self._battery_ref = self.bus.clock.monotonic()
        if sandbox is not None:
            self.bind_sandbox(sandbox)

    # -- odometry estimate (what pose() reports) ------------------------------------------------

    @property
    def _pose(self) -> BodyPose:
        return self._odom

    @_pose.setter
    def _pose(self, value: BodyPose) -> None:
        """Assigning the pose (the dashboard's scene reset does this) also teleports the robot."""
        self._odom = value
        if self._data is not None:
            self.reset_pose(value)

    def reset_pose(self, pose: BodyPose | None = None) -> None:
        """Teleport the physical robot (and head) to `pose`, at rest, and re-zero odometry there.

        A base or head motion in progress is interrupted and reports "cancelled".
        """
        pose = pose or BodyPose()
        m, d = self._model, self._data
        a, v = self._base_qadr, self._base_vadr
        yaw = math.radians(pose.heading_deg)
        d.qpos[a : a + 3] = (pose.x_mm / 1000.0, pose.y_mm / 1000.0, self._base_rest_z)
        d.qpos[a + 3 : a + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        d.qvel[v : v + 6] = 0.0
        d.qvel[self._wheel_vadr] = 0.0
        spec = self._caps.joints["head_tilt"]
        tilt = max(spec.min_deg, min(spec.max_deg, pose.head_tilt_deg))
        d.qpos[self._head_qadr] = math.radians(tilt)
        d.qvel[self._head_vadr] = 0.0
        self._halt_actuators()
        self._head_setpoint = self._head_target = tilt
        d.ctrl[self._servo_id] = math.radians(tilt)
        mujoco.mj_forward(m, d)
        self._reset_odometry()
        self._sync_world()
        self._wake()

    def _reset_odometry(self) -> None:
        truth = self.true_pose()
        self._odom = BodyPose(
            x_mm=truth.x_mm, y_mm=truth.y_mm, heading_deg=truth.heading_deg, head_tilt_deg=self._head_setpoint
        )
        self._yaw_est = self._imu_yaw_ref = truth.heading_deg
        self._enc_ticks = self._encoder_ticks()
        self._sensors.wheel_ticks_left, self._sensors.wheel_ticks_right = self._enc_ticks
        self._sensors.imu_yaw_deg = truth.heading_deg
        self._sensors.imu_pitch_deg = self._head_setpoint

    def _true_yaw_deg(self) -> float:
        a = self._base_qadr
        w, x, y, z = (float(c) for c in self._data.qpos[a + 3 : a + 7])
        return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def true_pose(self) -> BodyPose:
        """Ground-truth pose from the physics state (never available to the robot itself)."""
        q, qv = self._data.qpos, self._data.qvel
        a, v = self._base_qadr, self._base_vadr
        yaw = math.radians(self._true_yaw_deg())
        forward = float(qv[v]) * math.cos(yaw) + float(qv[v + 1]) * math.sin(yaw)
        return BodyPose(
            x_mm=float(q[a]) * 1000.0,
            y_mm=float(q[a + 1]) * 1000.0,
            heading_deg=_normalize_angle_deg(math.degrees(yaw)),
            head_tilt_deg=math.degrees(float(q[self._head_qadr])),
            linear_vel_mm_s=forward * 1000.0,
            angular_vel_deg_s=math.degrees(float(qv[v + 5])),
        )

    # -- lifecycle and HAL overrides ------------------------------------------------------------

    async def start(self) -> None:
        await super().start()
        # Keep the world evolving between commands in live runs; replay clocks only advance on demand.
        if isinstance(self.bus.clock, RealClock) and self._idle_task is None:
            self._idle_task = asyncio.create_task(self._idle_loop(), name="physics_body:idle")

    async def stop(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._idle_task
            self._idle_task = None
        await super().stop()

    async def health(self) -> ServiceHealth:
        health = await super().health()
        health.details = {
            **health.details,
            "true_pose": self.true_pose().to_dict(),
            "battery_pct": round(self._sensors.battery_pct, 1),
            "physics_ms_per_sim_s": round(self.physics_ms_per_sim_s(), 2),
        }
        return health

    def physics_ms_per_sim_s(self) -> float:
        """Average wall-clock milliseconds spent stepping physics per simulated second."""
        return 1000.0 * self._perf_wall_s / self._perf_sim_s if self._perf_sim_s > 0 else 0.0

    def sensors(self) -> SensorReadings:
        self._refresh_sensors()
        return dataclasses.replace(self._sensors)

    def set_estop(self, active: bool) -> None:
        super().set_estop(active)
        if active:
            self._halt_actuators()

    async def cancel(self, command_id: str | None = None) -> None:
        if command_id is None or command_id == self._active_cmd_id:
            self._halt_actuators()
        await super().cancel(command_id)

    def attach_sensor_model(self, model) -> None:
        """Sample the sandbox raycasts with ground truth and keep its objects in sync with physics."""
        super().attach_sensor_model(model)
        if self._sandbox is None and hasattr(model, "list_objects") and hasattr(model, "move_object"):
            self.bind_sandbox(model)

    def bind_sandbox(self, sandbox: Any) -> None:
        """Mirror the sandbox object list into physics and report pushed objects back to it."""
        self._sandbox = sandbox
        self._sync_world()

    def set_sandbox_sensors(self, front_tof_mm: float | None = None, cliff: bool | None = None) -> None:
        """Accept pushed raycast values only without an attached sensor model.

        With a model attached, every sensor read samples it with `true_pose()`, so values the
        sandbox computes from the odometry estimate (its `body.pose` subscriber) are ignored.
        """
        if self._sensor_model is None:
            super().set_sandbox_sensors(front_tof_mm=front_tof_mm, cliff=cliff)

    def _refresh_sensors(self) -> None:
        if self._data is None:
            return
        truth = self.true_pose()
        if self._sensor_model is not None:
            tof = float(self._sensor_model.compute_front_tof_mm(truth))
            cliff: bool | None = bool(self._sensor_model.check_cliff_triggered(truth))
        else:
            tof, cliff = self._raycast_tof_mm(), None
        if self.noise.tof_mm > 0:
            tof = max(0.0, tof + float(self._rng.normal(0.0, self.noise.tof_mm)))
        super().set_sandbox_sensors(front_tof_mm=round(tof, 1), cliff=cliff)
        self._update_battery(truth)

    def _update_battery(self, truth: BodyPose) -> None:
        """Idle drain per minute plus drain per metre of wheel travel; charging while docked."""
        now = self.bus.clock.monotonic()
        dt, self._battery_ref = max(0.0, now - self._battery_ref), now
        travel_m, self._battery_travel_mm = self._battery_travel_mm / 1000.0, 0.0
        is_charging = getattr(self._sensor_model, "is_charging", None)
        charging = bool(is_charging(truth)) if callable(is_charging) else False
        pct = self._sensors.battery_pct - self.battery_pct_per_m * travel_m
        if charging:
            pct += self.charge_pct_per_min * dt / 60.0
        else:
            pct -= self.battery_idle_pct_per_min * dt / 60.0
        self._sensors.battery_pct = min(100.0, max(0.0, pct))
        self._sensors.charging = charging

    def _sensor_dict(self) -> dict[str, Any]:
        sensors = self._sensors.to_dict()
        sensors.setdefault("battery_pct", round(self._sensors.battery_pct, 1))
        sensors.setdefault("charging", self._sensors.charging)
        return sensors

    async def _emit_pose(self) -> None:
        self._refresh_sensors()
        await self.bus.emit(
            "body.pose",
            source=self.name,
            payload={
                "command_id": self._active_cmd_id,
                "pose": self._pose.to_dict(),
                "sensors": self._sensor_dict(),
                "true_pose": self.true_pose().to_dict(),
            },
            priority=EventPriority.TELEMETRY,
        )

    async def _safety_result(
        self, cmd: BodyCommand, *, base: bool = False, forward: bool = False
    ) -> CommandResult | None:
        """SimBody's reflex checks on freshly sampled sensors; a reflex also de-energises the motors.

        Collisions are not taken from the sandbox geometry: physical contact is the bumper.
        """
        self._refresh_sensors()
        reason = None
        if self._sensors.estop_active:
            reason = ("estop_active", "Emergency stop is active")
        elif self._cancel_requested:
            return self._cancelled_result(cmd)
        elif base and self._sensors.picked_up:
            reason = ("picked_up", "Robot was picked up")
        elif base and self._sensors.cliff_triggered:
            reason = ("cliff_detected", "Cliff sensor triggered")
        elif forward and (
            not math.isfinite(self._sensors.front_tof_mm)
            or self._sensors.front_tof_mm <= self._caps.min_front_tof_stop_mm
        ):
            reason = ("obstacle_stop", "Front obstacle or invalid range reading")
        if reason:
            self._halt_actuators()
            await self._emit_reflex(*reason)
            await self._emit_pose()
            return CommandResult(cmd.id, cmd.tool, False, "reflex_stopped", reason[1], self._pose.to_dict())
        return None

    async def _move(self, cmd: BodyCommand, amount: float, speed: float, kind: str) -> CommandResult | None:
        """Hand a set point to the firmware loop and supervise it at the control tick rate."""
        if abs(amount) <= 1e-8:
            return None
        base = kind != "head"
        forward = kind == "drive" and amount > 0
        self.advance_to_now()
        self._fw_reflex = None
        if kind == "head":
            spec = self._caps.joints["head_tilt"]
            self._head_target = max(spec.min_deg, min(spec.max_deg, self._head_setpoint + amount))
            self._head_speed = speed
            self._head_settle_s = 0.0
        else:
            self._bump_s = self._stall_s = 0.0
            self._motion = _BaseMotion(
                kind=kind,
                sign=-1.0 if amount < 0 else 1.0,
                target=abs(amount),
                speed=speed,
                x0=self._odom.x_mm,
                y0=self._odom.y_mm,
                heading0=self._odom.heading_deg,
                yaw0=self._yaw_est,
                timeout_s=3.0 * abs(amount) / speed + 3.0,
            )
        motion = self._motion
        serial = self._halt_serial
        try:
            while True:
                result = await self._safety_result(cmd, base=base, forward=forward)
                if result:
                    return result
                if self._head_done() if kind == "head" else motion.done:
                    break
                await self.bus.clock.sleep(self.step_dt_s)
                self.advance_to_now()
                result = await self._firmware_reflex_result(cmd)
                if result:
                    return result
                result = await self._safety_result(cmd, base=base, forward=forward)
                if result:
                    return result
                if self._halt_serial != serial:  # interrupted by reset_pose()
                    return self._cancelled_result(cmd)
                await self._emit_pose()
            if motion is not None and motion.timed_out:
                return CommandResult(
                    cmd.id,
                    cmd.tool,
                    False,
                    "failed",
                    "Motion did not converge: wheels slipping or base obstructed.",
                    self._pose.to_dict(),
                )
            return None
        finally:
            # Also reached when the awaiting task is cancelled: never leave the motors energised.
            if kind == "head":
                self._head_target = self._head_setpoint
            elif self._motion is motion:
                self._motion = None
                self._data.ctrl[self._motor_ids] = 0.0
            self._wake()

    async def update_motion(self) -> str | None:
        """Run the physics and firmware up to now and report the reflex limiting speed control, if any."""
        if self._active_cmd_id is not None:
            return None
        self.advance_to_now()
        self._refresh_sensors()
        reason: tuple[str, str] | None = None
        if self._fw_reflex is not None:
            reason, self._fw_reflex = self._fw_reflex, None
        elif self._sensors.estop_active:
            reason = ("estop_active", "Emergency stop is active")
        elif self._sensors.cliff_triggered or self._sensors.picked_up:
            reason = ("cliff_detected", "Cliff sensor triggered")
        elif self._fw_blocked:
            reason = ("obstacle_stop", "Front obstacle or invalid range reading")
        if reason is not None and reason[0] != "obstacle_stop":
            self._halt_actuators()
        if reason is not None and reason[0] != self._vel_reflex:
            await self._emit_reflex(*reason)
        self._vel_reflex = reason[0] if reason else None
        if self._velocity_active() or reason is not None:
            await self._emit_pose()
        return reason[0] if reason else None

    async def _firmware_reflex_result(self, cmd: BodyCommand) -> CommandResult | None:
        if self._fw_reflex is None:
            return None
        reflex, reason = self._fw_reflex
        self._fw_reflex = None
        self._refresh_sensors()
        await self._emit_reflex(reflex, reason)
        await self._emit_pose()
        return CommandResult(
            cmd.id, cmd.tool, False, "reflex_stopped", reason, self._pose.to_dict(), {"reflex": reflex}
        )

    def _head_done(self) -> bool:
        if self._head_setpoint != self._head_target:
            return False
        error = abs(math.degrees(float(self._data.qpos[self._head_qadr])) - self._head_target)
        return error <= self.HEAD_TOLERANCE_DEG or self._head_settle_s >= 0.3

    def _halt_actuators(self) -> None:
        self._halt_serial += 1
        self._motion = None
        self._vel_target, self._vel_until = (0.0, 0.0), self.bus.clock.monotonic()
        self._fw_v = self._fw_w = self._fw_w_i = 0.0
        self._head_target = self._head_setpoint
        if self._data is not None:
            self._data.ctrl[self._motor_ids] = 0.0

    # -- physics stepping ------------------------------------------------------------------------

    async def _idle_loop(self) -> None:
        while True:
            await asyncio.sleep(self.IDLE_PERIOD_S)
            if self._active_cmd_id is None:
                try:
                    self.advance_to_now()
                except Exception:
                    logger.exception("Idle physics step failed")

    def _wake(self, seconds: float = 1.0) -> None:
        self._awake_until = max(self._awake_until, self._sim_time + seconds)

    def advance_to_now(self) -> None:
        """Step physics by the bus-clock time elapsed since the last call (catch-up is bounded).

        Commands call this every control tick, and with a `RealClock` an idle task calls it
        between commands. With a `ReplayClock` the world only evolves when this is called.
        """
        if self._advancing:
            return
        now = self.bus.clock.monotonic()
        elapsed, self._clock_ref = now - self._clock_ref, now
        if elapsed > 0:
            self._time_debt = min(self._time_debt + elapsed, self.MAX_CATCHUP_S)
        dt = self._model.opt.timestep
        steps = int(self._time_debt / dt + 1e-9)
        self._time_debt -= steps * dt
        self._advancing = True
        try:
            if steps > 0:
                if self._at_rest():
                    self._sim_time += steps * dt
                else:
                    started = time.perf_counter()
                    self._step_physics(steps)
                    self._perf_wall_s += time.perf_counter() - started
                    self._perf_sim_s += steps * dt
            self._sync_world()
        finally:
            self._advancing = False

    def _at_rest(self) -> bool:
        return (
            self._motion is None
            and not self._velocity_active()
            and self._head_setpoint == self._head_target
            and self._sim_time >= self._awake_until
            and float(np.max(np.abs(self._data.qvel))) < 1e-3
        )

    def _step_physics(self, steps: int) -> None:
        m, d = self._model, self._data
        while steps > 0:
            chunk = min(steps, self._fw_every - self._phase)
            mujoco.mj_step(m, d, nstep=chunk)
            steps -= chunk
            self._phase += chunk
            self._sim_time += chunk * m.opt.timestep
            if self._phase >= self._fw_every:
                self._phase = 0
                self._firmware_tick(self._fw_every * m.opt.timestep)

    # -- 100 Hz firmware: odometry, motion control, head servo, bumper -----------------------------

    def _encoder_ticks(self) -> tuple[int, int]:
        q = self._data.qpos
        fl, fr, rl, rr = (float(q[i]) for i in self._wheel_qadr)
        per_rad = self._caps.encoder_ticks_per_rev / (2.0 * math.pi)
        return round(0.5 * (fl + rl) * per_rad), round(0.5 * (fr + rr) * per_rad)

    def _firmware_tick(self, dt: float) -> None:
        d = self._data
        ticks = self._encoder_ticks()
        mm_per_tick = math.pi * self._caps.wheel_diameter_mm / self._caps.encoder_ticks_per_rev
        dl = (ticks[0] - self._enc_ticks[0]) * mm_per_tick * self._wheel_scale[0]
        dr = (ticks[1] - self._enc_ticks[1]) * mm_per_tick * self._wheel_scale[1]
        self._enc_ticks = ticks
        self._battery_travel_mm += 0.5 * (abs(dl) + abs(dr))
        # IMU reports the mean rate over the sample (internally integrated at a high rate).
        yaw = self._true_yaw_deg()
        gyro = _normalize_angle_deg(yaw - self._imu_yaw_ref) / dt + self._gyro_bias
        self._imu_yaw_ref = yaw
        if self.noise.gyro_noise_deg_s > 0:
            gyro += float(self._rng.normal(0.0, self.noise.gyro_noise_deg_s))
        # Zero-velocity gating: a stationary robot does not integrate gyro bias into its heading.
        if self._motion is not None or dl or dr or abs(gyro) > 1.0:
            dyaw = gyro * dt
        else:
            dyaw = gyro = 0.0
        ds = 0.5 * (dl + dr)
        mid = math.radians(self._yaw_est + 0.5 * dyaw)
        self._yaw_est += dyaw
        odom = self._odom
        odom.x_mm += ds * math.cos(mid)
        odom.y_mm += ds * math.sin(mid)
        odom.heading_deg = _normalize_angle_deg(self._yaw_est)
        odom.linear_vel_mm_s += 0.3 * (ds / dt - odom.linear_vel_mm_s)
        odom.angular_vel_deg_s += 0.3 * (gyro - odom.angular_vel_deg_s)
        self._sensors.wheel_ticks_left, self._sensors.wheel_ticks_right = ticks
        self._sensors.imu_yaw_deg = odom.heading_deg

        v_left, v_right = self._motion_control(dt, gyro)
        d.ctrl[self._left_motors] = v_left / self._r_nominal_m
        d.ctrl[self._right_motors] = v_right / self._r_nominal_m
        self._servo_tick(dt)
        self._monitor_bumper(dt, v_left, v_right)

    def _velocity_active(self) -> bool:
        return self.bus.clock.monotonic() < self._vel_until or abs(self._fw_v) > 1e-3 or abs(self._fw_w) > 1e-3

    def _velocity_control(self, dt: float, gyro_deg_s: float) -> tuple[float, float]:
        """Speed control: ramp to the commanded speeds, hold the turn rate on the gyro (watchdog included)."""
        fresh = self.bus.clock.monotonic() < self._vel_until
        target_v, target_w = self._vel_target if fresh else (0.0, 0.0)
        tof = self._sensors.front_tof_mm
        self._fw_blocked = target_v > 0 and (not math.isfinite(tof) or tof <= self._caps.min_front_tof_stop_mm)
        if self._fw_blocked:
            target_v = 0.0
        step_v, step_w = self.LINEAR_ACCEL_MM_S2 * dt, self.ANGULAR_ACCEL_DEG_S2 * dt
        self._fw_v += max(-step_v, min(step_v, target_v - self._fw_v))
        self._fw_w += max(-step_w, min(step_w, target_w - self._fw_w))
        if abs(self._fw_v) < 1e-3 and abs(self._fw_w) < 1e-3 and not fresh:
            self._fw_v = self._fw_w = self._fw_w_i = 0.0
            return 0.0, 0.0
        error = self._fw_w - gyro_deg_s
        self._fw_w_i = max(-30.0, min(30.0, self._fw_w_i + error * dt))
        omega = self._fw_w + self.RATE_P * error + self.RATE_I * self._fw_w_i
        turn = math.radians(omega) * self._half_track_m
        return self._fw_v / 1000.0 - turn, self._fw_v / 1000.0 + turn

    def _motion_control(self, dt: float, gyro_deg_s: float) -> tuple[float, float]:
        """Trapezoidal distance/heading tracking on the odometry estimate -> wheel surface speeds (m/s)."""
        mo = self._motion
        if mo is None:
            return self._velocity_control(dt, gyro_deg_s)
        odom = self._odom
        linear, omega = 0.0, 0.0
        if mo.kind == "drive":
            h0 = math.radians(mo.heading0)
            dx, dy = odom.x_mm - mo.x0, odom.y_mm - mo.y0
            remaining = mo.target - mo.sign * (dx * math.cos(h0) + dy * math.sin(h0))
            lateral = -dx * math.sin(h0) + dy * math.cos(h0)
            if not mo.arrived and remaining <= self.DRIVE_TOLERANCE_MM:
                mo.arrived = True
            if not mo.arrived:
                brake = math.sqrt(2.0 * self.LINEAR_DECEL_MM_S2 * remaining)
                ramp = min(mo.rate + self.LINEAR_ACCEL_MM_S2 * dt, brake)
                mo.rate = min(mo.speed, max(self.CREEP_MM_S, ramp))
                correction = max(-8.0, min(8.0, self.CROSS_TRACK_GAIN_DEG_MM * lateral))
                heading_ref = mo.heading0 - mo.sign * correction
                omega = self.HEADING_GAIN * _normalize_angle_deg(heading_ref - self._yaw_est)
                omega = max(-45.0, min(45.0, omega))
                linear = mo.sign * mo.rate
        else:
            remaining = mo.target - mo.sign * (self._yaw_est - mo.yaw0)
            if not mo.arrived and remaining <= self.SPIN_TOLERANCE_DEG:
                mo.arrived = True
            if not mo.arrived:
                brake = math.sqrt(2.0 * self.ANGULAR_ACCEL_DEG_S2 * remaining)
                ramp = min(mo.rate + self.ANGULAR_ACCEL_DEG_S2 * dt, brake)
                mo.rate = min(mo.speed, max(self.CREEP_DEG_S, ramp))
                command = mo.sign * mo.rate
                error = command - gyro_deg_s
                mo.rate_i = max(-30.0, min(30.0, mo.rate_i + error * dt))
                omega = command + self.RATE_P * error + self.RATE_I * mo.rate_i
        mo.elapsed_s += dt
        if mo.arrived:
            mo.settle_s += dt
            still = abs(odom.linear_vel_mm_s) < 5.0 and abs(gyro_deg_s) < 3.0
            mo.done = still or mo.settle_s >= 0.5
        elif mo.elapsed_s >= mo.timeout_s:
            mo.done = mo.timed_out = True
        if mo.arrived or mo.done:
            return 0.0, 0.0
        turn = math.radians(omega) * self._half_track_m
        return linear / 1000.0 - turn, linear / 1000.0 + turn

    def _servo_tick(self, dt: float) -> None:
        step = self._head_speed * dt
        delta = self._head_target - self._head_setpoint
        if abs(delta) <= step:
            self._head_setpoint = self._head_target
            self._head_settle_s += dt
        else:
            self._head_setpoint += math.copysign(step, delta)
        self._data.ctrl[self._servo_id] = math.radians(self._head_setpoint)
        # A hobby servo has no position feedback: the estimate is the commanded angle.
        self._odom.head_tilt_deg = self._head_setpoint
        self._sensors.imu_pitch_deg = self._head_setpoint

    def _monitor_bumper(self, dt: float, v_left: float, v_right: float) -> None:
        """Latch a bumper reflex on sustained chassis contact force or on stalled drive wheels."""
        if self._motion is None and not self._velocity_active():
            self._bump_s = self._stall_s = 0.0
            return
        force = self._external_contact_force_n()
        self._bump_s = self._bump_s + dt if force >= self.BUMPER_FORCE_N else 0.0
        qv = self._data.qvel
        fl, fr, rl, rr = (float(qv[i]) * self._r_nominal_m for i in self._wheel_vadr)
        stalled = any(
            abs(cmd) > 0.04 and abs(meas) < 0.25 * abs(cmd)
            for cmd, meas in ((v_left, 0.5 * (fl + rl)), (v_right, 0.5 * (fr + rr)))
        )
        self._stall_s = self._stall_s + dt if stalled else 0.0
        if self._bump_s >= self.BUMPER_HOLD_S:
            self._fw_reflex = ("bumper", f"Bumper contact ({force:.0f} N): pushing against an obstacle")
        elif self._stall_s >= self.STALL_HOLD_S:
            self._fw_reflex = ("bumper", "Drive wheels stalled against an obstacle")
        else:
            return
        self._halt_actuators()

    def _external_contact_force_n(self) -> float:
        d = self._data
        if d.ncon == 0:
            return 0.0
        geoms = d.contact.geom[: d.ncon]
        total = 0.0
        for i in np.nonzero((geoms == self._chassis_gid).any(axis=1))[0]:
            g1, g2 = geoms[i]
            if self._floor_gid in (g1, g2):
                continue
            mujoco.mj_contactForce(self._model, d, int(i), self._contact_force)
            total += abs(float(self._contact_force[0]))
        return total

    def _raycast_tof_mm(self) -> float:
        """Front ToF from a 5-ray fan (~24 deg field of view) against physics geometry."""
        d = self._data
        yaw = math.radians(self._true_yaw_deg())
        origin = d.site_xpos[self._tof_sid]
        nearest = _TOF_MAX_MM / 1000.0
        for offset in _TOF_FAN_DEG:
            a = yaw + math.radians(offset)
            vec = np.array([math.cos(a), math.sin(a), 0.0])
            dist = mujoco.mj_ray(self._model, d, origin, vec, _RAY_GROUPS, 1, -1, self._ray_geom)
            if 0.0 <= dist < nearest:
                nearest = dist
        return round(max(10.0, min(_TOF_MAX_MM, nearest * 1000.0)), 1)

    # -- sandbox object synchronisation ---------------------------------------------------------

    def _index_model(self) -> None:
        m = self._model
        base = m.joint("base_free")
        self._base_qadr, self._base_vadr = int(base.qposadr[0]), int(base.dofadr[0])
        self._base_rest_z = float(m.qpos0[self._base_qadr + 2])
        wheels = [m.joint(f"{w}_spin") for w in _WHEELS]
        self._wheel_qadr = [int(j.qposadr[0]) for j in wheels]
        self._wheel_vadr = [int(j.dofadr[0]) for j in wheels]
        head = m.joint("head_tilt")
        self._head_qadr, self._head_vadr = int(head.qposadr[0]), int(head.dofadr[0])
        self._motor_ids = [m.actuator(f"motor_{w}").id for w in _WHEELS]
        self._left_motors = [self._motor_ids[0], self._motor_ids[2]]
        self._right_motors = [self._motor_ids[1], self._motor_ids[3]]
        self._servo_id = m.actuator("servo_head").id
        self._chassis_gid = m.geom("chassis").id
        self._floor_gid = m.geom("floor").id
        self._tof_sid = m.site("front_tof").id
        for obj in (*self._xml_objects.values(), *self._dyn_objects.values()):
            body = m.body(obj.body)
            obj.mocap = int(m.body_mocapid[body.id])
            if obj.free:
                jid = int(m.body_jntadr[body.id])
                obj.qadr, obj.vadr = int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid])

    def _active_objects(self) -> dict[str, _PhysObject]:
        active = {obj_id: obj for obj_id, obj in self._xml_objects.items() if obj.in_use}
        active.update(self._dyn_objects)
        return active

    def _object_xy_mm(self, obj: _PhysObject) -> tuple[float, float]:
        if obj.free:
            q = self._data.qpos
            return float(q[obj.qadr]) * 1000.0, float(q[obj.qadr + 1]) * 1000.0
        pos = self._data.mocap_pos[obj.mocap]
        return float(pos[0]) * 1000.0, float(pos[1]) * 1000.0

    def object_xy_mm(self, obj_id: str) -> tuple[float, float] | None:
        """Ground-truth floor position of a mirrored sandbox object, or None if it is not mirrored."""
        obj = self._active_objects().get(obj_id)
        return None if obj is None else self._object_xy_mm(obj)

    def _place_object(self, obj: _PhysObject, x_m: float, y_m: float) -> None:
        """Teleport an object upright and at rest (sandbox edits are instantaneous)."""
        d = self._data
        if obj.free:
            d.qpos[obj.qadr : obj.qadr + 7] = (x_m, y_m, obj.rest_z, 1.0, 0.0, 0.0, 0.0)
            d.qvel[obj.vadr : obj.vadr + 6] = 0.0
        else:
            d.mocap_pos[obj.mocap] = (x_m, y_m, obj.rest_z)

    def _sync_world(self) -> None:
        """Sandbox edits (add/remove/drag/reset) move physics bodies; pushed bodies move sandbox objects."""
        if self._sandbox is None or self._data is None:
            return
        listing = {o["id"]: o for o in self._sandbox.list_objects()}
        changed = self._reconcile_bodies(listing)
        for obj_id, obj in self._active_objects().items():
            o = listing[obj_id]
            sandbox_xy = (float(o["x_mm"]), float(o["y_mm"]))
            if obj.sandbox_xy is None or math.dist(sandbox_xy, obj.sandbox_xy) > 0.5:
                self._place_object(obj, sandbox_xy[0] / 1000.0, sandbox_xy[1] / 1000.0)
                obj.sandbox_xy = obj.physics_xy = sandbox_xy
                changed = True
            elif obj.free:
                physics_xy = self._object_xy_mm(obj)
                if math.dist(physics_xy, obj.physics_xy) > self.OBJECT_SYNC_MM:
                    moved = self._sandbox.move_object(obj_id, *physics_xy)
                    obj.physics_xy = physics_xy
                    if moved is not None:
                        obj.sandbox_xy = (round(moved.x_mm, 1), round(moved.y_mm, 1))
        if changed:
            mujoco.mj_forward(self._model, self._data)
            self._wake()

    def _reconcile_bodies(self, listing: dict[str, dict[str, Any]]) -> bool:
        """Adopt, park, create or delete physics bodies so they match the sandbox's object list."""
        spec_changed = parked = False
        for obj_id, obj in list(self._dyn_objects.items()):
            o = listing.get(obj_id)
            if o is None or _signature(o) != obj.signature:
                self._spec.delete(self._spec.body(obj.body))
                del self._dyn_objects[obj_id]
                spec_changed = True
        for slot, (obj_id, obj) in enumerate(self._xml_objects.items()):
            o = listing.get(obj_id)
            if o is not None and obj.signature is None:
                obj.signature = _signature(o)  # the XML body stands in for the sandbox object first seen
            wanted = o is not None and obj_id not in self._dyn_objects and _signature(o) == obj.signature
            if wanted != obj.in_use:
                obj.in_use, obj.sandbox_xy = wanted, None
                if not wanted:
                    self._place_object(obj, _PARK_XY_M[0] + 0.5 * slot, _PARK_XY_M[1])
                    parked = True
        for obj_id, o in listing.items():
            xml = self._xml_objects.get(obj_id)
            if obj_id not in self._dyn_objects and (xml is None or not xml.in_use):
                self._dyn_objects[obj_id] = self._add_dynamic_body(o)
                spec_changed = True
        if spec_changed:
            # Recompiling preserves qpos/qvel/ctrl/mocap state; only addresses need re-indexing.
            self._model, self._data = self._spec.recompile(self._model, self._data)
            self._index_model()
        return spec_changed or parked

    def _add_dynamic_body(self, o: dict[str, Any]) -> _PhysObject:
        """Add a primitive body for a sandbox object that has no counterpart in the MJCF."""
        w, depth, h = (max(10.0, float(o[k])) / 1000.0 for k in ("width_mm", "depth_mm", "height_mm"))
        self._dyn_serial += 1
        name = f"dyn_{self._dyn_serial}"
        if o["shape"] == "sphere":
            gtype, size, rest_z = mujoco.mjtGeom.mjGEOM_SPHERE, [w / 2, 0, 0], w / 2
            volume = math.pi * w**3 / 6
        elif o["shape"] == "cylinder":
            gtype, size, rest_z = mujoco.mjtGeom.mjGEOM_CYLINDER, [w / 2, h / 2, 0], h / 2
            volume = math.pi * w * w * h / 4
        else:
            gtype, size, rest_z = mujoco.mjtGeom.mjGEOM_BOX, [w / 2, depth / 2, h / 2], h / 2
            volume = w * depth * h
        body = self._spec.worldbody.add_body(
            name=name, pos=[float(o["x_mm"]) / 1000.0, float(o["y_mm"]) / 1000.0, rest_z]
        )
        movable = bool(o.get("movable", True))
        if movable:
            body.add_freejoint()
        else:
            body.mocap = True
        body.add_geom(
            type=gtype,
            size=size,
            mass=max(0.03, min(1.5, 250.0 * volume)),  # light hollow toys
            contype=2,
            conaffinity=2,
            friction=[0.5, 0.005, 0.0001],
        )
        return _PhysObject(name, movable, rest_z, dynamic=True, signature=_signature(o), in_use=True)
