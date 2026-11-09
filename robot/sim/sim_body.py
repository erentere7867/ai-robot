"""Kinematic simulation of the base and head: perfect odometry, the same reflexes as the real body."""

from __future__ import annotations

import math

from robot.body.hal import BodyCapabilities, BodyPose, Reflex, SensorReadings
from robot.core.angles import wrap_deg
from robot.core.bus import EventBus
from robot.core.events import EventPriority
from robot.core.service import ServiceHealth


class SimBody:
    name = "sim_body"
    VELOCITY_TIMEOUT_S = 0.35  # motor watchdog: no fresh speed command -> stop
    LINEAR_ACCEL_MM_S2 = 900.0
    ANGULAR_ACCEL_DEG_S2 = 600.0

    def __init__(self, bus: EventBus, capabilities: BodyCapabilities, *, step_dt_s: float = 0.05,
                 battery_idle_pct_per_min: float = 0.5, battery_pct_per_m: float = 1.0,
                 charge_pct_per_min: float = 20.0) -> None:
        self.bus = bus
        self._caps = capabilities
        self.step_dt_s = step_dt_s
        self._pose = BodyPose()
        self._sensors = SensorReadings(front_tof_mm=capabilities.default_front_tof_mm)
        self._running = False
        self._ticks_left = 0.0
        self._ticks_right = 0.0
        self._tof_override: float | None = None
        self._cliff_override: bool | None = None
        self._world = None  # the SimulationSandbox whose walls and objects the sensors see
        self._battery_idle = battery_idle_pct_per_min
        self._battery_per_mm = battery_pct_per_m / 1000.0
        self._charge_rate = charge_pct_per_min
        self._battery_t = bus.clock.monotonic()
        self._vel_target: tuple[float, float] = (0.0, 0.0)
        self._vel_until = 0.0
        self._vel_t = bus.clock.monotonic()
        self._vel_reflex: Reflex | None = None
        self._head_target = 0.0

    def bind_world(self, sandbox) -> None:
        self._world = sandbox

    def _refresh_sensors(self) -> None:
        pose = self.true_pose()
        if self._world is not None:
            self._apply_ranges(self._world.compute_front_tof_mm(pose), self._world.check_cliff_triggered(pose))
        self._update_battery()

    def _apply_ranges(self, front_tof_mm: float, cliff: bool) -> None:
        self._sensors.front_tof_mm = float(front_tof_mm if self._tof_override is None else self._tof_override)
        self._sensors.cliff_triggered = bool(cliff if self._cliff_override is None else self._cliff_override)

    def _update_battery(self, travelled_mm: float = 0.0) -> None:
        now = self.bus.clock.monotonic()
        minutes = max(0.0, now - self._battery_t) / 60.0
        self._battery_t = now
        charging = self._world is not None and self._world.is_charging(self._pose)
        level = self._sensors.battery_pct
        level += minutes * (self._charge_rate if charging else -self._battery_idle)
        level -= abs(travelled_mm) * self._battery_per_mm
        self._sensors.battery_pct = max(0.0, min(100.0, level))
        self._sensors.charging = charging

    def _would_collide(self, x_mm: float, y_mm: float) -> bool:
        if self._world is None:
            return False
        return self._world.check_collision(BodyPose(x_mm, y_mm, self._pose.heading_deg, self._pose.head_tilt_deg))

    def capabilities(self) -> BodyCapabilities:
        return self._caps

    def pose(self) -> BodyPose:
        return self._pose.copy()

    def true_pose(self) -> BodyPose:
        """Ground truth, which sensors see the world from (here odometry is perfect)."""
        return self._pose.copy()

    def sensors(self) -> SensorReadings:
        self._refresh_sensors()
        return self._sensors.copy()

    async def start(self) -> None:
        self._running = True
        await self.publish_state()

    async def stop(self) -> None:
        await self.halt()
        self._running = False

    async def health(self) -> ServiceHealth:
        status = "estop_active" if self._sensors.estop_active else "ok"
        return ServiceHealth(name=self.name, healthy=True, status=status, details=self._pose.to_dict())

    def set_estop(self, active: bool) -> None:
        self._sensors.estop_active = active
        if active:
            self._stop_now()

    def set_simulated_obstacle(self, front_tof_mm: float | None = None, cliff: bool | None = None) -> None:
        """Fault injection: hold a sensor at a value, whatever the world says."""
        if front_tof_mm is not None:
            self._tof_override = float(front_tof_mm)
            self._sensors.front_tof_mm = self._tof_override
        if cliff is not None:
            self._cliff_override = bool(cliff)
            self._sensors.cliff_triggered = self._cliff_override

    def clear_simulated_obstacle(self) -> None:
        self._tof_override = self._cliff_override = None

    def reset_pose(self, pose: BodyPose | None = None) -> None:
        self._pose = pose or BodyPose()
        self._head_target = self._pose.head_tilt_deg
        self._vel_target, self._vel_until = (0.0, 0.0), self.bus.clock.monotonic()

    # -- commands --------------------------------------------------------------------------------------

    async def set_velocity(self, linear_mm_s: float, angular_deg_s: float) -> Reflex | None:
        reason = await self.update_motion()
        v, w = self._caps.clamp_velocity(linear_mm_s, angular_deg_s)
        if self._sensors.estop_active:
            v = w = 0.0
        self._vel_target = (v, w)
        self._vel_until = self.bus.clock.monotonic() + self.VELOCITY_TIMEOUT_S
        return reason

    def set_head(self, tilt_deg: float) -> None:
        self._head_target = self._caps.clamp_head(tilt_deg)

    async def halt(self) -> None:
        self._stop_now()

    def _stop_now(self) -> None:
        self._vel_target, self._vel_until = (0.0, 0.0), self.bus.clock.monotonic()
        self._pose.linear_vel_mm_s = self._pose.angular_vel_deg_s = 0.0

    def _head_step(self, dt: float) -> bool:
        error = self._head_target - self._pose.head_tilt_deg
        if abs(error) < 1e-6:
            return False
        limit = self._caps.head_tilt.max_speed_deg_s * dt
        self._pose.head_tilt_deg += max(-limit, min(limit, error))
        self._sensors.imu_pitch_deg = self._pose.head_tilt_deg
        return True

    async def update_motion(self) -> Reflex | None:
        now = self.bus.clock.monotonic()
        t, self._vel_t = self._vel_t, now
        reflex: Reflex | None = None
        moved = False
        v, w = self._pose.linear_vel_mm_s, self._pose.angular_vel_deg_s
        while t < now - 1e-9:
            dt = min(self.step_dt_s, now - t)
            target_v, target_w = self._vel_target if t < self._vel_until else (0.0, 0.0)
            v += max(-self.LINEAR_ACCEL_MM_S2 * dt, min(self.LINEAR_ACCEL_MM_S2 * dt, target_v - v))
            w += max(-self.ANGULAR_ACCEL_DEG_S2 * dt, min(self.ANGULAR_ACCEL_DEG_S2 * dt, target_w - w))
            t += dt
            moved = self._head_step(dt) or moved
            if abs(v) < 1e-6 and abs(w) < 1e-6:
                v = w = 0.0
                if t >= self._vel_until and self._pose.head_tilt_deg == self._head_target:
                    break  # at rest with nothing commanded: nothing more to integrate
                continue
            self._refresh_sensors()
            if self._sensors.estop_active:
                reflex, v, w = Reflex.ESTOP, 0.0, 0.0
            elif self._sensors.cliff_triggered or self._sensors.picked_up:
                reflex, v, w = Reflex.CLIFF, 0.0, 0.0
            elif v > 0 and (not math.isfinite(self._sensors.front_tof_mm)
                            or self._sensors.front_tof_mm <= self._caps.min_front_tof_stop_mm):
                reflex, v = Reflex.OBSTACLE_AHEAD, 0.0
            if w:
                self._apply_rotation(w * dt)
            if v:
                rad = math.radians(self._pose.heading_deg)
                nx = self._pose.x_mm + v * dt * math.cos(rad)
                ny = self._pose.y_mm + v * dt * math.sin(rad)
                if self._would_collide(nx, ny) and not self._would_collide(self._pose.x_mm, self._pose.y_mm):
                    reflex, v, w = Reflex.BUMPER, 0.0, 0.0
                else:
                    self._pose.x_mm, self._pose.y_mm = nx, ny
                    self._add_wheel_ticks(v * dt, v * dt)
                    self._update_battery(abs(v) * dt)
            moved = True
        self._pose.linear_vel_mm_s, self._pose.angular_vel_deg_s = v, w
        if reflex is not None and reflex != self._vel_reflex:
            await self._emit_reflex(reflex)
            self._pose.linear_vel_mm_s, self._pose.angular_vel_deg_s = v, w
        self._vel_reflex = reflex
        if reflex is not None and reflex.halts:
            self._vel_target, self._vel_until = (0.0, 0.0), now  # stays stopped until a new command
        if moved:
            await self.publish_state()
        return reflex

    def _apply_rotation(self, delta_deg: float) -> None:
        self._pose.heading_deg = wrap_deg(self._pose.heading_deg + delta_deg)
        self._sensors.imu_yaw_deg = self._pose.heading_deg
        arc_mm = math.radians(delta_deg) * (self._caps.track_width_mm / 2.0)
        self._add_wheel_ticks(-arc_mm, arc_mm)

    def _add_wheel_ticks(self, left_mm: float, right_mm: float) -> None:
        ticks_per_mm = self._caps.encoder_ticks_per_rev / (math.pi * self._caps.wheel_diameter_mm)
        self._ticks_left += left_mm * ticks_per_mm
        self._ticks_right += right_mm * ticks_per_mm
        self._sensors.wheel_ticks_left = round(self._ticks_left)
        self._sensors.wheel_ticks_right = round(self._ticks_right)

    async def publish_state(self) -> None:
        self._refresh_sensors()
        await self.bus.emit("body.pose", source=self.name,
                            payload={"pose": self._pose.to_dict(), "sensors": self._sensors.to_dict()},
                            priority=EventPriority.TELEMETRY)

    async def _emit_reflex(self, reflex: Reflex) -> None:
        self._pose.linear_vel_mm_s = self._pose.angular_vel_deg_s = 0.0
        await self.bus.emit("reflex.triggered", source=self.name,
                            payload={"reflex": str(reflex), "reason": reflex.reason, "pose": self._pose.to_dict(),
                                     "sensors": self._sensors.to_dict()},
                            priority=EventPriority.CRITICAL)
