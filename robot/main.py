"""Main RobotHarness assembly and CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import uuid
from pathlib import Path

from robot.actions.executor import ActionExecutor
from robot.actions.registry import CapabilityRegistry
from robot.audio.asr import SpeechRecognizer
from robot.audio.doa import MicArray, SimVoice
from robot.audio.service import AudioService
from robot.behavior.arbiter import BehaviorArbiter
from robot.behavior.attention import AttentionManager
from robot.behavior.autonomy import AutonomyController
from robot.behavior.drives import DriveSystem
from robot.behavior.emotion import EmotionEngine
from robot.behavior.fsm import InteractionFSM
from robot.body.hal import BodyCapabilities, BodyPose
from robot.body.sim_body import SimBody
from robot.cognition.agent import AgentLoop
from robot.cognition.backend import InferenceBackend, NullBackend
from robot.cognition.llamacpp import AutoBackend, LlamaCppBackend
from robot.cognition.prompts import PromptBuilder, estimate_tokens
from robot.control.fast_loop import FastLoop
from robot.control.safety import CollisionMonitor
from robot.core.bus import EventBus
from robot.core.clock import Clock, RealClock
from robot.core.config import RobotConfig
from robot.core.events import Event, EventPriority
from robot.core.realtime import plan_cores, tune_process
from robot.core.recorder import EventRecorder
from robot.core.service import ServiceSupervisor
from robot.face.animator import EYE_COLOR_THEMES
from robot.face.process import FaceProcessBridge
from robot.interfaces.console import ConsoleInterface
from robot.interfaces.dashboard import DashboardServer
from robot.memory.consolidator import MemoryConsolidator
from robot.memory.store import MemoryStore
from robot.navigation.planner import GridPlanner
from robot.perception.camera import CameraRig, SimCamera
from robot.perception.detector import VLMDetector
from robot.perception.geometry import default_rig, load_rig
from robot.skills import SkillContext, build_default_registry
from robot.world.localization import LocalizedBody, Localizer
from robot.world.sandbox import SimulationSandbox
from robot.world.spatial_map import SpatialMap
from robot.world.state import WorldState


class RobotHarness:
    """Top-level embodied AI robot harness coordinating all subsystems."""

    def __init__(
        self,
        config: RobotConfig | None = None,
        *,
        clock: Clock | None = None,
        backend: InferenceBackend | None = None,
        require_wake_word: bool | None = None,
        spawn_face_window: bool = False,
        enable_dashboard: bool = False,
        dashboard_port: int = 8090,
        enable_speaker: bool = False,
        enable_mic: bool = False,
        tts_mode: str = "auto",
        asr_mode: str = "auto",
        realtime_audio_sync: bool | None = None,
    ) -> None:
        self.config = config or RobotConfig.from_yaml()
        self.clock: Clock = clock or RealClock()
        self.bus = EventBus(clock=self.clock, default_queue_size=self.config.max_queue_size)

        self.recorder: EventRecorder | None = None
        if self.config.record_events:
            self.recorder = EventRecorder(self.config.recordings_dir)
            self.bus.add_record_hook(self.recorder.record)

        self.capabilities = (
            BodyCapabilities.from_yaml(self.config.body_config_path)
            if self.config.body_config_path.exists()
            else BodyCapabilities()
        )
        self.registry = CapabilityRegistry(self.capabilities)
        self.body = self._make_body()
        self.rig = load_rig(self.config.body_config_path) if self.config.body_config_path.exists() else default_rig()
        self.sandbox = SimulationSandbox(self.bus, rig=self.rig)
        self.executor = ActionExecutor(self.bus, self.body, self.registry)
        self.memory = MemoryStore(self.config.database_path, clock=self.clock)
        self.consolidator = MemoryConsolidator(self.bus, self.memory)
        self.world = WorldState(self.bus, clock=self.clock)
        self.spatial_map = SpatialMap(self.clock)
        # Lidar localization: everything downstream reads the corrected (map-frame) pose.
        self.localizer: Localizer | None = None
        if self.config.perception.localization and self.config.perception.lidar != "off":
            self.localizer = Localizer(self.spatial_map)
            self.body = LocalizedBody(self.body, self.localizer)
        self.planner = GridPlanner(room_half_extent_mm=self.spatial_map.half, resolution_mm=50.0)
        if self.config.camera == "sim":
            cams = [SimCamera(self.sandbox, self.body, self.clock, m) for m in self.rig]
        else:
            from robot.perception.camera import RealCamera

            cams = [RealCamera(self.body, self.clock, m, source=self.config.camera) for m in self.rig]
        self.camera = CameraRig(cams)
        self.mic_array = (MicArray.from_body_config(self.config.body_config_path)
                          if self.config.body_config_path.exists() else None)
        # In the simulator a typed message is the simulated person speaking: their voice reaches the simulated
        # microphone array from where they stand, and the robot finds them the way it would on hardware.
        self.voice = (SimVoice(self.sandbox, self.body, self.mic_array)
                      if self.mic_array is not None and self.config.body_backend != "ventuno" else None)
        self.face_bridge = FaceProcessBridge(
            self.bus,
            spawn_window=spawn_face_window,
            fps=self.config.face_fps,
        )
        self.audio = AudioService(
            self.bus,
            self.face_bridge,
            enable_speaker=enable_speaker,
            enable_mic=enable_mic,
            tts_mode=tts_mode,
            asr_mode=asr_mode,
            realtime_sync=realtime_audio_sync,
            mic_array=self.mic_array,
        )
        self.audio.asr = SpeechRecognizer(
            mode=self.config.asr_mode if asr_mode == "auto" else asr_mode,
            whisper_bin=self.config.asr_bin,
            whisper_model=self.config.asr_model_path,
        )

        self.fsm = InteractionFSM(
            clock=self.clock,
            follow_up_window_s=self.config.follow_up_window_s,
            idle_sleepy_timeout_s=self.config.idle_sleepy_timeout_s,
        )
        self.emotion = EmotionEngine(
            clock=self.clock,
            allowed_emotions=self.registry.allowed_emotions,
            decay_rate_per_s=self.config.emotion_decay_rate_per_s,
        )
        self.attention = AttentionManager(
            self.bus,
            self.fsm,
            wake_words=self.config.wake_words,
            require_wake_word=enable_mic if require_wake_word is None else require_wake_word,
        )
        self.prompts = PromptBuilder(self.config.soul_path, self.config.context_budget, self.config.name)

        self._llama_client: LlamaCppBackend | None = None
        if backend is not None:
            self.backend: InferenceBackend = backend
        elif self.config.llm.backend == "null":
            self.backend = NullBackend()
        elif self.config.llm.backend == "llamacpp":
            self._llama_client = LlamaCppBackend(self.config.llm)
            self.backend = self._llama_client
        else:
            self._llama_client = LlamaCppBackend(self.config.llm)
            self.backend = AutoBackend(self._llama_client)

        self.detector = VLMDetector(self.backend)  # the open-vocabulary eye: the decision model itself
        self.skills = build_default_registry(sorted(EYE_COLOR_THEMES))
        self.skill_context = SkillContext(
            bus=self.bus, executor=self.executor, body=self.body, camera=self.camera, detector=self.detector,
            spatial_map=self.spatial_map, planner=self.planner, memory=self.memory, emotion=self.emotion,
            clock=self.clock, registry=self.registry, on_event=self._emit_skill_event,
        )
        # Fast layer: the trained detector on every frame plus closed-loop behaviours. Without a trained
        # detector the VLM serves single looks and the behaviours still run on the map.
        self.fast_detector = self._load_fast_detector()
        self.lidar = self._make_lidar()
        pcfg = self.config.perception
        self.core_plan = plan_cores()
        dims = self.capabilities
        self.fast = FastLoop(
            bus=self.bus, clock=self.clock, body=self.body, camera=self.camera, spatial_map=self.spatial_map,
            planner=self.planner, detector=self.fast_detector or self.detector, lidar=self.lidar,
            localizer=self.localizer,
            rate_hz=pcfg.loop_hz,
            continuous=isinstance(self.clock, RealClock) and self.fast_detector is not None,
            monitor=CollisionMonitor(half_length_mm=0.5 * dims.length_mm, half_width_mm=0.5 * dims.width_mm)
            if pcfg.collision_monitor else None,
            threaded=pcfg.control_thread and isinstance(self.clock, RealClock),
            control_cores=self.core_plan.control, perception_cores=self.core_plan.perception,
            priority=pcfg.realtime_priority,
        )
        self.skill_context.fast = self.fast
        budget = self.config.context_budget
        self.agent = AgentLoop(self.backend, self.skills, self.config.agent,
                               prompt_limit_tokens=budget.total_tokens - budget.response_reserve_tokens,
                               on_event=self._emit_skill_event, robot_name=self.config.name)
        self.arbiter = BehaviorArbiter(
            bus=self.bus, fsm=self.fsm, emotion=self.emotion, prompts=self.prompts, agent=self.agent,
            skill_context=self.skill_context, memory=self.memory, world=self.world,
            consolidator=self.consolidator, agent_config=self.config.agent,
        )
        self.drives = DriveSystem(rate_scale=self.config.autonomy.drive_rate_scale)
        self.autonomy = AutonomyController(
            bus=self.bus, arbiter=self.arbiter, body=self.body, config=self.config.autonomy,
            attention=self.attention, audio=self.audio, camera=self.camera, spatial_map=self.spatial_map,
            drives=self.drives,
        )
        self.arbiter.autonomy = self.autonomy
        self.arbiter.drives = self.drives

        self.dashboard: DashboardServer | None = (
            DashboardServer(self, port=dashboard_port) if enable_dashboard else None
        )

        self.supervisor = ServiceSupervisor(self.bus, health_interval_s=5.0)
        self.supervisor.register(self.body)
        self.supervisor.register(self.executor)
        self.supervisor.register(self.fast)
        self.supervisor.register(self.face_bridge)
        self.supervisor.register(self.audio)
        if self.dashboard is not None:
            self.supervisor.register(self.dashboard)
        self.supervisor.register(self.autonomy)
        self._started = False
        self._closed = False
        self._runtime_task: asyncio.Task | None = None
        self._requests: dict[str, asyncio.Future] = {}
        self._request_subs: list[str] = []

    def _load_fast_detector(self):
        cfg = self.config.perception
        if cfg.fast_detector == "off":
            return None
        try:
            from robot.perception.fast_detector import FastDetector
        except ImportError:
            logging.getLogger(__name__).info("ultralytics is not installed; perception uses the VLM only")
            return None
        # A freshly trained detector in the runtime folder wins; otherwise the one shipped in the repository.
        detector = None
        for models_dir in (self.config.runtime_dir / "models", Path(__file__).resolve().parents[1] / "models"):
            detector = FastDetector.load_default(models_dir, prefer_ncnn=cfg.prefer_ncnn, imgsz=cfg.imgsz,
                                                 conf=cfg.confidence, gpu=cfg.detector_gpu,
                                                 threads=cfg.detector_threads)
            if detector is not None:
                break
        if detector is None:
            logging.getLogger(__name__).info("No trained fast detector (run scripts/train_detector.py); "
                                             "perception uses the VLM only")
        return detector

    def _make_lidar(self):
        source = self.config.perception.lidar
        if source == "off":
            return None
        if source == "sim":
            from robot.perception.lidar import SimLidar

            return SimLidar(self.sandbox, self.body, self.clock)
        from robot.perception.lidar import Ld19Lidar

        return Ld19Lidar(self.body, self.clock, port=source)

    def _make_body(self):
        tick = 1.0 / self.config.sim_tick_hz
        if self.config.body_backend == "ventuno":
            from robot.body.ventuno_body import VentunoBody

            return VentunoBody(self.bus, self.capabilities)
        if self.config.body_backend == "physics":
            from robot.body.physics_body import PhysicsBody

            return PhysicsBody(self.bus, self.capabilities, step_dt_s=tick)
        return SimBody(self.bus, self.capabilities, step_dt_s=tick)

    async def _emit_skill_event(self, event_type: str, payload: dict) -> None:
        await self.bus.emit(event_type, "agent", payload, priority=EventPriority.NORMAL,
                            correlation_id=self.skill_context.correlation_id)

    def _on_pose_for_map(self, event: Event) -> None:
        pose, sensors = event.payload.get("pose"), event.payload.get("sensors")
        if not isinstance(pose, dict) or not isinstance(sensors, dict):
            return
        try:
            body_pose = BodyPose(float(pose["x_mm"]), float(pose["y_mm"]), float(pose["heading_deg"]),
                                 float(pose.get("head_tilt_deg", 0.0)))
            if self.localizer is not None:
                body_pose = self.localizer.to_map(body_pose)  # events carry raw odometry
            self.spatial_map.update_range(body_pose, float(sensors.get("front_tof_mm", float("inf"))))
        except (KeyError, TypeError, ValueError):
            return

    def _on_voice(self, event: Event) -> None:
        """A message that came with a voice direction: find who said it (before the turn reads the map)."""
        voice = event.payload.get("voice")
        if not isinstance(voice, dict) or "bearing_deg" not in voice:
            return
        try:
            scan = self.lidar.latest() if self.lidar is not None else None
            self.spatial_map.hear_voice(self.body.pose(), float(voice["bearing_deg"]), scan=scan,
                                        origin_mm=self.mic_array.center_x_mm if self.mic_array else 25.0)
        except (TypeError, ValueError):
            logging.getLogger(__name__).warning("Bad voice direction %r", voice)

    def validate_user_text(self, text: str) -> None:
        if not text.strip() or len(text) > 4096:
            raise ValueError("Message must contain 1..4096 characters")
        if estimate_tokens(text) > 600:
            raise ValueError("Message is too long for the robot's context; please send a shorter request")

    async def submit_message(
        self, text: str, *, source: str = "dashboard", direct: bool = True, timeout_s: float = 300.0
    ) -> dict:
        """Shared, correlated input path; requests cannot consume each other's replies."""
        if not self._started:
            raise RuntimeError("Harness is not running")
        self.validate_user_text(text)
        if len(self._requests) >= 8:
            raise RuntimeError("Robot is busy")
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._requests[request_id] = future
        payload: dict = {"text": text, "direct": direct}
        if self.voice is not None and (voice := await asyncio.to_thread(self.voice.locate)) is not None:
            payload["voice"] = voice
        try:
            await self.bus.emit(
                "input.user_message",
                source,
                payload,
                priority=EventPriority.HIGH,
                correlation_id=request_id,
            )
            return await asyncio.wait_for(future, timeout=timeout_s)
        except (TimeoutError, asyncio.CancelledError):
            if self.arbiter._input_correlation_id == request_id:
                await self.bus.emit(
                    "action.cancel",
                    "request",
                    {"reason": "request_cancelled"},
                    priority=EventPriority.CRITICAL,
                )
            raise
        finally:
            self._requests.pop(request_id, None)

    def _resolve_request(self, event: Event) -> None:
        future = self._requests.get(event.correlation_id)
        if future is not None and not future.done():
            future.set_result(
                dict(event.payload)
                if event.type == "robot.response"
                else {"say": "", "backend": "cancelled", "actions": [], "memory": []}
            )

    async def _runtime_loop(self) -> None:
        while self._started:
            await asyncio.sleep(0.1)
            try:
                await self.arbiter.step_idle_timers()
                await self.autonomy.step()
            except Exception:
                logging.getLogger(__name__).exception("Runtime timer step failed")

    async def start(self) -> None:
        if self._started:
            return
        if self._closed:
            raise RuntimeError("Create a new harness after closing it")
        try:
            if self.recorder is not None:
                self.recorder.open()
            if self._llama_client is not None and await self._llama_client.ensure_server_started():
                await self.agent.warm_up(self.prompts.system_prompt)
            if self.config.body_backend != "ventuno":
                self.sandbox.bind(getattr(self.body, "raw", self.body))  # simulated sensors; a real body has its own
            self.spatial_map.load(self.config.spatial_map_path)
            start = self.body.pose()
            self.spatial_map.places["start"] = (start.x_mm, start.y_mm)
            self._request_subs.append(self.bus.subscribe("body.pose", self._on_pose_for_map, immediate=True))
            # Before attention binds: the speaker is placed before anything reacts to the words.
            self._request_subs.append(self.bus.subscribe("input.user_message", self._on_voice, immediate=True))
            self.world.bind()
            self.autonomy.bind()
            self.attention.bind()
            self.arbiter.bind()
            for event_type in ("robot.response", "turn.cancelled", "attention.ignored"):
                self._request_subs.append(
                    self.bus.subscribe(event_type, self._resolve_request, immediate=True)
                )
            await self.bus.start()
            await self.supervisor.start_all()
            self._started = True
            if self.config.perception.tune_process:
                logging.getLogger(__name__).info("Real-time tuning: %s", tune_process())
            if isinstance(self.clock, RealClock):
                self._runtime_task = asyncio.create_task(self._runtime_loop(), name="runtime:timers")
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        if self._closed:
            return
        self._started = False
        if self._runtime_task:
            self._runtime_task.cancel()
            await asyncio.gather(self._runtime_task, return_exceptions=True)
            self._runtime_task = None
        self.arbiter.cancel_active()
        await self.executor.cancel_all()
        if self.arbiter._active_turn:
            await asyncio.wait({self.arbiter._active_turn}, timeout=1.0)
        await self.supervisor.stop_all()
        try:
            self.spatial_map.save(self.config.spatial_map_path)
        except OSError:
            logging.getLogger(__name__).warning("Could not save the spatial map", exc_info=True)
        self.arbiter.unbind()
        self.attention.unbind()
        self.world.unbind()
        self.sandbox.unbind()
        for sub_id in self._request_subs:
            self.bus.unsubscribe(sub_id)
        self._request_subs.clear()
        for future in self._requests.values():
            if not future.done():
                future.cancel()
        self._requests.clear()
        await self.bus.stop()
        if self._llama_client is not None:
            await self._llama_client.close()
        if self.recorder is not None:
            await asyncio.to_thread(self.recorder.close)
        self.memory.close()
        self._closed = True

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_exc):
        await self.stop()


async def _async_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Embodied AI Robot Harness (Milestone H3)")
    parser.add_argument("--config", default="config/robot.yaml", help="Path to robot.yaml")
    parser.add_argument("--db", default=None, help="Override SQLite memory database path")
    parser.add_argument(
        "--backend",
        choices=["auto", "llamacpp", "null"],
        default=None,
        help="Override LLM inference backend mode",
    )
    parser.add_argument("--once", default=None, help="Send a single message, print response, and exit")
    parser.add_argument("--no-record", action="store_true", help="Disable JSONL event recording")
    parser.add_argument(
        "--face-window", action="store_true", help="Launch standalone 60 FPS SDL2 face window"
    )
    parser.add_argument("--no-dashboard", action="store_true", help="Disable web diagnostic dashboard")
    parser.add_argument("--dashboard-port", type=int, default=8090, help="Port for web diagnostic dashboard")
    parser.add_argument("--mute", action="store_true", help="Disable speaker playback for SFX and TTS")
    parser.add_argument("--mic", action="store_true", help="Enable live microphone VAD + offline ASR")
    parser.add_argument("--headless", action="store_true", help="Run services without an interactive console")
    parser.add_argument("--no-autonomy", action="store_true", help="Disable self-directed idle exploration")
    parser.add_argument("--no-mujoco-viewer", action="store_true", help="Do not auto-launch the native 60 FPS desktop MuJoCo 3D window")
    args = parser.parse_args(argv)
    if args.headless and args.once is not None:
        parser.error("--headless and --once cannot be combined")

    cfg = RobotConfig.from_yaml(args.config)
    if args.db:
        cfg.database_path = Path(args.db).expanduser().resolve()
    if args.backend:
        cfg.llm.backend = args.backend
    if args.no_record:
        cfg.record_events = False
    if args.no_autonomy or args.once is not None:
        cfg.autonomy.enabled = False

    enable_dash = (args.once is None) and (not args.no_dashboard)
    harness = RobotHarness(
        cfg,
        spawn_face_window=args.face_window,
        enable_dashboard=enable_dash,
        dashboard_port=args.dashboard_port,
        enable_speaker=not args.mute,
        enable_mic=args.mic,
    )
    console = None if args.headless else ConsoleInterface(harness, verbose_events=True)
    if console:
        console.bind()
    try:
        await harness.start()
        if enable_dash and not args.no_mujoco_viewer and not args.headless and harness.dashboard is not None:
            harness.sandbox.launch_native_viewer(harness.dashboard.url)
        if args.headless:
            stop_requested = asyncio.Event()
            loop = asyncio.get_running_loop()
            try:
                loop.add_signal_handler(signal.SIGTERM, stop_requested.set)
            except NotImplementedError:
                pass
            try:
                await stop_requested.wait()
            finally:
                try:
                    loop.remove_signal_handler(signal.SIGTERM)
                except NotImplementedError:
                    pass
        elif args.once is not None:
            await console.send_and_wait(args.once)
            await harness.arbiter.wait_until_done()
            await harness.audio.wait_until_done(timeout_s=30.0)
        else:
            await console.run_interactive()
    finally:
        if console:
            console.unbind()
        await harness.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_async_main(argv))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
