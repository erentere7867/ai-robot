"""Configuration loader and typed settings structures."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


def load_yaml_file(path: Path | str, _depth: int = 0) -> dict[str, Any]:
    """Load a YAML file into a dictionary. `extends: other.yaml` (relative to this file) starts from that file;
    each section given here replaces keys of the same section there, and a section set to null removes it."""
    file_path = Path(path).expanduser().resolve()
    text = file_path.read_text(encoding="utf-8")
    if yaml is None:
        raise RuntimeError(f"PyYAML is required to load {file_path}")
    data = dict(yaml.safe_load(text) or {})
    base_name = data.pop("extends", None)
    if base_name is None:
        return data
    if _depth > 4:
        raise ValueError(f"{file_path}: 'extends' nests too deeply")
    merged = load_yaml_file(file_path.parent / str(base_name), _depth + 1)
    for key, value in data.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


_BOOL_WORDS = {"true": True, "yes": True, "on": True, "1": True, "false": False, "no": False, "off": False, "0": False}


def _as_bool(value: Any, key: str) -> bool:
    """Parse a YAML switch strictly; `bool("false")` would silently be True."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in _BOOL_WORDS:
        return _BOOL_WORDS[value.strip().lower()]
    raise ValueError(f"{key} must be a boolean")


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a config section; a header whose keys are all commented out loads as None."""
    value = raw.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' must be a mapping")
    return value


@dataclass
class ContextBudgetConfig:
    total_tokens: int = 4096
    identity_tokens: int = 200
    memories_tokens: int = 160
    world_state_tokens: int = 260
    tools_tokens: int = 1100
    recent_turns_tokens: int = 700
    max_recent_turns: int = 6
    response_reserve_tokens: int = 300

    def __post_init__(self) -> None:
        for key, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"context_budget.{key} must be a nonnegative integer")
        if self.total_tokens < 512 or self.response_reserve_tokens >= self.total_tokens:
            raise ValueError("Context budget must leave room for instructions and a response")
        if self.total_tokens > 32768 or self.max_recent_turns > 64:
            raise ValueError("Context/history limit is too large for this harness")


@dataclass
class LLMConfig:
    backend: str = "auto"  # "auto" | "llamacpp" | "null"
    auto_start_server: bool = True
    base_url: str = "http://127.0.0.1:8080"
    model_name: str = "LFM3-VL-2B-Q4_K_M"
    model_path: Path = field(
        default_factory=lambda: Path("~/Downloads/LFM3-VL-2B-Q4_K_M.gguf").expanduser()
    )
    mmproj_path: Path = field(
        default_factory=lambda: Path("~/Downloads/mmproj-LFM3-VL-2B-F16.gguf").expanduser()
    )
    llama_server_bin: Path = field(
        default_factory=lambda: Path(
            "~/robot-runtime/llama.cpp/llama-server.exe"
            if sys.platform == "win32"
            else "~/robot-runtime/llama.cpp/build/bin/llama-server"
        ).expanduser()
    )
    gpu_layers: int = 99
    threads: int = 0  # CPU threads for llama-server (0 = its default); 4 on a Raspberry Pi 5
    ctx_size: int = 8192
    timeout_s: float = 90.0
    temperature: float = 0.0
    max_tokens: int = 256
    native_tools: bool = True  # kept for config compatibility; tool calls are always native
    startup_timeout_s: float = 90.0
    # LFM3-VL's thinking mode for decisions (detection never thinks). Measured on the decision benchmark:
    # off 95% at 0.7 s per decision, on 86% at 5-9 s, so it is off by default.
    thinking: bool = False
    thinking_budget_tokens: int = 512
    # Can this model see camera images? False for a text-only decision model (then `vision_llm` is the eye).
    vision: bool = True
    # Extra llama-server flags for this model, e.g. ["--swa-full", "--checkpoint-min-step", "256"] for LFM3.
    server_args: list[str] | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "null", "llamacpp"}:
            raise ValueError("Unknown LLM backend")
        if not isinstance(self.native_tools, bool):
            raise ValueError("llm.native_tools must be boolean")
        if not math.isfinite(self.timeout_s) or not 0 < self.timeout_s <= 300:
            raise ValueError("LLM timeout must be in (0, 300] seconds")
        if not math.isfinite(self.startup_timeout_s) or not 1 <= self.startup_timeout_s <= 600:
            raise ValueError("LLM startup timeout must be in [1, 600] seconds")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("LLM temperature must be in [0, 2]")
        if not 128 <= self.ctx_size <= 131072 or not 1 <= self.max_tokens < self.ctx_size:
            raise ValueError("Invalid LLM context/output limit")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not (PROJECT_ROOT / "config" / "robot.yaml").exists():
    PROJECT_ROOT = Path(sys.prefix) / "share" / "companion-robot"


@dataclass
class AgentConfig:
    max_steps: int = 14
    max_actions: int = 200
    timeout_s: float = 300.0
    plan: bool = True  # hand over a request with several parts one part at a time (see cognition/planner.py)
    # When the action a request asked for succeeds, say what the skill reports instead of asking the model
    # for a closing sentence (one model call fewer per request).
    fast_finish: bool = True

    def __post_init__(self) -> None:
        for key in ("max_steps", "max_actions"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 1000:
                raise ValueError(f"agent.{key} must be an integer in [1, 1000]")
        if not math.isfinite(self.timeout_s) or not 1 <= self.timeout_s <= 3600:
            raise ValueError("agent.timeout_s must be in [1, 3600]")


@dataclass
class AutonomyConfig:
    enabled: bool = True
    idle_after_s: float = 20.0
    think_interval_s: float = 45.0
    observation_interval_s: float = 1.0
    drive_rate_scale: float = 1.0
    # Speaking up unasked at most this often; it still acts (explores, looks) in between. Every 30-40 s
    # ("I'm feeling a little lonely! Would you like...?") wore people out at a demo.
    speak_interval_s: float = 180.0
    # "rules": drives pick a behaviour directly (explore, play, go charge) and the model is only woken for
    # events worth words (a scene change, a requested wake, loneliness when it may speak). "model": every
    # drive wakes the model (the old way: overnight it said one of three near-identical lines 150 times).
    policy: str = "rules"

    def __post_init__(self) -> None:
        for key in ("idle_after_s", "think_interval_s", "observation_interval_s", "drive_rate_scale",
                    "speak_interval_s"):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0.1:
                raise ValueError(f"autonomy.{key} must be finite and at least 0.1 seconds")
        if not isinstance(self.enabled, bool):
            raise ValueError("Autonomy switches must be booleans")
        if self.policy not in ("rules", "model"):
            raise ValueError("autonomy.policy must be 'rules' or 'model'")


@dataclass
class PerceptionConfig:
    """The fast layer: a small detector on every frame and a 20 Hz control loop."""

    fast_detector: str = "auto"  # auto: use ~/robot-runtime/models/fast_detector* if trained; off: VLM only
    # 2D lidar: "off", "sim" (raycast in the sandbox) or a serial port for an LD19/LD06 ("/dev/ttyUSB0").
    lidar: str = "off"
    # NCNN (no PyTorch) is the fastest CPU runtime for YOLO on ARM and is what the Pi runs; using it on the
    # laptop too means every simulator run exercises the exact perception code of the robot.
    prefer_ncnn: bool = True
    imgsz: int = 320
    confidence: float = 0.4
    loop_hz: float = 20.0
    # Real-time control (see robot/control/fast_loop.py and robot/core/realtime.py).
    control_thread: bool = False      # run the fast loop in its own pinned, prioritised thread
    realtime_priority: int = 0        # SCHED_FIFO priority for that thread (0: leave it normal)
    collision_monitor: bool = True    # limit every speed command to what can still stop in time
    tune_process: bool = False        # GC freeze, 1 ms GIL switch interval, mlockall (on the robot)
    localization: bool = False        # correct odometry drift by matching lidar scans to the map

    def __post_init__(self) -> None:
        if not 0 <= self.realtime_priority <= 99:
            raise ValueError("perception.realtime_priority must be 0..99")
        if self.fast_detector not in ("auto", "off"):
            raise ValueError("perception.fast_detector must be 'auto' or 'off'")
        if not 64 <= self.imgsz <= 1280 or not 0.05 <= self.confidence <= 0.95 or not 2 <= self.loop_hz <= 100:
            raise ValueError("perception settings are outside supported limits")


@dataclass
class RobotConfig:
    name: str = "Sprocket"
    wake_words: list[str] = field(default_factory=lambda: ["sprocket", "hey sprocket", "robot", "hey robot"])
    follow_up_window_s: float = 8.0
    idle_sleepy_timeout_s: float = 45.0
    emotion_decay_rate_per_s: float = 0.08
    runtime_dir: Path = field(default_factory=lambda: Path("~/robot-runtime").expanduser())
    database_path: Path = field(default_factory=lambda: Path("~/robot-runtime/data/memory.db").expanduser())
    recordings_dir: Path = field(default_factory=lambda: Path("~/robot-runtime/data/recordings").expanduser())
    soul_path: Path = field(default_factory=lambda: PROJECT_ROOT / "config" / "SOUL.md")
    body_config_path: Path = field(default_factory=lambda: PROJECT_ROOT / "config" / "body.yaml")
    sim_tick_hz: float = 20.0
    # "kinematic" | "physics" (simulated) | "serial" (a USB motor board) | "mcu" (VENTUNO Q's on-board STM32H5)
    body_backend: str = "kinematic"
    serial_port: str = "/dev/ttyACM0"  # the motor microcontroller (firmware/base_controller)
    camera: str = "sim"  # "sim" | "picamera" (Pi camera module) | "csi" (MIPI camera, GStreamer) | "usb"
    spatial_map_path: Path = field(
        default_factory=lambda: Path("~/robot-runtime/data/spatial_map.json").expanduser()
    )
    max_queue_size: int = 256
    record_events: bool = True
    face_fps: float = 30.0
    asr_mode: str = "auto"
    asr_bin: Path = field(
        default_factory=lambda: Path("~/robot-runtime/whisper.cpp/build/bin/whisper-cli").expanduser()
    )
    asr_model_path: Path = field(
        default_factory=lambda: Path("~/robot-runtime/models/ggml-tiny.en.bin").expanduser()
    )
    llm: LLMConfig = field(default_factory=LLMConfig)
    # Optional second model for the camera (naming things, yes/no checks, questions about the view) when
    # `llm` is a text-only decision model. None: `llm` does both.
    vision_llm: LLMConfig | None = None
    context_budget: ContextBudgetConfig = field(default_factory=ContextBudgetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    autonomy: AutonomyConfig = field(default_factory=AutonomyConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)

    def __post_init__(self) -> None:
        for key in (
            "follow_up_window_s",
            "idle_sleepy_timeout_s",
            "emotion_decay_rate_per_s",
            "sim_tick_hz",
            "face_fps",
        ):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not 1 <= self.sim_tick_hz <= 200 or not 1 <= self.max_queue_size <= 4096:
            raise ValueError("Simulation rate or queue size is outside supported limits")
        if self.body_backend not in ("kinematic", "physics", "serial", "mcu"):
            raise ValueError("runtime.body_backend must be 'kinematic', 'physics', 'serial' or 'mcu'")
        if self.camera not in ("sim", "picamera", "csi", "usb"):
            raise ValueError("runtime.camera must be 'sim', 'picamera', 'csi' or 'usb'")

    @classmethod
    def from_yaml(
        cls, path: Path | str = "config/robot.yaml", project_root: Path | None = None
    ) -> RobotConfig:
        root = (project_root or PROJECT_ROOT).resolve()
        cfg_path = Path(path)
        if not cfg_path.is_absolute():
            cwd_candidate = (Path.cwd() / cfg_path).resolve()
            cfg_path = cwd_candidate if cwd_candidate.exists() else (root / cfg_path).resolve()
        raw = load_yaml_file(cfg_path)
        r = _section(raw, "robot")
        rt = _section(raw, "runtime")
        bus = _section(raw, "bus")
        llm_raw = _section(raw, "llm")
        cb_raw = _section(raw, "context_budget")
        audio_raw = _section(raw, "audio")
        wake_words = r.get("wake_words", ["sprocket", "hey sprocket", "robot", "hey robot"])
        if isinstance(wake_words, str):
            # `wake_words: "sprocket"` would otherwise become one wake word per letter.
            wake_words = [wake_words]
        if not isinstance(wake_words, list):
            raise ValueError("robot.wake_words must be a list of phrases")

        def _resolve(p: str, default_rel: bool = False) -> Path:
            expanded = Path(p).expanduser()
            if expanded.is_absolute():
                return expanded
            return (root / expanded).resolve() if default_rel else expanded.resolve()

        def _llm(llm_raw: dict[str, Any], defaults: LLMConfig) -> LLMConfig:
            extra = llm_raw.get("server_args")
            if extra is not None and not (isinstance(extra, list) and all(isinstance(a, (str, int, float))
                                                                          for a in extra)):
                raise ValueError("llm.server_args must be a list of strings")
            mmproj = llm_raw.get("mmproj_path", str(defaults.mmproj_path))
            return LLMConfig(
                backend=str(llm_raw.get("backend", defaults.backend)),
                auto_start_server=_as_bool(llm_raw.get("auto_start_server", True), "llm.auto_start_server"),
                base_url=str(llm_raw.get("base_url", defaults.base_url)),
                model_name=str(llm_raw.get("model_name", defaults.model_name)),
                model_path=_resolve(str(llm_raw.get("model_path", str(defaults.model_path)))),
                mmproj_path=_resolve(str(mmproj)) if mmproj else Path(),
                llama_server_bin=_resolve(str(llm_raw.get("llama_server_bin", str(defaults.llama_server_bin)))),
                gpu_layers=int(llm_raw.get("gpu_layers", defaults.gpu_layers)),
                threads=int(llm_raw.get("threads", defaults.threads)),
                ctx_size=int(llm_raw.get("ctx_size", defaults.ctx_size)),
                timeout_s=float(llm_raw.get("timeout_s", defaults.timeout_s)),
                temperature=float(llm_raw.get("temperature", defaults.temperature)),
                max_tokens=int(llm_raw.get("max_tokens", defaults.max_tokens)),
                native_tools=llm_raw.get("native_tools", True),
                startup_timeout_s=float(llm_raw.get("startup_timeout_s", defaults.startup_timeout_s)),
                thinking=_as_bool(llm_raw.get("thinking", False), "llm.thinking"),
                thinking_budget_tokens=int(llm_raw.get("thinking_budget_tokens", 512)),
                vision=_as_bool(llm_raw.get("vision", bool(mmproj)), "llm.vision"),
                server_args=[str(a) for a in extra] if extra is not None else None,
            )

        llm = _llm(llm_raw, LLMConfig())
        vision_raw = _section(raw, "vision_llm")
        vision_llm = _llm(vision_raw, LLMConfig(base_url="http://127.0.0.1:8081")) if vision_raw else None
        defaults = ContextBudgetConfig()
        cb = ContextBudgetConfig(**{key: int(cb_raw.get(key, getattr(defaults, key))) for key in vars(defaults)})
        return cls(
            name=str(r.get("name", "Sprocket")),
            wake_words=[str(w).strip().lower() for w in wake_words if str(w).strip()],
            follow_up_window_s=float(r.get("follow_up_window_s", 8.0)),
            idle_sleepy_timeout_s=float(r.get("idle_sleepy_timeout_s", 45.0)),
            emotion_decay_rate_per_s=float(r.get("emotion_decay_rate_per_s", 0.08)),
            runtime_dir=_resolve(str(rt.get("runtime_dir", "~/robot-runtime"))),
            database_path=_resolve(str(rt.get("database_path", "~/robot-runtime/data/memory.db"))),
            recordings_dir=_resolve(str(rt.get("recordings_dir", "~/robot-runtime/data/recordings"))),
            soul_path=_resolve(str(rt.get("soul_path", "config/SOUL.md")), default_rel=True),
            body_config_path=_resolve(str(rt.get("body_config_path", "config/body.yaml")), default_rel=True),
            sim_tick_hz=float(rt.get("sim_tick_hz", 20.0)),
            body_backend=str(rt.get("body_backend", "kinematic")),
            serial_port=str(rt.get("serial_port", "/dev/ttyACM0")),
            camera=str(rt.get("camera", "sim")),
            spatial_map_path=_resolve(str(rt.get("spatial_map_path", "~/robot-runtime/data/spatial_map.json"))),
            max_queue_size=int(bus.get("max_queue_size", 256)),
            record_events=_as_bool(bus.get("record_events", True), "bus.record_events"),
            face_fps=float(rt.get("face_fps", 30.0)),
            asr_mode=str(audio_raw.get("asr_mode", "auto")),
            asr_bin=_resolve(
                str(audio_raw.get("asr_bin", "~/robot-runtime/whisper.cpp/build/bin/whisper-cli"))
            ),
            asr_model_path=_resolve(
                str(audio_raw.get("asr_model_path", "~/robot-runtime/models/ggml-tiny.en.bin"))
            ),
            llm=llm,
            vision_llm=vision_llm,
            context_budget=cb,
            agent=AgentConfig(**_section(raw, "agent")),
            autonomy=AutonomyConfig(**_section(raw, "autonomy")),
            perception=PerceptionConfig(**_section(raw, "perception")),
        )
