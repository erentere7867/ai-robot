"""Configuration: typed settings loaded from config/robot.yaml."""

from __future__ import annotations

import sys
import types
import typing
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not (PROJECT_ROOT / "config" / "robot.yaml").exists():
    PROJECT_ROOT = Path(sys.prefix) / "share" / "sprocket-robot"


def _runtime(relative: str) -> Path:
    return Path("~/robot-runtime").expanduser() / relative


def load_yaml_file(path: Path | str, _depth: int = 0) -> dict[str, Any]:
    """Load a YAML file into a dictionary. `extends: other.yaml` (relative to this file) starts from that file;
    each section given here replaces keys of the same section there, and a section set to null removes it."""
    file_path = Path(path).expanduser().resolve()
    data = dict(yaml.safe_load(file_path.read_text(encoding="utf-8")) or {})
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


@dataclass
class LLMConfig:
    backend: str = "auto"  # "auto" | "llamacpp" | "null"
    auto_start_server: bool = True
    base_url: str = "http://127.0.0.1:8080"
    model_name: str = "LFM3-VL-2B-Q4_K_M"
    model_path: Path = field(default_factory=lambda: Path("~/Downloads/LFM3-VL-2B-Q4_K_M.gguf").expanduser())
    mmproj_path: Path = field(default_factory=lambda: Path("~/Downloads/mmproj-LFM3-VL-2B-F16.gguf").expanduser())
    llama_server_bin: Path = field(default_factory=lambda: _runtime(
        "llama.cpp/llama-server.exe" if sys.platform == "win32" else "llama.cpp/build/bin/llama-server"))
    gpu_layers: int = 99
    threads: int = 0  # CPU threads for llama-server (0 = its default)
    ctx_size: int = 8192
    timeout_s: float = 90.0
    temperature: float = 0.0
    max_tokens: int = 256
    startup_timeout_s: float = 90.0
    # Extra llama-server flags for this model, e.g. ["--swa-full", "--checkpoint-min-step", "256"] for LFM3.
    server_args: list[str] | None = None


@dataclass
class AgentConfig:
    max_steps: int = 14
    timeout_s: float = 300.0


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


@dataclass
class PerceptionConfig:
    """The fast layer: a small detector on every frame, a control loop, and the real-time layout."""

    fast_detector: str = "auto"  # auto: the trained detector if there is one; off: the VLM only
    # 2D lidar: "off", "sim" (raycast in the sandbox) or a serial port for an LD19/LD06 ("/dev/ttyUSB0").
    lidar: str = "off"
    confidence: float = 0.4
    detector_gpu: bool = False  # run the detector on the GPU through Vulkan (NCNN)
    detector_threads: int = 4  # CPU threads for the detector when it runs on the CPU
    loop_hz: float = 20.0
    realtime_priority: int = 0  # SCHED_FIFO priority for the control thread (0: leave it normal)
    tune_process: bool = False  # GC freeze, 1 ms GIL switch interval, mlockall (on the robot)
    localization: bool = False  # correct odometry drift by matching lidar scans to the map


@dataclass
class RobotConfig:
    name: str = "Sprocket"
    wake_words: list[str] = field(default_factory=lambda: ["sprocket", "hey sprocket", "robot", "hey robot"])
    follow_up_window_s: float = 8.0
    idle_sleepy_timeout_s: float = 45.0
    emotion_decay_rate_per_s: float = 0.08
    runtime_dir: Path = field(default_factory=lambda: _runtime(""))
    database_path: Path = field(default_factory=lambda: _runtime("data/memory.db"))
    recordings_dir: Path = field(default_factory=lambda: _runtime("data/recordings"))
    soul_path: Path = field(default_factory=lambda: PROJECT_ROOT / "config" / "SOUL.md")
    body_config_path: Path = field(default_factory=lambda: PROJECT_ROOT / "config" / "body.yaml")
    sim_tick_hz: float = 20.0
    # "kinematic" | "physics" (simulated) | "ventuno" (the VENTUNO Q's on-board STM32H5, firmware/ventuno_mcu)
    body_backend: str = "kinematic"
    camera: str = "csi"  # on the robot: "csi" (MIPI cameras through the ISP, GStreamer) | "usb"
    spatial_map_path: Path = field(default_factory=lambda: _runtime("data/spatial_map.json"))
    max_queue_size: int = 256
    record_events: bool = True
    face_fps: float = 30.0
    asr_bin: Path = field(default_factory=lambda: _runtime("whisper.cpp/build/bin/whisper-cli"))
    asr_model_path: Path = field(default_factory=lambda: _runtime("models/ggml-tiny.en.bin"))
    llm: LLMConfig = field(default_factory=LLMConfig)
    context_budget: ContextBudgetConfig = field(default_factory=ContextBudgetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    autonomy: AutonomyConfig = field(default_factory=AutonomyConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)

    @classmethod
    def from_yaml(cls, path: Path | str = "config/robot.yaml", project_root: Path | None = None) -> RobotConfig:
        root = (project_root or PROJECT_ROOT).resolve()
        cfg_path = Path(path)
        if not cfg_path.is_absolute():
            in_cwd = (Path.cwd() / cfg_path).resolve()
            cfg_path = in_cwd if in_cwd.exists() else (root / cfg_path).resolve()
        raw = load_yaml_file(cfg_path)
        flat: dict[str, Any] = {}
        for section in FLAT_SECTIONS:
            flat.update(raw.pop(section, None) or {})
        return _build(cls, {**flat, **raw}, "config", root)


FLAT_SECTIONS = ("robot", "runtime", "bus", "audio")  # grouped in the file, flat in RobotConfig


def _build(cls: type, raw: dict[str, Any], where: str, root: Path) -> Any:
    hints = typing.get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{where}: unknown setting(s) {sorted(unknown)}")
    return cls(**{key: _coerce(hints[key], value, f"{where}.{key}", root) for key, value in raw.items()})


def _coerce(hint: Any, value: Any, where: str, root: Path) -> Any:
    """A value from the file as the type the setting declares."""
    if isinstance(hint, types.UnionType):
        return None if value is None else _coerce(next(a for a in typing.get_args(hint) if a is not type(None)), value,
                                                  where, root)
    if hint is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{where} must be true or false")
        return value
    if hint in (int, float, str):
        return hint(value)
    if hint is Path:
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else (root / path).resolve()
    if typing.get_origin(hint) is list:
        items = [value] if isinstance(value, str) else value  # `wake_words: "sprocket"` is one word, not six letters
        return [str(item) for item in items]
    if value is not None and not isinstance(value, dict):
        raise ValueError(f"{where} must be a section")
    return _build(hint, value or {}, where, root)  # a nested section
