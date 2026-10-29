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


def load_yaml_file(path: Path | str) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    file_path = Path(path).expanduser().resolve()
    text = file_path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text)
        return dict(data or {})
    raise RuntimeError(f"PyYAML is required to load {file_path}")


@dataclass
class ContextBudgetConfig:
    total_tokens: int = 2048
    identity_tokens: int = 320
    memories_tokens: int = 320
    world_state_tokens: int = 200
    tools_tokens: int = 360
    recent_turns_tokens: int = 700
    max_recent_turns: int = 8
    response_reserve_tokens: int = 256

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
        default_factory=lambda: Path("~/Downloads/mmproj-LFM3-VL-2B-Q8_0.gguf").expanduser()
    )
    llama_server_bin: Path = field(
        default_factory=lambda: Path(
            "~/robot-runtime/llama.cpp/llama-server.exe"
            if sys.platform == "win32"
            else "~/robot-runtime/llama.cpp/build/bin/llama-server"
        ).expanduser()
    )
    gpu_layers: int = 99
    ctx_size: int = 4096
    timeout_s: float = 25.0
    temperature: float = 0.0
    max_tokens: int = 220
    native_tools: bool = True

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "null", "llamacpp"}:
            raise ValueError("Unknown LLM backend")
        if not isinstance(self.native_tools, bool):
            raise ValueError("llm.native_tools must be boolean")
        if not math.isfinite(self.timeout_s) or not 0 < self.timeout_s <= 120:
            raise ValueError("LLM timeout must be in (0, 120] seconds")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("LLM temperature must be in [0, 2]")
        if not 128 <= self.ctx_size <= 131072 or not 1 <= self.max_tokens < self.ctx_size:
            raise ValueError("Invalid LLM context/output limit")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not (PROJECT_ROOT / "config" / "robot.yaml").exists():
    PROJECT_ROOT = Path(sys.prefix) / "share" / "companion-robot"


@dataclass
class AgentConfig:
    max_steps: int = 32
    max_actions: int = 200
    timeout_s: float = 240.0

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
    idle_after_s: float = 12.0
    think_interval_s: float = 30.0
    observation_interval_s: float = 1.0

    def __post_init__(self) -> None:
        for key in ("idle_after_s", "think_interval_s", "observation_interval_s"):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0.1:
                raise ValueError(f"autonomy.{key} must be finite and at least 0.1 seconds")
        if not isinstance(self.enabled, bool):
            raise ValueError("Autonomy switches must be booleans")


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
    context_budget: ContextBudgetConfig = field(default_factory=ContextBudgetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    autonomy: AutonomyConfig = field(default_factory=AutonomyConfig)

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
        r = raw.get("robot", {})
        rt = raw.get("runtime", {})
        bus = raw.get("bus", {})
        llm_raw = raw.get("llm", {})
        cb_raw = raw.get("context_budget", {})

        def _resolve(p: str, default_rel: bool = False) -> Path:
            expanded = Path(p).expanduser()
            if expanded.is_absolute():
                return expanded
            return (root / expanded).resolve() if default_rel else expanded.resolve()

        llm = LLMConfig(
            backend=str(llm_raw.get("backend", "auto")),
            auto_start_server=bool(llm_raw.get("auto_start_server", True)),
            base_url=str(llm_raw.get("base_url", "http://127.0.0.1:8080")),
            model_name=str(llm_raw.get("model_name", "LFM3-VL-2B-Q4_K_M")),
            model_path=_resolve(str(llm_raw.get("model_path", "~/Downloads/LFM3-VL-2B-Q4_K_M.gguf"))),
            mmproj_path=_resolve(
                str(llm_raw.get("mmproj_path", "~/Downloads/mmproj-LFM3-VL-2B-Q8_0.gguf"))
            ),
            llama_server_bin=_resolve(
                str(llm_raw.get("llama_server_bin", str(LLMConfig().llama_server_bin)))
            ),
            gpu_layers=int(llm_raw.get("gpu_layers", 99)),
            ctx_size=int(llm_raw.get("ctx_size", 4096)),
            timeout_s=float(llm_raw.get("timeout_s", 25.0)),
            temperature=float(llm_raw.get("temperature", 0.0)),
            max_tokens=int(llm_raw.get("max_tokens", 220)),
            native_tools=llm_raw.get("native_tools", True),
        )
        cb = ContextBudgetConfig(
            total_tokens=int(cb_raw.get("total_tokens", 2048)),
            identity_tokens=int(cb_raw.get("identity_tokens", 320)),
            memories_tokens=int(cb_raw.get("memories_tokens", 320)),
            world_state_tokens=int(cb_raw.get("world_state_tokens", 200)),
            tools_tokens=int(cb_raw.get("tools_tokens", 360)),
            recent_turns_tokens=int(cb_raw.get("recent_turns_tokens", 700)),
            max_recent_turns=int(cb_raw.get("max_recent_turns", 8)),
            response_reserve_tokens=int(cb_raw.get("response_reserve_tokens", 256)),
        )
        return cls(
            name=str(r.get("name", "Sprocket")),
            wake_words=[
                str(w).lower()
                for w in r.get("wake_words", ["sprocket", "hey sprocket", "robot", "hey robot"])
            ],
            follow_up_window_s=float(r.get("follow_up_window_s", 8.0)),
            idle_sleepy_timeout_s=float(r.get("idle_sleepy_timeout_s", 45.0)),
            emotion_decay_rate_per_s=float(r.get("emotion_decay_rate_per_s", 0.08)),
            runtime_dir=_resolve(str(rt.get("runtime_dir", "~/robot-runtime"))),
            database_path=_resolve(str(rt.get("database_path", "~/robot-runtime/data/memory.db"))),
            recordings_dir=_resolve(str(rt.get("recordings_dir", "~/robot-runtime/data/recordings"))),
            soul_path=_resolve(str(rt.get("soul_path", "config/SOUL.md")), default_rel=True),
            body_config_path=_resolve(str(rt.get("body_config_path", "config/body.yaml")), default_rel=True),
            sim_tick_hz=float(rt.get("sim_tick_hz", 20.0)),
            max_queue_size=int(bus.get("max_queue_size", 256)),
            record_events=bool(bus.get("record_events", True)),
            face_fps=float(rt.get("face_fps", 30.0)),
            asr_mode=str(raw.get("audio", {}).get("asr_mode", "auto")),
            asr_bin=_resolve(
                str(raw.get("audio", {}).get("asr_bin", "~/robot-runtime/whisper.cpp/build/bin/whisper-cli"))
            ),
            asr_model_path=_resolve(
                str(raw.get("audio", {}).get("asr_model_path", "~/robot-runtime/models/ggml-tiny.en.bin"))
            ),
            llm=llm,
            context_budget=cb,
            agent=AgentConfig(**raw.get("agent", {})),
            autonomy=AutonomyConfig(**raw.get("autonomy", {})),
        )
