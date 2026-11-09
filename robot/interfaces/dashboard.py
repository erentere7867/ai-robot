"""Zero-dependency async HTTP diagnostic dashboard and REST API server."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from robot.core.events import CancelReason, Event, EventPriority
from robot.core.service import ServiceHealth
from robot.perception.camera import encode_png, fit_width
from robot.world.occupancy import CELL_MM

if TYPE_CHECKING:
    from robot.main import RobotHarness

logger = logging.getLogger(__name__)

# Events that explain a decision. They are tapped synchronously (cheap deque appends) so none is
# lost to the lossy dashboard queue, and so the camera frame of a view can be paired with it.
THINKING_PATTERNS = (
    "cognition.*",
    "skill.*",
    "perception.view",
    "memory.retrieved",
    "robot.response",
    "turn.cancelled",
    "reflex.triggered",
)
MAX_THINKING_ENTRIES = 240
KEPT_VIEW_FRAMES = 6
COVERAGE_DOWNSAMPLE = 2  # spatial-map cells per coverage cell sent to the browser


def _clip_text(value: Any, limit: int) -> str:
    text = str(value if value is not None else "")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _clip_json(value: Any, limit: int) -> Any:
    """A JSON-safe copy of `value`, or a truncated JSON string when its encoding is long."""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str, allow_nan=False)
    except (TypeError, ValueError):
        return _clip_text(value, limit)
    return json.loads(text) if len(text) <= limit else text[: limit - 1] + "…"


def _query_flag(query: dict[str, list[str]], key: str) -> bool:
    return (query.get(key) or [""])[0].strip().lower() in {"1", "true", "yes", "on"}


DASHBOARD_HTML = Path(__file__).with_name("dashboard.html").read_bytes()


class DashboardServer:
    """Lightweight asyncio HTTP server providing live diagnostics and interactive controls."""

    name = "dashboard"

    def __init__(
        self,
        harness: RobotHarness,
        *,
        host: str = "0.0.0.0",
        port: int = 8090,
        max_events: int = 120,
    ) -> None:
        self.harness = harness
        self.host = host
        self.port = port
        self.bound_port: int = port
        self._server: asyncio.AbstractServer | None = None
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._last_chat_correlation: str | None = None
        self._conversations: deque[dict[str, Any]] = deque(maxlen=30)
        self._sub_ids: list[str] = []
        self._last_user_text: str = ""
        self._last_action = "none"
        self._memory_cache: dict[str, Any] | None = None
        self._memory_revision = -1
        self._camera_cache: dict[str, tuple[Any, str]] = {}
        self._clients: set[asyncio.Task] = set()
        # "Thinking" stream: why the robot did what it did.
        self._thinking: deque[dict[str, Any]] = deque(maxlen=MAX_THINKING_ENTRIES)
        self._thinking_seq = 0
        self._thinking_turn: str | None = None
        self._view: dict[str, Any] | None = None  # latest camera view the robot itself looked at
        self._view_frames: deque[tuple[int, bytes]] = deque(maxlen=KEPT_VIEW_FRAMES)
        self._view_version = 0

    @property
    def url(self) -> str:
        h = "127.0.0.1" if self.host in ("0.0.0.0", "::", "") else self.host
        return f"http://{h}:{self.bound_port}"

    async def start(self) -> None:
        if self._server is not None:
            return
        self._sub_ids.append(
            self.harness.bus.subscribe(
                "*", self._on_any_event, name="dashboard:tap", max_queue_size=64, lossy=True
            )
        )
        for pattern in THINKING_PATTERNS:
            self._sub_ids.append(
                self.harness.bus.subscribe(
                    pattern, self._on_thinking_event, name=f"dashboard:think:{pattern}", immediate=True
                )
            )
        self._server = await asyncio.start_server(
            self._handle_client,
            host=self.host,
            port=self.port,
            limit=8192,
        )
        sockets = self._server.sockets or []
        if sockets:
            self.bound_port = int(sockets[0].getsockname()[1])

    async def stop(self) -> None:
        for sid in self._sub_ids:
            self.harness.bus.unsubscribe(sid)
        self._sub_ids.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        clients = tuple(self._clients)
        for task in clients:
            task.cancel()
        if clients:
            await asyncio.gather(*clients, return_exceptions=True)

    async def health(self) -> ServiceHealth:
        running = self._server is not None and self._server.is_serving()
        return ServiceHealth(
            name=self.name,
            healthy=running,
            status="ok" if running else "stopped",
            details={"url": self.url},
        )

    def build_state_payload(self) -> dict[str, Any]:
        """Assemble full diagnostic state JSON for `/api/state`."""
        face_params = self.harness.face_bridge.current_frame.to_dict()
        if self.harness.sandbox is not None:
            self.harness.sandbox.update_face_state(face_params)
        if self._memory_cache is None or self._memory_revision != self.harness.memory.revision:
            self._memory_cache = self.harness.memory.dashboard_snapshot()
            self._memory_revision = self.harness.memory.revision
        mem_export = self._memory_cache

        live_pose = self.harness.body.pose().to_dict()
        live_sensors = self.harness.body.sensors().to_dict()
        expression = self.harness.emotion.snapshot()
        world_snapshot = {"pose": live_pose, "sensors": live_sensors, "activity_state": expression.activity,
                          "emotion": expression.emotion, "emotion_intensity": expression.intensity,
                          "last_action_summary": self._last_action, "observations": []}

        services_health = {
            "body": "online",
            "face_bridge": "online",
            "audio": "online" if (self.harness.audio.enable_speaker or self.harness.audio.enable_mic) else "standby",
            "dashboard": "online",
            "autonomy": "active" if self.harness.autonomy.enabled else "idle",
        }

        return {
            "robot_name": self.harness.config.name,
            "fsm_state": str(self.harness.fsm.state),
            "in_follow_up": self.harness.fsm.state == "follow_up"
            and self.harness.clock.monotonic() <= self.harness.fsm.follow_up_until_ts,
            "idle_seconds": round(self.harness.fsm.idle_seconds(), 1),
            "face": face_params,
            "audio": {
                "is_speaking": self.harness.audio.is_speaking,
                "muted": self.harness.audio.muted,
                "enable_speaker": self.harness.audio.enable_speaker,
                "enable_mic": self.harness.audio.enable_mic,
                "sfx_played_count": self.harness.audio.sfx_played_count,
                "tts_spoken_count": self.harness.audio.tts_spoken_count,
                "barge_in_count": self.harness.audio.barge_in_count,
                "last_spoken_text": self.harness.audio.last_spoken_text,
                "last_asr_text": self.harness.audio.last_asr_text,
            },
            "world": world_snapshot,
            "telemetry": {
                "pose": live_pose,
                "sensors": live_sensors,
            },
            "services": services_health,
            "autonomy": self.harness.autonomy.snapshot(),
            "goal": self.harness.arbiter.active_goal,
            "sandbox": None if self.harness.sandbox is None else {
                **self.harness.sandbox.snapshot(self.harness.body.pose()), "cameras": self._camera_views()},
            "memory": {
                "active_memories": mem_export["active_memories"],
                "tombstones": mem_export["tombstones"],
                "recent_episodes": mem_export["recent_episodes"],
                "consolidation_pending": self.harness.consolidator.pending_count,
                "last_consolidation": self.harness.consolidator.last_report.to_dict(),
            },
            "bus_stats": {
                "published_count": self.harness.bus.published_count,
                "dropped_count": self.harness.bus.dropped_count,
            },
            "recording": None
            if self.harness.recorder is None
            else {
                "recorded": self.harness.recorder.recorded_count,
                "dropped": self.harness.recorder.dropped_count,
                "failed": self.harness.recorder.failed,
            },
            "recent_conversations": list(self._conversations),
            "recent_events": list(self._events)[-45:],
        }

    def _on_any_event(self, event: Event) -> None:
        self._events.append(event.to_dict())
        if event.type.startswith("memory."):
            self._memory_cache = None
        if event.type == "attention.engaged":
            self._last_user_text = str(event.payload.get("text", ""))
        elif event.type == "robot.response":
            self._memory_cache = None
            p = event.payload
            if not p.get("say"):
                return
            origin = p.get("origin", "user")
            user_text = self._last_user_text if origin == "user" and event.correlation_id != self._last_chat_correlation else ""
            self._last_chat_correlation = event.correlation_id
            self._conversations.append(
                {
                    "user": user_text,
                    "origin": origin,
                    "say": p.get("say", ""),
                    "emotion": (p.get("emotion") or {}).get("name", "neutral"),
                    "sound": p.get("sound", "none"),
                    "actions": p.get("actions", []),
                    "status": p.get("status", "completed"),
                    "steps": p.get("steps", 0),
                    "goal_state": p.get("goal_state"),
                    "backend": p.get("backend", "unknown"),
                    "latency_ms": p.get("latency_ms", 0.0),
                }
            )

    def _camera_views(self) -> dict[str, str]:
        """The newest frame of each non-front camera, as small PNG data URLs (frames the fast loop already has)."""
        out: dict[str, str] = {}
        for name, (frame, _dets) in list(self.harness.fast.views.items()):
            if name == self.harness.rig[0].name or frame.rgb is None:
                continue
            cached = self._camera_cache.get(name)
            if cached is not None and cached[0] is frame:
                out[name] = cached[1]
                continue
            url = "data:image/png;base64," + base64.b64encode(encode_png(fit_width(frame.rgb, 320))).decode("ascii")
            self._camera_cache[name] = (frame, url)
            out[name] = url
        return out

    # ---- "why" views: spatial map, thinking stream, drives ------------------------------------------
    def build_map_payload(self, *, truth: bool = False) -> dict[str, Any]:
        """The robot's own spatial map (odometry frame) for `/api/map`; `truth` adds simulator ground truth."""
        harness = self.harness
        pose = harness.body.pose()
        smap = harness.spatial_map
        data = smap.snapshot(pose)
        for obj in data["objects"]:
            for key in ("x_mm", "y_mm", "radius_mm", "first_seen", "last_seen"):
                obj[key] = round(float(obj[key]), 1)
            obj["confidence"] = round(float(obj["confidence"]), 2)
        seen = np.asarray(smap.grid.seen, dtype=bool)
        k = COVERAGE_DOWNSAMPLE
        m = seen.shape[0] // k
        coarse = seen[: m * k, : m * k].reshape(m, k, m, k).any(axis=(1, 3))
        model = harness.camera.model
        data.update(
            {
                "robot": {
                    "x_mm": round(pose.x_mm, 1),
                    "y_mm": round(pose.y_mm, 1),
                    "heading_deg": round(pose.heading_deg, 1),
                    "head_tilt_deg": round(pose.head_tilt_deg, 1),
                },
                "half_extent_mm": smap.grid.half,
                "cell_mm": CELL_MM,
                "coverage_grid": {
                    "n": m,
                    "cell_mm": CELL_MM * k,
                    "origin_mm": -smap.grid.half,
                    # Row i is x, column j is y: bits[i * n + j].
                    "bits": "".join("1" if v else "0" for v in coarse.ravel().tolist()),
                },
                "camera": {
                    "hfov_deg": round(float(model.hfov_deg), 1),
                    "coverage_range_mm": 2200.0,
                },
                "prompt_view": smap.render_for_prompt(pose),
            }
        )
        sandbox = harness.sandbox
        if truth and sandbox is not None:
            true_pose = harness.body.true_pose()
            data["sim_truth"] = {
                "room_half_extent_mm": sandbox.room_half_extent_mm,
                "robot": {
                    "x_mm": round(true_pose.x_mm, 1),
                    "y_mm": round(true_pose.y_mm, 1),
                    "heading_deg": round(true_pose.heading_deg, 1),
                },
                "objects": [
                    {
                        key: obj[key]
                        for key in ("id", "label", "shape", "x_mm", "y_mm", "width_mm", "depth_mm", "color")
                    }
                    for obj in sandbox.list_objects()
                ],
            }
        return data

    def build_thinking_payload(self, since: int = 0) -> dict[str, Any]:
        """Thinking entries newer than `since`, plus the goal and the latest view the robot looked at."""
        since = max(0, since)
        reset = since > self._thinking_seq  # the server restarted under a long-lived page
        entries = [e for e in self._thinking if reset or e["seq"] > since]
        arbiter = self.harness.arbiter
        return {
            "now": self.harness.clock.now(),
            "latest_seq": self._thinking_seq,
            "reset": reset,
            "entries": entries,
            "view": self._view,
            "kept_frames": [version for version, _ in self._view_frames],
            "goal": dict(arbiter.active_goal) if arbiter.active_goal else None,
            "busy": arbiter.busy,
            "fsm_state": str(self.harness.fsm.state),
            "backend": self.harness.backend.name,
        }

    def build_autonomy_payload(self) -> dict[str, Any]:
        """Autonomy switches, drive levels and the body's battery for `/api/autonomy`."""
        sensors = self.harness.body.sensors()
        arbiter = self.harness.arbiter
        return {
            **self.harness.autonomy.snapshot(),
            "body": {
                "battery_pct": round(sensors.battery_pct, 1),
                "charging": sensors.charging,
            },
            "goal": dict(arbiter.active_goal) if arbiter.active_goal else None,
            "busy": arbiter.busy,
        }

    def view_frame(self, version: int | None = None) -> bytes | None:
        """PNG of a recent view the robot looked at (the latest when `version` is None)."""
        if not self._view_frames:
            return None
        if version is None:
            return self._view_frames[-1][1]
        return next((png for v, png in self._view_frames if v == version), None)

    def _on_thinking_event(self, event: Event) -> None:
        try:
            self._record_thinking(event)
        except Exception:  # an observer must never disturb the robot
            logger.debug("Dashboard could not record %s", event.type, exc_info=True)

    def _append_thinking(self, kind: str, event: Event, data: dict[str, Any]) -> None:
        self._thinking_seq += 1
        self._thinking.append(
            {"seq": self._thinking_seq, "ts": event.ts, "type": kind, "turn": event.correlation_id, **data}
        )

    def _turn_header(self, turn: str, payload: dict[str, Any]) -> dict[str, Any]:
        goal = self.harness.arbiter.active_goal or {}
        if goal.get("correlation_id") == turn:
            origin = str(goal.get("origin", "user"))
            objective = str(goal.get("objective", ""))
        else:
            origin = str(payload.get("origin", "user"))
            objective = self._last_user_text if origin == "user" else ""
        trigger = self.harness.autonomy.last_trigger if origin == "autonomous" else None
        return {"origin": origin, "objective": _clip_text(objective, 300), "trigger": trigger}

    def _record_thinking(self, event: Event) -> None:
        kind, p, turn = event.type, event.payload, event.correlation_id
        if turn and turn != self._thinking_turn and kind not in ("turn.cancelled", "reflex.triggered"):
            self._thinking_turn = turn
            self._append_thinking("turn.started", event, self._turn_header(turn, p))
        if kind == "cognition.model_call":
            calls = [c for c in (p.get("tool_calls") or []) if isinstance(c, dict)][:8]
            data = {
                "step": p.get("step"),
                "latency_ms": p.get("latency_ms"),
                "prompt_tokens": p.get("prompt_tokens"),
                "completion_tokens": p.get("completion_tokens"),
                "content": _clip_text(p.get("content"), 400),
                "tool_calls": [
                    {"name": _clip_text(c.get("name"), 40), "args": _clip_json(c.get("args"), 300)} for c in calls
                ],
            }
        elif kind == "skill.started":
            data = {"tool": _clip_text(p.get("tool"), 40), "args": _clip_json(p.get("args"), 300)}
        elif kind == "skill.completed":
            data = {
                "tool": _clip_text(p.get("tool"), 40),
                "args": _clip_json(p.get("args"), 300),
                "ok": bool(p.get("ok")),
                "status": _clip_text(p.get("status"), 24),
                "message": _clip_text(p.get("message"), 400),
                "data": _clip_json(p.get("data"), 600),
            }
        elif kind == "perception.view":
            data = self._record_view(event)
        elif kind == "memory.retrieved":
            memories = [m for m in (p.get("memories") or []) if isinstance(m, dict)][:4]
            data = {
                "count": int(p.get("count", len(memories)) or 0),
                "memories": [_clip_text(m.get("content", ""), 140) for m in memories],
            }
        elif kind == "robot.response":
            emotion = p.get("emotion") if isinstance(p.get("emotion"), dict) else {}
            data = {
                "say": _clip_text(p.get("say"), 400),
                "status": _clip_text(p.get("status", "completed"), 24),
                "steps": p.get("steps", 0),
                "backend": _clip_text(p.get("backend", ""), 40),
                "latency_ms": p.get("latency_ms", 0.0),
                "origin": _clip_text(p.get("origin", "user"), 16),
                "next_wake_s": p.get("next_wake_s"),
                "emotion": _clip_text(emotion.get("name", ""), 24),
                "tools": [_clip_text(t.get("tool"), 40) for t in (p.get("actions") or []) if isinstance(t, dict)],
            }
        elif kind == "cognition.recovered_calls":
            data = {"calls": [_clip_text(c, 40) for c in (p.get("calls") or [])][:8]}
        elif kind == "cognition.unavailable":
            data = {"error": _clip_text(p.get("error"), 200)}
        else:  # turn.cancelled, reflex.triggered, any other cognition.* event
            data = {"detail": _clip_json(p, 300)}
        self._append_thinking(kind, event, data)

    def _record_view(self, event: Event) -> dict[str, Any]:
        """Pair a perception.view with the frame it was detected on.

        `perceive()` stores the captured frame on the skill context right before emitting the view,
        and this handler runs synchronously inside that emit, so the pairing is exact.
        """
        p = event.payload
        detections = []
        for det in (p.get("detections") or [])[:16]:
            if not isinstance(det, dict):
                continue
            try:
                bbox = [round(float(v), 3) for v in det.get("bbox", [])][:4]
            except (TypeError, ValueError):
                continue
            if len(bbox) == 4:
                detections.append({"label": _clip_text(det.get("label"), 48), "bbox": bbox})
        frame = self.harness.skill_context.attached_image
        png = frame.png_bytes if frame is not None else None
        has_image = isinstance(png, bytes | bytearray) and len(png) > 0
        self._view_version += 1
        if has_image:
            self._view_frames.append((self._view_version, bytes(png)))
        pose = frame.pose if frame is not None else None
        view = {
            "version": self._view_version,
            "ts": event.ts,
            "turn": event.correlation_id,
            "has_image": has_image,
            "pose": None
            if pose is None
            else {
                "x_mm": round(pose.x_mm, 1),
                "y_mm": round(pose.y_mm, 1),
                "heading_deg": round(pose.heading_deg, 1),
                "head_tilt_deg": round(pose.head_tilt_deg, 1),
            },
            "query": _clip_text(p.get("query"), 60) if p.get("query") else None,
            "detections": detections,
            "seen": [_clip_text(i, 32) for i in (p.get("seen") or [])][:16],
            "lost": [_clip_text(i, 32) for i in (p.get("lost") or [])][:16],
        }
        self._view = view
        return {k: v for k, v in view.items() if k not in ("ts", "turn")}
    @staticmethod
    def _is_allowed_origin(origin: str, host_header: str | None) -> bool:
        if not origin:
            return True
        try:
            import ipaddress
            from urllib.parse import urlparse
            parsed = urlparse(origin)
            if parsed.scheme not in ("http", "https"):
                return False
            if host_header and parsed.netloc.lower() == host_header.lower():
                return True
            orig_host = (parsed.hostname or "").lower()
            if orig_host in ("localhost", "127.0.0.1", "::1") or orig_host.endswith(".ts.net") or "." not in orig_host:
                return True
            ipaddress.ip_address(orig_host)  # any IP-literal origin (LAN, VPN)
            return True
        except Exception:
            pass
        return False


    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        admitted = len(self._clients) < 8
        if admitted:
            self._clients.add(task)
        status, content_type, data = (
            "503 Service Unavailable",
            "application/json",
            b'{"error":"robot is busy"}',
        )
        try:
            if admitted:
                async with asyncio.timeout(5.0):
                    line = await reader.readline()
                    if len(line) > 2048:
                        raise ValueError("Request target is too long")
                    parts = line.decode("ascii").strip().split()
                    if len(parts) != 3:
                        raise ValueError("Malformed request")
                    method, target, _version = parts
                    headers = {}
                    size = 0
                    for _ in range(32):
                        line = await reader.readline()
                        size += len(line)
                        if size > 8192:
                            raise ValueError("Headers are too large")
                        if line == b"\r\n":
                            break
                        if not line or b":" not in line:
                            raise ValueError("Malformed headers")
                        key, value = line.decode("ascii").split(":", 1)
                        key = key.lower().strip()
                        if key in headers:
                            raise ValueError("Duplicate header")
                        headers[key] = value.strip()
                    else:
                        raise ValueError("Too many headers")
                    if "host" not in headers:
                        raise ValueError("Missing Host header")
                    if headers.get("origin") and not self._is_allowed_origin(headers.get("origin"), headers.get("host")):
                        raise PermissionError("Cross-origin control is forbidden")
                    length = int(headers.get("content-length", "0"))
                    if not 0 <= length <= 16384 or "transfer-encoding" in headers:
                        raise ValueError("Invalid or oversized body")
                    if (
                        method == "POST"
                        and headers.get("content-type", "").split(";", 1)[0].strip() != "application/json"
                    ):
                        raise ValueError("Expected application/json")
                    raw = await reader.readexactly(length) if length else b"{}"
                    body = json.loads(
                        raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON"))
                    )
                    if not isinstance(body, dict):
                        raise ValueError("JSON body must be an object")
                parsed = urlparse(target)
                status, content_type, data = await self._route_request(
                    method.upper(), parsed.path, parse_qs(parsed.query), body
                )
        except PermissionError as exc:
            status, data = "403 Forbidden", json.dumps({"error": str(exc)}).encode()
        except (ValueError, TypeError, KeyError, OverflowError, asyncio.IncompleteReadError) as exc:
            status, data = "400 Bad Request", json.dumps({"error": str(exc)}).encode()
        except TimeoutError:
            status, data = "408 Request Timeout", b'{"error":"request timed out"}'
        except RuntimeError as exc:
            status, data = "503 Service Unavailable", json.dumps({"error": str(exc)}).encode()
        except Exception:
            logger.exception("Dashboard request failed")
            status, data = "500 Internal Server Error", b'{"error":"request failed"}'
        finally:
            try:
                header = (
                    f"HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {len(data)}\r\n"
                    "Cache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n"
                    "Content-Security-Policy: frame-ancestors 'none'\r\nConnection: close\r\n\r\n"
                ).encode()
                writer.write(header + data)
                await asyncio.wait_for(writer.drain(), timeout=1.0)
            except (ConnectionError, TimeoutError):
                pass
            writer.close()
            self._clients.discard(task)

    async def _route_request(
        self,
        method: str,
        path: str,
        query: dict[str, list[str]],
        body: dict[str, Any],
    ) -> tuple[str, str, bytes]:
        if method == "GET" and path in ("/", "/index.html"):
            return "200 OK", "text/html; charset=utf-8", DASHBOARD_HTML

        if method == "GET" and path == "/api/state":
            data = self.build_state_payload()
            return "200 OK", "application/json", json.dumps(data).encode("utf-8")

        if method == "GET" and path == "/api/events":
            data = {"events": list(self._events)}
            return "200 OK", "application/json", json.dumps(data).encode("utf-8")

        if method == "GET" and path == "/api/map":
            data = self.build_map_payload(truth=_query_flag(query, "truth"))
            return "200 OK", "application/json", json.dumps(data).encode("utf-8")

        if method == "GET" and path == "/api/thinking":
            since = int((query.get("since") or ["0"])[0] or 0)
            data = self.build_thinking_payload(since)
            return "200 OK", "application/json", json.dumps(data).encode("utf-8")

        if method == "GET" and path == "/api/thinking/frame":
            raw_version = (query.get("v") or [""])[0]
            png = self.view_frame(int(raw_version) if raw_version else None)
            if png is None:
                return "404 Not Found", "application/json", b'{"error":"no camera view kept for that version"}'
            return "200 OK", "image/png", png

        if method == "GET" and path == "/api/autonomy":
            return "200 OK", "application/json", json.dumps(self.build_autonomy_payload()).encode("utf-8")

        if method == "POST" and path == "/api/message":
            text = str(body.get("text", "")).strip()
            if not text:
                return "400 Bad Request", "application/json", b'{"error":"empty text"}'
            self._last_user_text = text
            resp = await self.harness.submit_message(text)
            return "200 OK", "application/json", json.dumps(resp).encode("utf-8")

        if method == "POST" and path == "/api/autonomy":
            if not body or any(k not in {"enabled", "paused", "quiet"} or not isinstance(v, bool) for k, v in body.items()):
                return "400 Bad Request", "application/json", b'{"error":"expected autonomy boolean switches"}'
            await self.harness.autonomy.set_mode(**body)
            return "200 OK", "application/json", json.dumps(self.harness.autonomy.snapshot()).encode("utf-8")

        if method == "POST" and path == "/api/expression":
            if "theme" in body:
                await self.harness.bus.emit(
                    "face.theme",
                    source="dashboard",
                    payload={"theme": str(body["theme"])},
                    priority=EventPriority.HIGH,
                )
            if any(k in body for k in ("emotion", "intensity", "gaze_target")):
                emo = str(body.get("emotion", self.harness.emotion.snapshot().emotion))
                inten = float(body.get("intensity", self.harness.emotion.snapshot().intensity))
                gaze = body.get("gaze_target")
                await self.harness.bus.emit(
                    "expression.manual",
                    source="dashboard",
                    payload={"emotion": emo, "intensity": inten, "gaze_target": gaze},
                    priority=EventPriority.HIGH,
                )
            await self.harness.bus.drain()
            return (
                "200 OK",
                "application/json",
                json.dumps(self.harness.face_bridge.current_frame.to_dict()).encode("utf-8"),
            )

        if method == "POST" and path == "/api/action":
            done = await self.harness.run_manual(str(body.get("tool", "")), dict(body.get("args") or {}))
            self._last_action = done.message
            await self.harness.bus.drain()
            status = "completed" if done.ok else "cancelled" if done.cancelled else "failed"
            return (
                "200 OK",
                "application/json",
                json.dumps(
                    {"ok": done.ok, "status": status, "message": done.message, "pose": self.harness.body.pose().to_dict()}
                ).encode(),
            )

        if path in ("/api/sandbox", "/api/obstacle") and self.harness.sandbox is None:
            return "404 Not Found", "application/json", b'{"error":"this robot is not simulated"}'

        if method == "POST" and path == "/api/sandbox":
            op = str(body.get("op", "snapshot")).strip().lower()
            if op == "move":
                self.harness.sandbox.move_object(
                    str(body.get("id", "")),
                    float(body.get("x_mm", 0.0)),
                    float(body.get("y_mm", 0.0)),
                )
            elif op == "ray_move":
                self.harness.sandbox.move_object_by_overview_ray(
                    str(body.get("id", "")),
                    float(body.get("u_norm", 0.5)),
                    float(body.get("v_norm", 0.5)),
                )
            elif op == "orbit":
                self.harness.sandbox.set_orbit_camera(
                    azimuth_deg=float(body["azimuth_deg"]) if "azimuth_deg" in body else None,
                    elevation_deg=float(body["elevation_deg"]) if "elevation_deg" in body else None,
                    distance_m=float(body["distance_m"]) if "distance_m" in body else None,
                    lookat_m=list(body["lookat_m"]) if "lookat_m" in body else None,
                )
            elif op == "launch_viewer":
                self.harness.sandbox.launch_native_viewer(self.url)
            elif op == "add":
                self.harness.sandbox.add_object(
                    label=str(body.get("label", "Orange Cone")),
                    shape=str(body.get("shape", "cylinder")),
                    x_mm=float(body.get("x_mm", 550.0)),
                    y_mm=float(body.get("y_mm", 0.0)),
                    size_mm=float(body.get("size_mm", 100.0)),
                    color=str(body.get("color", "#f97316")),
                )
            elif op == "remove":
                self.harness.sandbox.remove_object(str(body.get("id", "")))
            elif op == "cliff_zone":
                self.harness.sandbox.set_cliff_zone(bool(body.get("enabled", False)))
            elif op == "reset":
                await self.harness.reset_simulation()
            await self.harness.bus.drain()
            return (
                "200 OK",
                "application/json",
                json.dumps(self.harness.sandbox.snapshot(self.harness.body.pose())).encode("utf-8"),
            )

        if method == "POST" and path == "/api/estop":
            active = bool(body.get("active", False))
            self.harness.body.set_estop(active)
            if active:
                await self.harness.bus.emit(
                    "action.cancel", "dashboard", {"reason": CancelReason.ESTOP}, priority=EventPriority.CRITICAL
                )
            await self.harness.body.publish_state()
            return "200 OK", "application/json", json.dumps({"estop_active": active}).encode("utf-8")

        if method == "POST" and path == "/api/obstacle":
            tof = body.get("front_tof_mm")
            cliff = body.get("cliff")
            self.harness.body.set_simulated_obstacle(
                front_tof_mm=float(tof) if tof is not None else None,
                cliff=bool(cliff) if cliff is not None else None,
            )
            await self.harness.body.publish_state()
            await self.harness.bus.drain()
            return (
                "200 OK",
                "application/json",
                json.dumps({"sensors": self.harness.body.sensors().to_dict()}).encode("utf-8"),
            )

        if method == "POST" and path == "/api/memory/forget":
            topic = str(body.get("topic", "")).strip()
            await self.harness.bus.emit(
                "action.cancel", "dashboard", {"reason": CancelReason.FORGET}, priority=EventPriority.CRITICAL
            )
            count, tomb = self.harness.memory.forget(topic)
            self.harness.prompts.clear()
            self.harness.consolidator.clear_pending()
            self._memory_cache = None
            return (
                "200 OK",
                "application/json",
                json.dumps({"forgotten_count": count, "tombstone": tomb.__dict__}).encode("utf-8"),
            )

        if method == "POST" and path == "/api/memory/consolidate":
            rep = await self.harness.consolidator.consolidate(emit_when_empty=True)
            await self.harness.bus.drain()
            return "200 OK", "application/json", json.dumps(rep.to_dict()).encode("utf-8")

        if method == "POST" and path == "/api/audio/sfx":
            snd = str(body.get("sound", "ack_chirp")).strip().lower()
            samples = await self.harness.audio.play_sfx(snd)
            await self.harness.bus.drain()
            return (
                "200 OK",
                "application/json",
                json.dumps({"sound": snd, "samples": int(len(samples))}).encode("utf-8"),
            )

        if method == "POST" and path == "/api/audio/speak":
            txt = str(body.get("text", "")).strip()
            speech = await self.harness.audio.speak(txt)
            await self.harness.bus.drain()
            return (
                "200 OK",
                "application/json",
                json.dumps(
                    {
                        "text": speech.text,
                        "duration_s": round(speech.duration_s, 3),
                        "backend": speech.backend_name,
                    }
                ).encode("utf-8"),
            )

        if method == "POST" and path == "/api/audio/mute":
            muted = bool(body.get("muted", False))
            self.harness.audio.muted = muted
            return "200 OK", "application/json", json.dumps({"muted": muted}).encode("utf-8")

        if method == "POST" and path == "/api/audio/barge_in":
            await self.harness.bus.emit(
                "audio.barge_in",
                source="dashboard",
                payload={"rms": 0.25},
                priority=EventPriority.CRITICAL,
            )
            await self.harness.bus.drain()
            return "200 OK", "application/json", json.dumps({"barge_in": True}).encode("utf-8")

        return "404 Not Found", "application/json", b'{"error":"not found"}'
