"""Independent face process runner and UDP bridge service."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from robot.core.bus import EventBus
from robot.core.events import Event
from robot.core.service import ServiceHealth
from robot.face.animator import FaceAnimator, FaceFrameParams

logger = logging.getLogger(__name__)


class FaceProcessBridge:
    """Brain-side service that keeps an in-process FaceAnimator in sync and streams UDP updates to the face process."""

    name = "face_bridge"

    def __init__(
        self,
        bus: EventBus,
        *,
        udp_host: str = "127.0.0.1",
        udp_port: int = 18991,
        spawn_window: bool = False,
        fps: float = 60.0,
    ) -> None:
        self.bus = bus
        self.udp_host = udp_host
        self.udp_port = udp_port
        self.spawn_window = spawn_window
        self.fps = fps
        self.animator = FaceAnimator()
        self._sock: socket.socket | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._tick_task: asyncio.Task[None] | None = None
        self._sub_ids: list[str] = []
        self._running = False
        self.packets_sent: int = 0

    @property
    def current_frame(self) -> FaceFrameParams:
        return self.animator.current

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)

        self._sub_ids.append(
            self.bus.subscribe("expression.updated", self._on_expression_updated, name="face:expr")
        )
        self._sub_ids.append(self.bus.subscribe("face.theme", self._on_face_theme, name="face:theme"))
        self._sub_ids.append(
            self.bus.subscribe("memory.updated", self._on_memory_updated, name="face:mem_theme")
        )

        if self.spawn_window:
            self._spawn_subprocess()

        self._tick_task = asyncio.create_task(self._animation_loop(), name="face:animator")

    async def stop(self) -> None:
        self._running = False
        if self._tick_task and not self._tick_task.done():
            self._tick_task.cancel()
            await asyncio.gather(self._tick_task, return_exceptions=True)

        for sid in self._sub_ids:
            self.bus.unsubscribe(sid)
        self._sub_ids.clear()

        if self._sock is not None:
            if self._proc is not None:
                self._send_udp({"cmd": "quit"})
            self._sock.close()
            self._sock = None

        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            # Waiting for the window to exit must not freeze the event loop (and reflexes) for seconds.
            for escalate in (None, proc.terminate, proc.kill):
                if escalate is not None:
                    escalate()
                try:
                    await asyncio.to_thread(proc.wait, 1.0)
                    break
                except subprocess.TimeoutExpired:
                    continue

    async def health(self) -> ServiceHealth:
        proc_ok = self._tick_task is not None and not self._tick_task.done()
        if self.spawn_window and self._proc is not None and self._proc.poll() is not None:
            proc_ok = False
        return ServiceHealth(
            name=self.name,
            healthy=proc_ok,
            status="ok" if proc_ok else "face_window_exited",
            details={
                "packets_sent": self.packets_sent,
                "theme": self.animator.current.theme,
                "emotion": self.animator.current.emotion,
                "activity": self.animator.current.activity,
            },
        )

    def set_theme(self, theme: str) -> FaceFrameParams:
        self.animator.set_state(theme=theme)
        self._send_udp({"theme": self.animator.current.theme})
        return self.animator.current

    def set_mouth_open(self, value: float | None) -> None:
        self.animator.set_mouth_open(value)
        self._send_udp({"mouth_open": value})

    def _spawn_subprocess(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "robot.face.process",
            "--host",
            self.udp_host,
            "--port",
            str(self.udp_port),
        ]
        # `-m robot.face.process` resolves from the child's cwd; pin it to the package root.
        self._proc = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parents[2]))

    def _send_udp(self, payload: dict[str, Any]) -> None:
        if self._sock is None:
            return
        try:
            raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self._sock.sendto(raw, (self.udp_host, self.udp_port))
            self.packets_sent += 1
        except Exception:
            pass

    async def _animation_loop(self) -> None:
        dt = 1.0 / max(10.0, self.fps)
        while self._running:
            try:
                self.animator.step(dt)
                await asyncio.sleep(dt)
            except asyncio.CancelledError:
                break

    async def _on_expression_updated(self, event: Event) -> None:
        p = event.payload
        self.animator.set_state(
            activity=p.get("activity"),
            emotion=p.get("emotion"),
            intensity=p.get("intensity"),
            gaze_target=p.get("gaze_target"),
            theme=p.get("theme"),
        )
        self._send_udp(
            {
                "activity": self.animator.current.activity,
                "emotion": self.animator.current.emotion,
                "intensity": self.animator.current.intensity,
                "gaze_target": self.animator.current.gaze_target,
                "theme": self.animator.current.theme,
            }
        )

    async def _on_face_theme(self, event: Event) -> None:
        theme = str(event.payload.get("theme", "cyan"))
        self.set_theme(theme)

    async def _on_memory_updated(self, event: Event) -> None:
        # Automatically honor memories about preferred robot eye color!
        rec = event.payload.get("record") or {}
        subj = str(rec.get("subject", "")).lower()
        content = str(rec.get("content", "")).lower()
        if "eye color" in subj or "eye color" in content or "eyes" in subj:
            for candidate in (
                "green",
                "emerald",
                "cyan",
                "amber",
                "gold",
                "blue",
                "violet",
                "purple",
                "coral",
                "red",
            ):
                if candidate in content:
                    self.set_theme(candidate)
                    break


def run_standalone_face_loop(
    *,
    host: str = "127.0.0.1",
    port: int = 18991,
    fps: int = 60,
    max_frames: int | None = None,
    headless: bool = False,
) -> int:
    """Run the independent 60 FPS SDL2/pygame face renderer loop."""
    import pygame

    from robot.face.renderer import PygameFaceRenderer

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.setblocking(False)

    animator = FaceAnimator()
    renderer = PygameFaceRenderer(800, 480, headless=headless)
    renderer.init_surface()
    clock = pygame.time.Clock()
    frames = 0
    running = True

    try:
        while running:
            if not headless:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        running = False
                    elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                        running = False

            # Drain all pending UDP state packets without blocking
            while True:
                try:
                    data, _addr = sock.recvfrom(65535)
                except OSError:
                    # Empty queue, ICMP reset, or (Windows) an oversized datagram: never crash the face.
                    break
                try:
                    msg = json.loads(data.decode("utf-8"))
                    if msg.get("cmd") == "quit":
                        running = False
                        break
                    if "mouth_open" in msg:
                        animator.set_mouth_open(msg.get("mouth_open"))
                    animator.set_state(
                        activity=msg.get("activity"),
                        emotion=msg.get("emotion"),
                        intensity=msg.get("intensity"),
                        gaze_target=msg.get("gaze_target"),
                        theme=msg.get("theme"),
                    )
                except Exception:
                    pass

            dt = 1.0 / max(1, fps)
            params = animator.step(dt)
            renderer.render(params)
            frames += 1
            if max_frames is not None and frames >= max_frames:
                break
            if not headless:
                clock.tick(fps)
            else:
                time.sleep(0.001)
    finally:
        sock.close()
        pygame.quit()
    return frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Standalone 60 FPS SDL2 Robot Face Process")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18991)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args(argv)
    run_standalone_face_loop(
        host=args.host,
        port=args.port,
        fps=args.fps,
        max_frames=args.max_frames,
        headless=args.headless,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
