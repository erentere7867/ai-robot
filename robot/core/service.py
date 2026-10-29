"""Service protocol and supervisor with health checks and crash recovery."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from robot.core.bus import EventBus
from robot.core.events import EventPriority

logger = logging.getLogger(__name__)


@dataclass
class ServiceHealth:
    """Health report returned by a managed service."""

    name: str
    healthy: bool = True
    status: str = "ok"
    details: dict[str, Any] = field(default_factory=dict)


class Service(Protocol):
    """Lifecycle interface implemented by all background services."""

    name: str

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def health(self) -> ServiceHealth: ...


class ServiceSupervisor:
    """Coordinates startup, shutdown, health checks, and restarts of managed services."""

    def __init__(self, bus: EventBus, health_interval_s: float = 5.0) -> None:
        self.bus = bus
        self.health_interval_s = health_interval_s
        self._services: list[Service] = []
        self._running = False
        self._monitor_task: asyncio.Task[None] | None = None
        self.restart_counts: dict[str, int] = {}
        self._started_services: list[Service] = []

    def register(self, service: Service) -> None:
        self._services.append(service)
        self.restart_counts.setdefault(service.name, 0)

    async def start_all(self) -> None:
        """Start all registered services in registration order."""
        self._running = True
        for svc in self._services:
            # Include the current service so its partial startup is unwound too.
            self._started_services.append(svc)
            try:
                await svc.start()
            except BaseException:
                await self.stop_all()
                raise
            await self.bus.emit(
                "system.service_started",
                source="supervisor",
                payload={"service": svc.name},
                priority=EventPriority.TELEMETRY,
            )
        if self.health_interval_s > 0:
            self._monitor_task = asyncio.create_task(
                self._monitor_loop(),
                name="supervisor:health",
            )

    async def stop_all(self) -> None:
        """Stop all registered services in reverse registration order."""
        self._running = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            await asyncio.gather(self._monitor_task, return_exceptions=True)

        for svc in reversed(self._started_services):
            try:
                await svc.stop()
            except Exception:
                logger.exception("Error stopping service %s", svc.name)
        self._started_services.clear()

    async def check_health(self) -> dict[str, ServiceHealth]:
        """Collect health status across all services and attempt recovery if unhealthy."""
        report: dict[str, ServiceHealth] = {}
        for svc in self._services:
            try:
                h = await svc.health()
            except Exception as exc:
                h = ServiceHealth(name=svc.name, healthy=False, status=f"error: {exc}")
            report[svc.name] = h
            if not h.healthy and self._running:
                logger.warning("Service %s unhealthy (%s); restarting", svc.name, h.status)
                try:
                    await svc.stop()
                    await svc.start()
                    self.restart_counts[svc.name] = self.restart_counts.get(svc.name, 0) + 1
                except Exception:
                    logger.exception("Failed to restart service %s", svc.name)
        return report

    async def _monitor_loop(self) -> None:
        while self._running:
            try:
                # A background loop must never auto-advance a ReplayClock.
                await asyncio.sleep(self.health_interval_s)
                if not self._running:
                    break
                report = await self.check_health()
                await self.bus.emit(
                    "system.health",
                    source="supervisor",
                    payload={
                        name: {
                            "healthy": h.healthy,
                            "status": h.status,
                            "restarts": self.restart_counts.get(name, 0),
                            **h.details,
                        }
                        for name, h in report.items()
                    },
                    priority=EventPriority.TELEMETRY,
                )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Supervisor health check failed")
