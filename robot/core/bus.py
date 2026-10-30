"""Asynchronous prioritized event bus with bounded queues and drop policies."""

from __future__ import annotations

import asyncio
import fnmatch
import heapq
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from robot.core.clock import Clock, RealClock
from robot.core.events import DropPolicy, Event, EventPriority, default_drop_policy

logger = logging.getLogger(__name__)

EventHandler = Callable[[Event], Awaitable[None] | None]


class BusOverloaded(RuntimeError):
    """A reliable subscriber is full; publishers must retry or report busy."""


class EventQueue(asyncio.PriorityQueue):
    """Bounded heap with explicit removal, preserving queue task accounting."""

    def discard(self, predicate, *, one: bool = False) -> int:
        candidates = sorted(self._queue, key=lambda item: item[1])
        removed = 0
        for item in candidates:
            if predicate(item[2]):
                self._queue.remove(item)
                self.task_done()
                removed += 1
                if one:
                    break
        if removed:
            heapq.heapify(self._queue)
        return removed


@dataclass
class Subscription:
    """Active subscriber registration with its own bounded queue and worker."""

    id: str
    pattern: str
    handler: EventHandler
    max_queue_size: int = 256
    immediate: bool = False
    lossy: bool = False
    queue: asyncio.PriorityQueue[tuple[int, int, Event]] = field(init=False)
    task: asyncio.Task[None] | None = None
    delivered_count: int = 0
    dropped_count: int = 0
    error_count: int = 0

    def __post_init__(self) -> None:
        self.queue = EventQueue(maxsize=self.max_queue_size)

    def matches(self, event_type: str) -> bool:
        if self.pattern == "*":
            return True
        if "*" in self.pattern or "?" in self.pattern:
            return fnmatch.fnmatchcase(event_type, self.pattern)
        return self.pattern == event_type


class EventBus:
    """In-process typed event bus with priority ordering, backpressure, and recording hooks."""

    def __init__(
        self,
        clock: Clock | None = None,
        default_queue_size: int = 256,
    ) -> None:
        self.clock: Clock = clock or RealClock()
        self.default_queue_size = default_queue_size
        self._subscriptions: dict[str, Subscription] = {}
        self._seq: int = 0
        self._sub_seq: int = 0
        self._active_handlers: int = 0
        self._running: bool = False
        self._record_hooks: list[Callable[[Event], None]] = []
        self.published_count: int = 0
        self.dropped_count: int = 0
        self._background_tasks: set[asyncio.Task] = set()
        self._match_cache: dict[str, tuple[Subscription, ...]] = {}

    def add_record_hook(self, hook: Callable[[Event], None]) -> None:
        """Attach a synchronous recorder hook invoked on every published event."""
        self._record_hooks.append(hook)

    def subscribe(
        self,
        pattern: str,
        handler: EventHandler,
        *,
        name: str | None = None,
        max_queue_size: int | None = None,
        immediate: bool = False,
        lossy: bool = False,
    ) -> str:
        """Subscribe `handler` to events matching `pattern` (exact or glob such as `action.*`)."""
        self._sub_seq += 1
        sub_id = name or f"sub_{self._sub_seq}"
        if sub_id in self._subscriptions:
            sub_id = f"{sub_id}_{self._sub_seq}"
        sub = Subscription(
            id=sub_id,
            pattern=pattern,
            handler=handler,
            max_queue_size=max_queue_size or self.default_queue_size,
            immediate=immediate,
            lossy=lossy,
        )
        self._subscriptions[sub_id] = sub
        self._match_cache.clear()
        if self._running and not immediate:
            sub.task = asyncio.create_task(self._run_subscriber(sub), name=f"bus:{sub.id}")
        return sub_id

    def unsubscribe(self, sub_id: str) -> None:
        """Remove a subscription and cancel its worker task."""
        sub = self._subscriptions.pop(sub_id, None)
        self._match_cache.clear()
        if sub and sub.task and not sub.task.done():
            sub.task.cancel()

    async def start(self) -> None:
        """Start subscriber dispatch loops."""
        if self._running:
            return
        self._running = True
        for sub in self._subscriptions.values():
            if not sub.immediate and (sub.task is None or sub.task.done()):
                sub.task = asyncio.create_task(self._run_subscriber(sub), name=f"bus:{sub.id}")

    async def stop(self) -> None:
        """Drain pending items and stop subscriber tasks cleanly."""
        if not self._running:
            return
        await self.drain(timeout=2.0)
        self._running = False
        tasks = [s.task for s in self._subscriptions.values() if s.task and not s.task.done()]
        tasks.extend(task for task in self._background_tasks if not task.done())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def drain(self, timeout: float = 2.0) -> None:
        """Wait until all subscriber queues and cascading handlers have settled."""
        # Deadlines must expire even when a replay clock is paused.
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            pending = sum(s.queue.qsize() for s in self._subscriptions.values())
            if (
                pending == 0
                and self._active_handlers == 0
                and not any(not task.done() for task in self._background_tasks)
            ):
                await asyncio.sleep(0)
                pending2 = sum(s.queue.qsize() for s in self._subscriptions.values())
                if (
                    pending2 == 0
                    and self._active_handlers == 0
                    and not any(not task.done() for task in self._background_tasks)
                ):
                    return
            if asyncio.get_running_loop().time() >= deadline:
                return
            await asyncio.sleep(0.002)

    async def emit(
        self,
        event_type: str,
        source: str,
        payload: dict[str, Any] | None = None,
        *,
        priority: EventPriority = EventPriority.NORMAL,
        correlation_id: str | None = None,
    ) -> Event:
        """Construct and publish an Event with the bus clock's current timestamp."""
        event = Event(
            type=event_type,
            source=source,
            payload=dict(payload or {}),
            ts=self.clock.now(),
            priority=priority,
            correlation_id=correlation_id,
        )
        await self.publish(event)
        return event

    async def publish(self, event: Event) -> Event:
        """Publish an Event to all matching subscribers."""
        if event.ts == 0.0:
            event = Event(
                id=event.id,
                type=event.type,
                source=event.source,
                payload=event.payload,
                ts=self.clock.now(),
                priority=event.priority,
                correlation_id=event.correlation_id,
            )

        self.published_count += 1
        subscribers = self._match_cache.get(event.type)
        if subscribers is None:
            subscribers = tuple(sub for sub in self._subscriptions.values() if sub.matches(event.type))
            if len(self._match_cache) >= 64:
                self._match_cache.pop(next(iter(self._match_cache)))
            self._match_cache[event.type] = subscribers
        if event.priority != EventPriority.CRITICAL:
            for sub in subscribers:
                if sub.immediate or sub.lossy or not sub.queue.full():
                    continue
                if default_drop_policy(event.type, event.priority) != DropPolicy.DROP_OLDEST and not any(
                    default_drop_policy(item[2].type, item[2].priority) == DropPolicy.DROP_OLDEST
                    for item in sub.queue._queue
                ):
                    raise BusOverloaded(f"Subscriber {sub.id!r} is full")
        # Safety handlers run before recording or any potentially saturated queue.
        for sub in subscribers:
            if sub.immediate:
                try:
                    res = sub.handler(event)
                    if isinstance(res, Awaitable):
                        await res
                    sub.delivered_count += 1
                except Exception:
                    sub.error_count += 1
                    if event.priority != EventPriority.CRITICAL:
                        raise
                    logger.exception("Critical observer %s failed; continuing safety delivery", sub.id)
        for hook in self._record_hooks:
            try:
                hook(event)
            except Exception:
                logger.exception("Event recorder hook failed for %s", event.type)

        policy = default_drop_policy(event.type, event.priority)

        for sub in subscribers:
            if sub.immediate:
                continue
            self._seq += 1
            item = (int(event.priority), self._seq, event)

            if sub.queue.full():
                removed = sub.queue.discard(
                    lambda queued: (
                        default_drop_policy(queued.type, queued.priority) == DropPolicy.DROP_OLDEST
                    ),
                    one=True,
                )
                if (
                    removed
                    or sub.lossy
                    or policy == DropPolicy.DROP_OLDEST
                    or event.priority == EventPriority.CRITICAL
                ):
                    sub.dropped_count += 1
                    self.dropped_count += 1
                    if removed:
                        sub.queue.put_nowait(item)
                else:
                    # Never await our own worker's queue: that can deadlock a cascade.
                    raise BusOverloaded(f"Subscriber {sub.id!r} is full")
            else:
                sub.queue.put_nowait(item)

        return event

    async def _run_subscriber(self, sub: Subscription) -> None:
        while self._running:
            try:
                _prio, _seq, event = await sub.queue.get()
            except asyncio.CancelledError:
                break
            self._active_handlers += 1
            try:
                res = sub.handler(event)
                if asyncio.iscoroutine(res) or isinstance(res, Awaitable):
                    await res
                sub.delivered_count += 1
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is None or current.cancelling():
                    break
                # The handler awaited something that was cancelled; the worker itself was not.
                sub.error_count += 1
                logger.warning("Subscriber %s was cancelled internally handling %s", sub.id, event.type)
            except Exception:
                sub.error_count += 1
                logger.exception(
                    "Subscriber %s failed handling event %s (%s)",
                    sub.id,
                    event.type,
                    event.id,
                )
            finally:
                self._active_handlers = max(0, self._active_handlers - 1)
                sub.queue.task_done()

    def discard_pending(self, event_type: str) -> int:
        """Invalidate queued work synchronously when its controlling turn ends."""
        return sum(
            sub.queue.discard(lambda ev: ev.type == event_type) for sub in self._subscriptions.values()
        )

    def track_task(self, task: asyncio.Task) -> None:
        """Include explicitly owned work in drain/shutdown without blocking a subscriber."""
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def stats(self) -> dict[str, Any]:
        """Return observability metrics for the bus and each subscriber."""
        return {
            "published_count": self.published_count,
            "dropped_count": self.dropped_count,
            "subscribers": {
                sub.id: {
                    "pattern": sub.pattern,
                    "queued": sub.queue.qsize(),
                    "delivered": sub.delivered_count,
                    "dropped": sub.dropped_count,
                    "errors": sub.error_count,
                }
                for sub in self._subscriptions.values()
            },
        }
