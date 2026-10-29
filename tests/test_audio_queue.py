"""Exercise real asynchronous playback rather than a blocking speak() stub."""

from __future__ import annotations

import asyncio

import numpy as np

from robot.audio.service import AudioService
from robot.audio.tts import SynthesizedSpeech
from robot.core.bus import EventBus
from robot.core.events import EventPriority


async def test_realtime_sentences_finish_playing_in_order():
    bus = EventBus()
    audio = AudioService(bus, enable_speaker=False, realtime_sync=True)
    audio.tts.synthesize = lambda text: SynthesizedSpeech(text, np.zeros(2400, dtype=np.float32), 24000, [0.5] * 6)
    events = []
    started = asyncio.Event()

    def observe(event):
        events.append((event.type, event.payload.get("text")))
        if event.type == "audio.speech_started":
            started.set()

    for event_type in ("audio.speech_started", "audio.speech_completed", "audio.interrupted"):
        bus.subscribe(event_type, observe, immediate=True)
    await bus.start()
    await audio.start()
    try:
        await bus.emit("robot.response", "test", {"say": "First"}, correlation_id="one-goal")
        await asyncio.wait_for(started.wait(), 1)
        await bus.emit("robot.response", "test", {"say": "Second"}, correlation_id="one-goal")
        await bus.emit("robot.response", "test", {"say": "Third"}, correlation_id="one-goal")
        await audio.wait_until_done(timeout_s=3)
        assert events == [(kind, text) for text in ("First", "Second", "Third")
                          for kind in ("audio.speech_started", "audio.speech_completed")]
        assert audio.tts_spoken_count == 3
    finally:
        await audio.stop()
        await bus.stop()


async def test_stop_interrupts_real_playback_and_discards_pending_sentence():
    bus = EventBus()
    audio = AudioService(bus, enable_speaker=False, realtime_sync=True)
    audio.tts.synthesize = lambda text: SynthesizedSpeech(text, np.zeros(24000, dtype=np.float32), 24000, [0.5] * 60)
    spoken = []
    started = asyncio.Event()

    def on_started(event):
        spoken.append(event.payload["text"])
        started.set()

    bus.subscribe("audio.speech_started", on_started, immediate=True)
    await bus.start()
    await audio.start()
    try:
        await bus.emit("robot.response", "test", {"say": "First"})
        await asyncio.wait_for(started.wait(), 1)
        await bus.emit("robot.response", "test", {"say": "Must not play"})
        await asyncio.wait_for(bus.emit("action.cancel", "test", {"reason": "user_stop_command"},
                                       priority=EventPriority.CRITICAL), 1)
        assert spoken == ["First"]
        assert not audio.is_speaking
        assert audio._response_queue.empty()
        assert audio._speak_task is None
    finally:
        await audio.stop()
        await bus.stop()
