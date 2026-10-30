"""AudioService: coordinates SFX playback, TTS with RMS mouth sync, mic VAD/ASR, and barge-in."""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

try:
    import sounddevice as sd
except (ImportError, OSError):  # pragma: no cover
    sd = None  # type: ignore[assignment]

from robot.audio.asr import SpeechRecognizer
from robot.audio.sfx import PetSoundSynthesizer
from robot.audio.tts import RobotTTSEngine, SynthesizedSpeech
from robot.audio.vad import VoiceActivityDetector
from robot.core.bus import EventBus
from robot.core.clock import RealClock
from robot.core.events import Event, EventPriority
from robot.core.service import ServiceHealth
from robot.face.process import FaceProcessBridge

logger = logging.getLogger(__name__)

MIC_STALL_TIMEOUT_S = 3.0
MIC_RETRY_INITIAL_S = 1.0
MIC_RETRY_MAX_S = 30.0


class AudioService:
    """Manages speaker output, real-time mouth-envelope sync, microphone VAD/ASR, and barge-in."""

    name = "audio_service"

    def __init__(
        self,
        bus: EventBus,
        face_bridge: FaceProcessBridge | None = None,
        *,
        enable_speaker: bool = True,
        enable_mic: bool = False,
        tts_mode: str = "auto",
        asr_mode: str = "auto",
        realtime_sync: bool | None = None,
    ) -> None:
        self.bus = bus
        self.face_bridge = face_bridge
        self.enable_speaker = enable_speaker
        self.enable_mic = enable_mic
        self.realtime_sync = realtime_sync if realtime_sync is not None else isinstance(bus.clock, RealClock)
        effective_tts_mode = tts_mode if tts_mode != "auto" else ("auto" if enable_speaker else "warble")
        self.sfx = PetSoundSynthesizer()
        self.tts = RobotTTSEngine(mode=effective_tts_mode)
        self.vad = VoiceActivityDetector()
        self.asr = SpeechRecognizer(mode=asr_mode)

        self.is_speaking: bool = False
        self.muted: bool = False
        self.sfx_played_count: int = 0
        self.tts_spoken_count: int = 0
        self.barge_in_count: int = 0
        self.last_spoken_text: str = ""
        self.last_asr_text: str = ""
        self.last_mouth_envelope: list[float] = []

        self._cancel_playback: bool = False
        self._speak_task: asyncio.Task[None] | None = None
        self._mic_task: asyncio.Task[None] | None = None
        self._sub_ids: list[str] = []
        self._running: bool = False
        self._speech_epoch = 0
        self._synthesis_lock = asyncio.Lock()
        self._device_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio-device")
        self._pool_closed = False
        self._utterance_task: asyncio.Task | None = None
        self._response_queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=16)
        self._synthesis_future: asyncio.Future | None = None
        self._asr_task: asyncio.Task | None = None
        self._asr_future: asyncio.Future | None = None
        self._asr_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        self._mic_generation = 0
        self._mic_retry_s = MIC_RETRY_INITIAL_S

    async def start(self) -> None:
        if self._running:
            return
        if self._pool_closed:
            self._device_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio-device")
            self._pool_closed = False
        self._running = True
        self._sub_ids.append(
            self.bus.subscribe("expression.updated", self._on_expression_updated, name="audio:expr")
        )
        self._sub_ids.append(
            self.bus.subscribe("robot.response", self._queue_response, name="audio:resp", immediate=True)
        )
        self._sub_ids.append(
            self.bus.subscribe("action.cancel", self._on_cancel, name="audio:cancel", immediate=True)
        )
        self._sub_ids.append(
            self.bus.subscribe("audio.barge_in", self._on_barge_in, name="audio:barge_in", immediate=True)
        )
        if self.enable_mic:
            self.start_mic_listener()

    async def stop(self) -> None:
        self._running = False
        await self.interrupt_playback(reason="service_stop")
        if self._mic_task and not self._mic_task.done():
            self._mic_task.cancel()
            await asyncio.gather(self._mic_task, return_exceptions=True)
            self._mic_task = None
        if self._asr_task:
            self._asr_task.cancel()
            await asyncio.gather(self._asr_task, return_exceptions=True)
            self._asr_task = None
        while not self._asr_queue.empty():
            self._asr_queue.get_nowait()
            self._asr_queue.task_done()
        for sid in self._sub_ids:
            self.bus.unsubscribe(sid)
        self._sub_ids.clear()
        self._device_pool.shutdown(wait=False, cancel_futures=True)
        self._pool_closed = True

    async def health(self) -> ServiceHealth:
        mic_alive = self._mic_task is not None and not self._mic_task.done()
        # Without sounddevice a restart cannot help; it would only cut off speech every check.
        mic_missing = self.enable_mic and sd is None
        status = "speaking" if self.is_speaking else "ok"
        return ServiceHealth(
            name=self.name,
            healthy=not self.enable_mic or mic_missing or mic_alive,
            status="mic_unavailable" if mic_missing else status,
            details={
                "enable_speaker": self.enable_speaker,
                "enable_mic": self.enable_mic,
                "muted": self.muted,
                "sfx_played": self.sfx_played_count,
                "tts_spoken": self.tts_spoken_count,
                "barge_ins": self.barge_in_count,
            },
        )

    async def interrupt_playback(self, *, reason: str = "barge_in") -> None:
        """Immediately halt active speaker playback and reset mouth animation."""
        was_speaking = self.is_speaking or (self._speak_task is not None and not self._speak_task.done())
        self._cancel_playback = True
        self._speech_epoch += 1
        if reason != "new_utterance":
            while not self._response_queue.empty():
                self._response_queue.get_nowait()
                self._response_queue.task_done()
        if (
            self._utterance_task
            and self._utterance_task is not asyncio.current_task()
            and not self._utterance_task.done()
        ):
            self._utterance_task.cancel()
            await asyncio.gather(self._utterance_task, return_exceptions=True)
            self._utterance_task = None
        if reason != "new_utterance":
            self.bus.discard_pending("robot.response")
        if was_speaking and sd is not None and self.enable_speaker:
            try:
                await asyncio.get_running_loop().run_in_executor(self._device_pool, sd.stop)
            except Exception:
                pass
        if self._speak_task and not self._speak_task.done():
            self._speak_task.cancel()
            await asyncio.gather(self._speak_task, return_exceptions=True)
        self._speak_task = None
        self.is_speaking = False
        if self.face_bridge is not None:
            self.face_bridge.set_mouth_open(None)
        if was_speaking:
            await self.bus.emit(
                "audio.interrupted",
                source=self.name,
                payload={"reason": reason},
                priority=EventPriority.HIGH,
            )

    async def wait_until_done(self, timeout_s: float = 10.0) -> None:
        """Wait for any active speech playback task to complete."""
        async with asyncio.timeout(timeout_s):
            if self._utterance_task and not self._utterance_task.done():
                await asyncio.shield(self._utterance_task)
            if self._speak_task is not None and not self._speak_task.done():
                await asyncio.shield(self._speak_task)

    async def play_sfx(self, sound_name: str) -> np.ndarray:
        """Synthesize and play a non-verbal pet sound effect."""
        samples = self.sfx.synthesize(sound_name)
        if len(samples) == 0:
            return samples

        self.sfx_played_count += 1
        if self.enable_speaker and not self.muted and sd is not None:
            try:
                sr = self.sfx.sample_rate
                await asyncio.get_running_loop().run_in_executor(
                    self._device_pool,
                    lambda: sd.play(samples, samplerate=sr, blocking=False),
                )
            except Exception:
                logger.debug("sounddevice play failed for SFX %s", sound_name, exc_info=True)

        await self.bus.emit(
            "audio.sfx_played",
            source=self.name,
            payload={
                "sound": sound_name,
                "duration_ms": round(len(samples) * 1000.0 / self.sfx.sample_rate, 1),
            },
            priority=EventPriority.TELEMETRY,
        )
        return samples

    async def speak(
        self,
        text: str,
        *,
        lead_in_sound: str = "none",
        realtime: bool | None = None,
        correlation_id: str | None = None,
    ) -> SynthesizedSpeech:
        """Synthesize speech, play it, and drive real-time 60 Hz RMS mouth animation."""
        await self.interrupt_playback(reason="new_utterance")
        self._cancel_playback = False
        epoch = self._speech_epoch

        loop = asyncio.get_running_loop()
        async with self._synthesis_lock:
            # Cancelling an await cannot kill a running native synthesizer. Reuse
            # one owned future instead of accumulating jobs in the executor.
            if self._synthesis_future is not None and not self._synthesis_future.done():
                try:
                    await asyncio.shield(self._synthesis_future)
                except Exception:
                    # An abandoned utterance's failure belongs to that utterance, not this one.
                    logger.debug("Abandoned speech synthesis failed", exc_info=True)
            if epoch != self._speech_epoch:
                return SynthesizedSpeech("", np.zeros(0, dtype=np.float32), 24000, [])
            self._synthesis_future = loop.run_in_executor(None, self.tts.synthesize, text[:400])
            speech: SynthesizedSpeech = await asyncio.shield(self._synthesis_future)
            self._synthesis_future = None
        if epoch != self._speech_epoch:
            return speech
        self.last_spoken_text = speech.text
        self.last_mouth_envelope = list(speech.mouth_envelope)
        self.tts_spoken_count += 1

        use_realtime = self.realtime_sync if realtime is None else realtime
        if use_realtime:
            self._speak_task = asyncio.create_task(
                self._run_playback_with_mouth_sync(
                    speech, lead_in_sound=lead_in_sound, realtime=True, correlation_id=correlation_id
                ),
                name="audio:speak",
            )
        else:
            await self._run_playback_with_mouth_sync(
                speech, lead_in_sound=lead_in_sound, realtime=False, correlation_id=correlation_id
            )
        return speech

    async def process_mic_frame(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Feed a 16 kHz mono audio frame into VAD; handle barge-in and transcribe completed utterances."""
        vad_events = self.vad.process_frame(frame, robot_is_speaking=self.is_speaking)
        results: list[dict[str, Any]] = []

        for ev in vad_events:
            if ev.kind == "barge_in":
                self.barge_in_count += 1
                await self.bus.emit(
                    "audio.barge_in",
                    source=self.name,
                    payload={"rms": round(ev.rms, 4)},
                    priority=EventPriority.CRITICAL,
                )
                results.append({"kind": "barge_in", "rms": ev.rms})

            elif ev.kind == "speech_start":
                self._mic_generation += 1
                await self.bus.emit(
                    "speech.detected",
                    source="microphone",
                    payload={"speech_active": True, "rms": round(ev.rms, 4)},
                    priority=EventPriority.HIGH,
                )
                results.append({"kind": "speech_start"})

            elif ev.kind == "speech_end" and ev.audio_samples is not None:
                await self.bus.emit(
                    "speech.detected",
                    source="microphone",
                    payload={"speech_active": False, "samples": len(ev.audio_samples)},
                    priority=EventPriority.NORMAL,
                )
                if self._asr_queue.full():
                    self._asr_queue.get_nowait()
                    self._asr_queue.task_done()
                self._asr_queue.put_nowait((ev, self._mic_generation))
                if self._asr_task is None or self._asr_task.done():
                    self._asr_task = asyncio.create_task(self._recognition_loop(), name="audio:asr")
                results.append({"kind": "speech_end", "pending": True})

        return results

    async def _recognition_loop(self) -> None:
        """One active utterance plus one latest pending utterance; VAD never waits for ASR."""
        while self._running:
            event, generation = await self._asr_queue.get()
            try:
                if self._asr_future is not None and not self._asr_future.done():
                    await asyncio.shield(self._asr_future)
                self._asr_future = asyncio.get_running_loop().run_in_executor(
                    None, self.asr.transcribe_samples, event.audio_samples, event.sample_rate
                )
                result = await asyncio.shield(self._asr_future)
                self._asr_future = None
                if generation != self._mic_generation:
                    continue
                self.last_asr_text = result.text[:4096]
                await self.bus.emit(
                    "speech.transcribed",
                    self.name,
                    {
                        "text": self.last_asr_text,
                        "confidence": result.confidence,
                        "latency_ms": result.latency_ms,
                        "backend": result.backend_name,
                    },
                )
                if result.confidence >= 0.55 and self.last_asr_text.strip():
                    await self.bus.emit(
                        "input.user_message",
                        "microphone",
                        {"text": self.last_asr_text.strip(), "direct": False},
                        priority=EventPriority.HIGH,
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Speech recognition failed")
            finally:
                self._asr_queue.task_done()

    def start_mic_listener(self) -> None:
        if self._mic_task is not None and not self._mic_task.done():
            return
        self._mic_task = asyncio.create_task(self._mic_capture_loop(), name="audio:mic")

    async def _mic_capture_loop(self) -> None:
        if sd is None:
            logger.warning("sounddevice not available; microphone capture disabled")
            return
        self._mic_retry_s = MIC_RETRY_INITIAL_S
        while self._running and self.enable_mic:
            try:
                await self._capture_mic_stream()
                return  # Only returns once the service stopped or the mic was disabled.
            except asyncio.CancelledError:
                return
            except Exception:
                # Device unplugged, busy or stalled: reopen with backoff instead of going deaf.
                logger.warning(
                    "Microphone input stream stopped; retrying in %.1fs", self._mic_retry_s, exc_info=True
                )
            try:
                await asyncio.sleep(self._mic_retry_s)
            except asyncio.CancelledError:
                return
            self._mic_retry_s = min(MIC_RETRY_MAX_S, self._mic_retry_s * 2)

    async def _capture_mic_stream(self) -> None:
        sr = self.vad.sample_rate
        blocksize = int(sr * self.vad.frame_ms / 1000)
        q: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=16)
        loop = asyncio.get_running_loop()

        def enqueue(mono: np.ndarray) -> None:
            if q.full():
                q.get_nowait()
            q.put_nowait(mono)

        def _callback(indata: np.ndarray, _frames: int, _time_info: Any, _status: Any) -> None:
            try:
                loop.call_soon_threadsafe(enqueue, indata[:, 0].copy())
            except RuntimeError:
                pass  # The loop closed while PortAudio was still delivering a block.

        def _open() -> Any:
            stream = sd.InputStream(
                samplerate=sr, channels=1, dtype="float32", blocksize=blocksize, callback=_callback
            )
            try:
                stream.start()
            except BaseException:
                stream.close()
                raise
            return stream

        def _close_abandoned(opening: asyncio.Future) -> None:
            if opening.cancelled() or opening.exception() is not None:
                return
            # The device pool may already be shut down; never block the loop on PortAudio.
            threading.Thread(target=opening.result().close, name="audio-mic-close", daemon=True).start()

        # Opening and closing a device can block for hundreds of ms; keep it off the loop.
        opening = loop.run_in_executor(self._device_pool, _open)
        try:
            stream = await asyncio.shield(opening)
        except asyncio.CancelledError:
            # Stopped mid-open: the thread still finishes opening, so close what it opens.
            opening.add_done_callback(_close_abandoned)
            raise
        try:
            while self._running and self.enable_mic:
                try:
                    frame = await asyncio.wait_for(q.get(), MIC_STALL_TIMEOUT_S)
                except TimeoutError:
                    raise OSError("Microphone stream delivered no audio") from None
                self._mic_retry_s = MIC_RETRY_INITIAL_S  # A working stream earns a fast retry next time.
                await self.process_mic_frame(frame)
        finally:
            try:
                await asyncio.shield(loop.run_in_executor(self._device_pool, stream.close))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Closing microphone stream failed", exc_info=True)

    async def _run_playback_with_mouth_sync(
        self,
        speech: SynthesizedSpeech,
        *,
        lead_in_sound: str = "none",
        realtime: bool = True,
        correlation_id: str | None = None,
    ) -> None:
        try:
            self.is_speaking = True
            if lead_in_sound and lead_in_sound != "none":
                sfx_samples = await self.play_sfx(lead_in_sound)
                if realtime:
                    sfx_wait = min(0.40, len(sfx_samples) / max(1, self.sfx.sample_rate))
                    if sfx_wait > 0:
                        await asyncio.sleep(sfx_wait)

            if self._cancel_playback:
                return

            if self.enable_speaker and not self.muted and sd is not None and len(speech.samples) > 0:
                try:
                    await asyncio.get_running_loop().run_in_executor(
                        self._device_pool,
                        lambda: sd.play(speech.samples, samplerate=speech.sample_rate, blocking=False),
                    )
                except Exception:
                    logger.debug("sounddevice play failed for TTS", exc_info=True)

            await self.bus.emit(
                "audio.speech_started",
                source=self.name,
                payload={
                    "text": speech.text,
                    "duration_s": round(speech.duration_s, 2),
                    "backend": speech.backend_name,
                },
                priority=EventPriority.NORMAL,
                correlation_id=correlation_id,
            )

            # Drive FaceAnimator.mouth_open from the real RMS audio envelope at 60 Hz!
            if realtime:
                # Schedule against a deadline: sleep(dt) per frame drifts by the timer
                # granularity (~15.6 ms on Windows), stretching speech far past the audio.
                loop = asyncio.get_running_loop()
                dt = 1.0 / max(1, speech.envelope_fps)
                envelope = speech.mouth_envelope
                started = loop.time()
                while not self._cancel_playback:
                    index = int((loop.time() - started) / dt)
                    if index >= len(envelope):
                        break
                    if self.face_bridge is not None:
                        self.face_bridge.set_mouth_open(float(envelope[index]))
                    await asyncio.sleep(max(0.0, started + (index + 1) * dt - loop.time()))
            else:
                if self.face_bridge is not None and speech.mouth_envelope:
                    self.face_bridge.set_mouth_open(max(speech.mouth_envelope))

        except asyncio.CancelledError:
            pass
        finally:
            self.is_speaking = False
            if self.face_bridge is not None:
                self.face_bridge.set_mouth_open(None)
            if not self._cancel_playback:
                await self.bus.emit(
                    "audio.speech_completed",
                    source=self.name,
                    payload={"text": speech.text},
                    priority=EventPriority.LOW,
                    correlation_id=correlation_id,
                )

    async def _on_expression_updated(self, event: Event) -> None:
        # Play immediate `<15 ms` ack_chirp when robot starts listening or startle_pip on safety reflex
        p = event.payload
        if p.get("activity") == "listening" and p.get("sound") == "ack_chirp":
            await self.play_sfx("ack_chirp")
        elif str(p.get("reason", "")).startswith("reflex:") and p.get("sound"):
            await self.play_sfx(str(p["sound"]))

    def _queue_response(self, event: Event) -> None:
        # Goal progress is sequential speech. A later progress message must not cut
        # off the previous sentence. Cancellation still clears the whole queue.
        if not event.payload.get("say") and event.payload.get("sound", "none") == "none":
            return
        if self._response_queue.full():
            self._response_queue.get_nowait()
            self._response_queue.task_done()
        self._response_queue.put_nowait(event)
        if self._utterance_task and not self._utterance_task.done():
            return
        self._utterance_task = asyncio.create_task(self._drain_responses(), name="audio:utterance")
        self._utterance_task.add_done_callback(self._utterance_done)
        self.bus.track_task(self._utterance_task)

    async def _drain_responses(self) -> None:
        while not self._response_queue.empty():
            event = self._response_queue.get_nowait()
            try:
                await self._produce_response(event)
            except Exception:
                # One failed sentence must not strand the rest of the queue until the next reply.
                logger.exception("Speech production failed for %s", event.id)
            finally:
                self._response_queue.task_done()

    async def _produce_response(self, event: Event) -> None:
        say = str(event.payload.get("say", "")).strip()
        sound = str(event.payload.get("sound", "none")).strip()
        if say:
            await self.speak(say, lead_in_sound=sound, correlation_id=event.correlation_id)
            playback = self._speak_task
            if playback is not None:
                # speak() returns after synthesis when realtime playback starts.
                # Keep the producer on this utterance until the sound finishes.
                await asyncio.shield(playback)
        elif sound and sound != "none":
            await self.play_sfx(sound)

    def _utterance_done(self, task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception() is not None:
            logger.error("Speech production failed: %s", task.exception())
        if self._utterance_task is task:
            self._utterance_task = None

    async def _on_cancel(self, _event: Event) -> None:
        await self.interrupt_playback(reason="action_cancel")

    async def _on_barge_in(self, _event: Event) -> None:
        await self.interrupt_playback(reason="user_barge_in")
        await self.bus.emit(
            "action.cancel",
            source=self.name,
            payload={"reason": "user_barge_in"},
            priority=EventPriority.CRITICAL,
        )
