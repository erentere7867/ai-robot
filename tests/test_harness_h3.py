"""Automated verification suite for Milestone H3 (Voice, Pet SFX, Mouth Sync, VAD/ASR & Barge-In)."""

from __future__ import annotations

import asyncio
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from robot.actions.registry import CapabilityRegistry
from robot.audio.asr import SpeechRecognizer
from robot.audio.sfx import PetSoundSynthesizer
from robot.audio.tts import RobotTTSEngine, compute_mouth_envelope
from robot.audio.vad import VoiceActivityDetector
from robot.behavior.fsm import InteractionState
from robot.body.hal import BodyCapabilities
from robot.cognition.backend import NullBackend
from robot.cognition.validator import ResponseValidator
from robot.core.clock import ReplayClock
from robot.core.config import RobotConfig
from robot.main import RobotHarness


def _make_h3_harness(
    tmp_path: Path,
    *,
    enable_dashboard: bool = False,
    realtime_audio_sync: bool = False,
) -> tuple[RobotHarness, ReplayClock]:
    clock = ReplayClock()
    cfg = RobotConfig.from_yaml("config/robot.yaml")
    cfg.database_path = tmp_path / "test_memory_h3.sqlite3"
    cfg.recordings_dir = tmp_path / "recordings_h3"
    cfg.record_events = False
    harness = RobotHarness(
        cfg,
        clock=clock,
        backend=NullBackend(),
        require_wake_word=False,
        spawn_face_window=False,
        enable_dashboard=enable_dashboard,
        dashboard_port=0,
        enable_speaker=False,
        enable_mic=False,
        realtime_audio_sync=realtime_audio_sync,
    )
    return harness, clock


def test_pet_sound_synthesizer_all_sounds() -> None:
    """Verify procedural PetSoundSynthesizer generates all 10 non-verbal sounds in <15ms."""
    synth = PetSoundSynthesizer(sample_rate=24000)

    t0 = time.perf_counter()
    ack = synth.synthesize("ack_chirp")
    ack_ms = (time.perf_counter() - t0) * 1000.0
    assert ack_ms < 15.0, f"ack_chirp took {ack_ms:.2f}ms (must be <15ms for latency masking)"
    assert len(ack) > 1000
    assert float(np.max(np.abs(ack))) > 0.2

    for name in (
        "ack_chirp",
        "happy_trill",
        "curious_chirp",
        "sad_whimper",
        "angry_grumble",
        "sleepy_yawn",
        "startle_pip",
        "soft_beep",
        "giggle",
        "purr",
    ):
        samples = synth.synthesize(name)
        assert samples.dtype == np.float32
        assert len(samples) > 1500, f"Sound {name} generated empty/short waveform"
        assert float(np.max(np.abs(samples))) <= 1.0
        wav_bytes = synth.synthesize_wav_bytes(name)
        assert wav_bytes[:4] == b"RIFF"

    assert len(synth.synthesize("none")) == 0


def test_robot_tts_stylizer_and_60hz_mouth_envelope() -> None:
    """Verify TTS synthesis, character voice stylization, and 60 Hz RMS mouth openness envelope."""
    tts_warble = RobotTTSEngine(mode="warble")
    res_w = tts_warble.synthesize("Yay! Watch my happy dance moves!")
    assert res_w.backend_name == "melodic_warble"
    assert res_w.duration_s > 0.4
    assert len(res_w.mouth_envelope) >= int(res_w.duration_s * 50)
    assert max(res_w.mouth_envelope) > 0.50
    assert (max(res_w.mouth_envelope) - min(res_w.mouth_envelope)) > 0.40

    # Also test Windows offline SAPI stylized voice
    tts_sapi = RobotTTSEngine(mode="sapi")
    res_s = tts_sapi.synthesize("Hello there, I am Sprocket!")
    assert res_s.duration_s > 0.4
    assert len(res_s.samples) > 8000
    assert max(res_s.mouth_envelope) > 0.30

    # Also test LFM3-Omni-2B native TTS (`llama-tts.exe` with `-c 512`)
    tts_omni = RobotTTSEngine(mode="lfm3_omni")
    res_o = tts_omni.synthesize("Hi there, watch my dance moves!")
    assert res_o.backend_name == "lfm3_omni_tts"
    assert res_o.sample_rate == 24000
    assert res_o.duration_s > 0.8
    assert max(res_o.mouth_envelope) > 0.35

    # Verify compute_mouth_envelope directly on a pulsed waveform
    sr = 24000
    t = np.linspace(0.0, 0.5, sr // 2, endpoint=False, dtype=np.float32)
    carrier = np.sin(2.0 * np.pi * 240.0 * t)
    mod = (np.sin(2.0 * np.pi * 5.0 * t) > 0).astype(np.float32)
    env = compute_mouth_envelope(carrier * mod * 0.5, sr, fps=60)
    assert len(env) >= 29
    assert max(env) > 0.5
    assert min(env) < 0.15


def test_vad_utterance_segmentation_and_barge_in() -> None:
    """Verify VAD pre-roll buffering, speech start/end segmentation, and echo-aware barge-in."""
    vad = VoiceActivityDetector(
        sample_rate=16000,
        frame_ms=30,
        start_rms_threshold=0.02,
        barge_in_rms_threshold=0.06,
        min_speech_frames=3,
        silence_hangover_frames=5,
        pre_roll_frames=6,
    )
    frame_len = vad.frame_samples
    silence = np.zeros(frame_len, dtype=np.float32)
    speech_frame = (0.12 * np.sin(np.linspace(0, 20, frame_len))).astype(np.float32)

    events = []
    for _ in range(4):
        events.extend(vad.process_frame(silence))
    for _ in range(6):
        events.extend(vad.process_frame(speech_frame))
    for _ in range(7):
        events.extend(vad.process_frame(silence))

    kinds = [e.kind for e in events]
    assert "speech_start" in kinds
    assert "speech_end" in kinds
    end_ev = next(e for e in events if e.kind == "speech_end")
    assert end_ev.audio_samples is not None
    # Pre-roll + 6 speech frames + 5 hangover frames = at least 14 frames
    assert len(end_ev.audio_samples) >= 14 * frame_len

    # Now test barge-in while robot_is_speaking=True
    barge_events = []
    for _ in range(4):
        barge_events.extend(vad.process_frame(speech_frame, robot_is_speaking=True))
    assert any(e.kind == "barge_in" for e in barge_events)


@pytest.mark.asyncio
async def test_audio_service_mouth_sync_and_barge_in_interruption(tmp_path: Path) -> None:
    """Verify full harness turn plays SFX + TTS, drives mouth_open, and barge-in halts speech & actions."""
    harness, _clock = _make_h3_harness(
        tmp_path,
        enable_dashboard=True,
        realtime_audio_sync=True,
    )
    interrupted_events: list[dict] = []
    canceled_events: list[dict] = []
    speech_started = asyncio.Event()
    harness.bus.subscribe("audio.interrupted", lambda e: interrupted_events.append(e.payload))
    harness.bus.subscribe("action.cancel", lambda e: canceled_events.append(e.payload))
    harness.bus.subscribe("audio.speech_started", lambda e: speech_started.set(), immediate=True)

    await harness.start()
    try:
        # 1. Trigger a turn that speaks and dances
        await harness.bus.emit(
            "input.user_message",
            source="test",
            payload={"text": "Sprocket, do a dance for me!", "direct": True},
        )
        # drain now includes the entire queued playback; barge in while it is live.
        await asyncio.wait_for(speech_started.wait(), timeout=2)

        # Wait briefly so background _speak_task enters 60 Hz mouth sync loop
        for _ in range(30):
            if harness.audio.is_speaking and harness.face_bridge.animator.external_mouth_open is not None:
                break
            await asyncio.sleep(0.02)

        assert harness.audio.sfx_played_count >= 1
        assert harness.audio.tts_spoken_count == 1
        assert harness.audio.is_speaking is True
        assert len(harness.audio.last_mouth_envelope) > 10

        # 2. Simulate loud user speech into microphone while robot is speaking (barge-in!)
        frame_len = harness.audio.vad.frame_samples
        loud_frame = (0.18 * np.sin(np.linspace(0, 30, frame_len))).astype(np.float32)
        for _ in range(4):
            await harness.audio.process_mic_frame(loud_frame)
        await harness.bus.drain()

        assert harness.audio.barge_in_count >= 1
        assert harness.audio.is_speaking is False
        assert len(interrupted_events) >= 1
        assert len(canceled_events) >= 1
        assert harness.fsm.state == InteractionState.ATTENDING
        assert harness.face_bridge.current_frame.activity == "listening"
        assert harness.face_bridge.animator.external_mouth_open is None

        # 3. Verify dashboard audio endpoints work
        assert harness.dashboard is not None
        base_url = harness.dashboard.url
        loop = asyncio.get_running_loop()
        state = await loop.run_in_executor(
            None,
            lambda: json.loads(
                urllib.request.urlopen(f"{base_url}/api/state", timeout=5.0).read().decode("utf-8")
            ),
        )
        assert "audio" in state
        assert state["audio"]["barge_in_count"] >= 1
    finally:
        await harness.stop()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows offline speech integration")
def test_offline_windows_asr_transcribes_speech() -> None:
    """Verify offline SpeechRecognizer transcribes a spoken WAV phrase in <2.5s."""
    tts = RobotTTSEngine(mode="sapi")
    raw_sapi = tts._try_windows_sapi("hello robot")
    assert raw_sapi is not None
    samples, sr = raw_sapi

    asr = SpeechRecognizer(mode="windows")
    res = asr.transcribe_samples(samples, sr)
    assert res.backend_name == "windows_offline_asr"
    assert res.latency_ms < 2500.0
    assert len(res.text.strip()) > 0


def test_self_name_vocative_stripping() -> None:
    """Ensure the robot never addresses the human user by its own name ('Sprocket')."""
    validator = ResponseValidator(CapabilityRegistry(BodyCapabilities()))
    v1 = validator.validate({"say": "Hey there, Sprocket! I'm ready to help you out."})
    assert v1.say == "Hey there! I'm ready to help you out."
    v2 = validator.validate({"say": "I'm doing well, Sprocket! How are you?"})
    assert v2.say == "I'm doing well! How are you?"
    v3 = validator.validate({"say": "Hi! I'm Sprocket, your companion robot."})
    assert v3.say == "Hi! I'm Sprocket, your companion robot."
