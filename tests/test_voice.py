"""Tests for voice plumbing, voiceprint extraction, onset DOA locking, and procedural chimes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot.interfaces.voice import (
    DOARingBuffer,
    Listener,
    addressed,
    chime,
    extract_voiceprint,
    read_wav,
    to_16k,
    write_wav,
)


def test_a_wav_round_trips_and_is_resampled_to_what_whisper_hears():
    t = np.arange(48000) / 48000
    tone = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)  # one second at 48 kHz, as a browser records
    samples, rate = read_wav(write_wav(tone, 48000))
    assert rate == 48000 and np.abs(samples - tone).max() < 1e-3
    at16 = to_16k(samples, rate)
    assert at16.size == 16000
    peak = np.argmax(np.abs(np.fft.rfft(at16)))  # still a 440 Hz tone
    assert abs(peak - 440) <= 1


def test_the_microphone_stream_is_cut_into_utterances_at_pauses():
    rng = np.random.default_rng(0)
    room = lambda s: 0.002 * rng.standard_normal(int(s * 16000))  # noqa: E731 - quiet room noise
    voice = lambda s: 0.2 * np.sin(2 * np.pi * 220 * np.arange(int(s * 16000)) / 16000)  # noqa: E731
    stream = np.concatenate([
        room(1.0), voice(0.8), room(0.2), voice(0.5), room(1.0), voice(0.05), room(1.0), voice(1.2), room(1.0)
    ])
    ear, heard = Listener(), []
    for k in range(0, stream.size, 1600):  # 100 ms at a time, as a microphone delivers
        heard += ear.feed(stream[k: k + 1600])
    assert len(heard) == 2  # a short gap inside speech does not split it; a click is not speech
    assert 1.3 * 16000 < heard[0].size < 1.7 * 16000 and 1.1 * 16000 < heard[1].size < 1.4 * 16000


def test_only_what_is_meant_for_the_robot_is_acted_on():
    assert addressed("Sprocket, go to the chair.") == "go to the chair."
    assert addressed("hey Sprocket circle the table") == "circle the table"
    assert addressed("Stop!") == "Stop!"  # anyone may stop it
    assert addressed("I left my keys in the kitchen") is None  # talk in the room
    assert addressed("The red one.", waiting=True) == "The red one."  # an answer to its question


def test_extract_voiceprint_consistency_and_separation():
    # Empty or near-silent audio returns a zero vector
    assert np.all(extract_voiceprint(np.zeros(0, dtype=np.float32)) == 0.0)
    assert np.all(extract_voiceprint(np.ones(100, dtype=np.float32) * 1e-6) == 0.0)

    # Synthetic Speaker 1: pitch 150 Hz with formant harmonics
    t1 = np.arange(16000) / 16000.0
    t2 = np.arange(24000) / 16000.0
    s1_clip1 = 0.5 * np.sin(2 * np.pi * 150 * t1) + 0.3 * np.sin(2 * np.pi * 300 * t1) + 0.2 * np.sin(2 * np.pi * 450 * t1)
    s1_clip2 = 0.45 * np.sin(2 * np.pi * 150 * t2) + 0.35 * np.sin(2 * np.pi * 300 * t2) + 0.2 * np.sin(2 * np.pi * 450 * t2)

    # Synthetic Speaker 2: pitch 280 Hz with different harmonic distribution
    s2 = 0.5 * np.sin(2 * np.pi * 280 * t1) + 0.3 * np.sin(2 * np.pi * 560 * t1)

    v1_a = extract_voiceprint(s1_clip1, rate=16000)
    v1_b = extract_voiceprint(s1_clip2, rate=16000)
    v2 = extract_voiceprint(s2, rate=16000)

    assert v1_a.shape == (64,)
    assert np.linalg.norm(v1_a) == pytest.approx(1.0, abs=1e-4)

    # Clips from the same speaker should match tightly
    sim_same = float(np.dot(v1_a, v1_b))
    assert sim_same > 0.90, f"Expected high similarity for same speaker, got {sim_same}"

    # Different speaker should separate cleanly
    sim_diff = float(np.dot(v1_a, v2))
    assert sim_diff < 0.60, f"Expected low similarity for different speakers, got {sim_diff}"


def test_doa_ring_buffer_onset_locking():
    ring = DOARingBuffer(window_s=2.5)

    # Push historical chunks: speech started from +45 deg (0.785 rad) at t=10.0
    t0 = 10.0
    ring.push(np.zeros(1600, dtype=np.float32), doa_rad=0.785, now=t0)
    ring.push(np.zeros(1600, dtype=np.float32), doa_rad=0.785, now=t0 + 0.1)

    # Later noise arrived from -90 deg (-1.57 rad) at t=10.9
    ring.push(np.zeros(1600, dtype=np.float32), doa_rad=-1.57, now=t0 + 0.9)

    # Speech ended at t=11.0 with duration 1.0 s -> onset is t=10.0 + 0.15 = 10.15 s
    onset_angle = ring.doa_at(timestamp=t0 + 0.15)
    assert onset_angle is not None
    assert math.isclose(onset_angle, 0.785, abs_tol=0.01)

    # Integration test with Listener and utterance_with_doa
    listener = Listener(rate=16000, pause=0.3, shortest=0.2)
    ring2 = DOARingBuffer(window_s=2.5)

    voice_tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(8000) / 16000.0)  # 0.5 s voice
    silence = np.zeros(6400, dtype=np.float32)  # 0.4 s silence to finish utterance

    now_clock = 50.0
    # Push voice from direction 1.2 rad
    res1 = ring2.utterance_with_doa(listener, voice_tone, doa_rad=1.2, now=now_clock)
    assert len(res1) == 0  # not finished yet

    # Push pause from direction -0.5 rad (room echo or chatter)
    now_clock += 0.5
    res2 = ring2.utterance_with_doa(listener, silence, doa_rad=-0.5, now=now_clock)
    assert len(res2) == 1
    utt, speaker_doa, vprint = res2[0]
    # Locked onto speaker's original direction at onset
    assert math.isclose(speaker_doa, 1.2, abs_tol=0.1)
    assert vprint.shape == (64,)


def test_chime_generation_for_all_six_kinds():
    kinds = ["wake", "listen", "curious", "happy", "uh_oh", "yield"]
    for kind in kinds:
        wav_bytes = chime(kind, rate=16000)
        assert len(wav_bytes) > 1000
        samples, rate = read_wav(wav_bytes)
        assert rate == 16000
        assert samples.ndim == 1
        assert len(samples) > 2000
        # Peak amplitude is pleasant and not clipping
        peak = float(np.max(np.abs(samples)))
        assert 0.10 <= peak <= 0.95
        # Envelopes start and end softly (no harsh click at boundaries)
        assert abs(samples[0]) < 0.05
        assert abs(samples[-1]) < 0.05

    with pytest.raises(ValueError, match="Unknown chime kind"):
        chime("unsupported_chirp")
