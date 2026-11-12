"""Speech plumbing (robot/interfaces/voice.py) without the models: WAV in and out, and resampling for Whisper."""

import numpy as np

from robot.interfaces.voice import read_wav, to_16k, write_wav


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
    from robot.interfaces.voice import Listener

    rng = np.random.default_rng(0)
    room = lambda s: 0.002 * rng.standard_normal(int(s * 16000))  # noqa: E731 - quiet room noise
    voice = lambda s: 0.2 * np.sin(2 * np.pi * 220 * np.arange(int(s * 16000)) / 16000)  # noqa: E731
    stream = np.concatenate([room(1.0), voice(0.8), room(0.2), voice(0.5), room(1.0), voice(0.05), room(1.0),
                             voice(1.2), room(1.0)])
    ear, heard = Listener(), []
    for k in range(0, stream.size, 1600):  # 100 ms at a time, as a microphone delivers
        heard += ear.feed(stream[k:k + 1600])
    assert len(heard) == 2  # a short gap inside speech does not split it; a click is not speech
    assert 1.3 * 16000 < heard[0].size < 1.7 * 16000 and 1.1 * 16000 < heard[1].size < 1.4 * 16000


def test_only_what_is_meant_for_the_robot_is_acted_on():
    from robot.interfaces.voice import addressed

    assert addressed("Sprocket, go to the chair.") == "go to the chair."
    assert addressed("hey Sprocket circle the table") == "circle the table"
    assert addressed("Stop!") == "Stop!"  # anyone may stop it
    assert addressed("I left my keys in the kitchen") is None  # talk in the room
    assert addressed("The red one.", waiting=True) == "The red one."  # an answer to its question
