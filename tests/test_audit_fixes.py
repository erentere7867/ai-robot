"""Regression tests for peripheral-subsystem bugs found in the bus, recorder, audio, face, console and config."""

from __future__ import annotations

import asyncio
import builtins
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import robot.audio.service as audio_service
import robot.audio.tts as tts_module
import robot.face.process as face_process
from robot.audio.asr import SpeechRecognizer
from robot.audio.service import AudioService
from robot.audio.tts import RobotTTSEngine, SynthesizedSpeech
from robot.core.bus import EventBus
from robot.core.config import AgentConfig, RobotConfig
from robot.core.events import Event
from robot.core.recorder import EventRecorder
from robot.core.service import ServiceSupervisor
from robot.face.process import FaceProcessBridge, run_standalone_face_loop
from robot.interfaces.console import ConsoleInterface

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _speech(text: str, seconds: float) -> SynthesizedSpeech:
    frames = max(1, round(seconds * 60))
    return SynthesizedSpeech(text, np.zeros(int(24000 * seconds), dtype=np.float32), 24000, [0.5] * frames)


async def _wait_for(predicate, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.01)


# --- event bus -----------------------------------------------------------------------------------


async def test_subscriber_survives_handler_awaiting_a_cancelled_future():
    bus = EventBus()
    seen = []

    async def handler(event):
        seen.append(event.payload["n"])
        if event.payload["n"] == 0:
            future = asyncio.get_running_loop().create_future()
            future.cancel()
            await future

    bus.subscribe("work", handler, name="worker")
    await bus.start()
    try:
        await bus.emit("work", "test", {"n": 0})
        await bus.emit("work", "test", {"n": 1})
        await bus.drain()
        assert seen == [0, 1]
        assert bus.stats()["subscribers"]["worker"]["errors"] == 1
    finally:
        await bus.stop()


async def test_unsubscribe_still_cancels_a_worker_blocked_in_its_handler():
    bus = EventBus()
    entered, never = asyncio.Event(), asyncio.Event()

    async def handler(_event):
        entered.set()
        await never.wait()

    sub_id = bus.subscribe("work", handler)
    await bus.start()
    task = bus._subscriptions[sub_id].task
    await bus.emit("work", "test")
    await asyncio.wait_for(entered.wait(), 1)
    bus.unsubscribe(sub_id)
    await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 1)
    assert task.done()
    await bus.stop()


# --- recorder ------------------------------------------------------------------------------------


def test_recording_torn_by_a_crash_still_replays(tmp_path):
    path = tmp_path / "session.jsonl"
    good = [Event("one", "test", {"v": 1}, ts=1.0), Event("two", "test", {"v": 2}, ts=2.0)]
    path.write_text("".join(e.to_json() + "\n" for e in good) + good[0].to_json()[:17], encoding="utf-8")
    assert [e.type for e in EventRecorder.load_events(path)] == ["one", "two"]

    path.write_text(good[0].to_json() + "\n{broken\n" + good[1].to_json() + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        EventRecorder.load_events(path)


def test_recorder_skips_unserializable_and_unreplayable_events(tmp_path):
    recorder = EventRecorder(tmp_path)
    recorder.open()
    recorder.record(Event("bad", "test", {"value": object()}))
    recorder.record(Event("huge", "test", {"blob": "x" * 70_000}))
    recorder.record(Event("ok", "test", {"v": 1}))
    recorder.close()
    assert not recorder.failed
    assert recorder.dropped_count == 2
    assert [e.type for e in EventRecorder.load_events(recorder.file_path)] == ["ok"]


def test_recorder_flushes_while_events_never_pause(tmp_path):
    recorder = EventRecorder(tmp_path)
    recorder.open()
    try:
        deadline = time.monotonic() + 1.2
        while time.monotonic() < deadline:
            recorder.record(Event("telemetry.tick", "test", {"t": time.monotonic()}))
            time.sleep(0.05)
        # Without a periodic flush a steady stream keeps everything in the buffer until close/crash.
        assert recorder.file_path.stat().st_size > 0
    finally:
        recorder.close()


# --- audio ---------------------------------------------------------------------------------------


async def test_one_failed_sentence_does_not_strand_the_queued_ones():
    bus = EventBus()
    audio = AudioService(bus, enable_speaker=False, tts_mode="warble", realtime_sync=False)

    def synthesize(text):
        if text == "bad":
            raise RuntimeError("synthesizer crashed")
        return _speech(text, 0.05)

    audio.tts.synthesize = synthesize
    started = []
    bus.subscribe("audio.speech_started", lambda e: started.append(e.payload["text"]), immediate=True)
    await bus.start()
    await audio.start()
    try:
        await bus.emit("robot.response", "test", {"say": "bad"})
        await bus.emit("robot.response", "test", {"say": "good"})
        await _wait_for(lambda: started == ["good"])
        assert audio._response_queue.empty()
    finally:
        await audio.stop()
        await bus.stop()


async def test_realtime_mouth_sync_lasts_as_long_as_the_audio():
    bus = EventBus()
    audio = AudioService(bus, enable_speaker=False, tts_mode="warble", realtime_sync=True)
    audio.tts.synthesize = lambda text: _speech(text, 0.6)
    stamps = {}
    loop = asyncio.get_running_loop()
    for kind in ("audio.speech_started", "audio.speech_completed"):
        bus.subscribe(kind, lambda e: stamps.setdefault(e.type, loop.time()), immediate=True)
    await bus.start()
    await audio.start()
    try:
        await bus.emit("robot.response", "test", {"say": "Hello there"})
        await _wait_for(lambda: "audio.speech_completed" in stamps)
        elapsed = stamps["audio.speech_completed"] - stamps["audio.speech_started"]
        # Per-frame sleep(1/60) took ~1.85x the audio length on Windows' 15.6 ms timer.
        assert 0.55 <= elapsed < 0.75
    finally:
        await audio.stop()
        await bus.stop()


async def test_missing_sound_backend_does_not_restart_audio_and_cut_speech(monkeypatch):
    monkeypatch.setattr(audio_service, "sd", None)
    bus = EventBus()
    audio = AudioService(bus, enable_speaker=False, enable_mic=True, tts_mode="warble", realtime_sync=True)
    audio.tts.synthesize = lambda text: _speech(text, 2.0)
    interrupted = []
    bus.subscribe("audio.interrupted", lambda e: interrupted.append(e.payload), immediate=True)
    supervisor = ServiceSupervisor(bus, health_interval_s=0)
    supervisor.register(audio)
    await bus.start()
    await supervisor.start_all()
    try:
        await bus.emit("robot.response", "test", {"say": "A long sentence"})
        await _wait_for(lambda: audio.is_speaking)
        await _wait_for(lambda: audio._mic_task is not None and audio._mic_task.done())
        report = await supervisor.check_health()
        assert report["audio_service"].healthy
        assert report["audio_service"].status == "mic_unavailable"
        assert supervisor.restart_counts["audio_service"] == 0
        assert interrupted == [] and audio.is_speaking
    finally:
        await supervisor.stop_all()
        await bus.stop()


class _FakeStream:
    instances: list[_FakeStream] = []

    def __init__(self, *, samplerate, channels, dtype, blocksize, callback):
        self.callback = callback
        self.blocksize = blocksize
        self.opened_on = threading.get_ident()
        self.closed_on: int | None = None
        self.closed = threading.Event()
        _FakeStream.instances.append(self)

    def start(self):
        if len(_FakeStream.instances) == 1:
            return  # The first device "stalls": it never delivers a block.

        def feed():
            while not self.closed.is_set():
                self.callback(np.zeros((self.blocksize, 1), dtype=np.float32), self.blocksize, None, None)
                time.sleep(0.01)

        threading.Thread(target=feed, daemon=True).start()

    def close(self):
        self.closed_on = threading.get_ident()
        self.closed.set()


async def test_microphone_opens_off_the_loop_and_recovers_from_a_stalled_device(monkeypatch):
    _FakeStream.instances = []
    monkeypatch.setattr(audio_service, "sd", SimpleNamespace(InputStream=_FakeStream))
    monkeypatch.setattr(audio_service, "MIC_STALL_TIMEOUT_S", 0.2)
    monkeypatch.setattr(audio_service, "MIC_RETRY_INITIAL_S", 0.05)
    bus = EventBus()
    audio = AudioService(bus, enable_speaker=False, enable_mic=True, tts_mode="warble")
    frames = []

    async def process(frame):
        frames.append(frame)

    audio.process_mic_frame = process
    loop_thread = threading.get_ident()
    await bus.start()
    await audio.start()
    try:
        await _wait_for(lambda: len(frames) >= 3)
        first, second = _FakeStream.instances[:2]
        assert first.closed.is_set() and first.closed_on != loop_thread
        assert first.opened_on != loop_thread and second.opened_on != loop_thread
        assert (await audio.health()).healthy
    finally:
        await audio.stop()
        await bus.stop()
    assert second.closed.is_set() and second.closed_on != loop_thread


async def test_stopping_while_the_microphone_opens_still_closes_the_device(monkeypatch):
    opening, streams = threading.Event(), []

    class SlowOpeningStream:
        def __init__(self, **_kwargs):
            self.closed = threading.Event()
            streams.append(self)

        def start(self):
            opening.set()
            time.sleep(0.3)  # Opening a real device can take this long on Windows.

        def close(self):
            self.closed.set()

    monkeypatch.setattr(audio_service, "sd", SimpleNamespace(InputStream=SlowOpeningStream))
    bus = EventBus()
    audio = AudioService(bus, enable_speaker=False, enable_mic=True, tts_mode="warble")
    await bus.start()
    await audio.start()
    assert await asyncio.to_thread(opening.wait, 2)
    await audio.stop()
    await bus.stop()
    # Previously the stream finished opening after the task was cancelled and stayed open forever.
    assert await asyncio.to_thread(streams[0].closed.wait, 2)


async def test_microphone_backoff_resets_after_the_stream_works_again(monkeypatch):
    streams = []

    class FlakyStream:
        def __init__(self, *, callback, blocksize, **_kwargs):
            self.callback, self.blocksize = callback, blocksize
            self.closed = threading.Event()
            streams.append(self)

        def start(self):
            if len(streams) <= 3:
                raise OSError("device busy")
            for _ in range(3):
                self.callback(np.zeros((self.blocksize, 1), dtype=np.float32), self.blocksize, None, None)

        def close(self):
            self.closed.set()

    monkeypatch.setattr(audio_service, "sd", SimpleNamespace(InputStream=FlakyStream))
    monkeypatch.setattr(audio_service, "MIC_RETRY_INITIAL_S", 0.01)
    monkeypatch.setattr(audio_service, "MIC_STALL_TIMEOUT_S", 0.2)
    bus = EventBus()
    audio = AudioService(bus, enable_speaker=False, enable_mic=True, tts_mode="warble")
    frames = []

    async def process(frame):
        frames.append(frame)

    audio.process_mic_frame = process
    await bus.start()
    await audio.start()
    try:
        await _wait_for(lambda: len(frames) >= 3)
        assert audio._mic_retry_s == 0.01  # Three failures had doubled it; a working stream resets it.
    finally:
        await audio.stop()
        await bus.stop()


def test_whisper_non_speech_tags_never_become_user_text(tmp_path, monkeypatch):
    whisper_bin, model, wav = tmp_path / "whisper-cli", tmp_path / "model.bin", tmp_path / "u.wav"
    for path in (whisper_bin, model, wav):
        path.write_bytes(b"x")
    outputs = iter([" [BLANK_AUDIO]\n", " (wind blowing)\n", " Hello there. [MUSIC]\n How are you?\n"])
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(kwargs)
        return subprocess.CompletedProcess(cmd, 0, stdout=next(outputs), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    asr = SpeechRecognizer(mode="whisper_cpp", whisper_bin=whisper_bin, whisper_model=model)
    silent, noise, speech = (asr.transcribe_wav(wav) for _ in range(3))
    assert (silent.text, silent.confidence) == ("", 0.0)
    assert (noise.text, noise.confidence) == ("", 0.0)
    assert speech.text == "Hello there. How are you?" and speech.confidence > 0.55
    assert all(call["creationflags"] == NO_WINDOW for call in calls)
    # whisper.cpp writes UTF-8; decoding with the Windows code page turned "♪" into "â™ª" text.
    assert all(call["encoding"] == "utf-8" and call["errors"] == "replace" for call in calls)


def test_speech_subprocesses_never_flash_a_console_window(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(kwargs)
        return subprocess.CompletedProcess(cmd, 0, stdout="0.91|hello", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert SpeechRecognizer._try_windows_offline_asr(tmp_path / "u.wav").text == "hello"
    binary = tmp_path / "llama-tts.exe"
    binary.write_bytes(b"x")
    RobotTTSEngine(tts_bin=binary, model_path=binary, mmproj_path=binary)._try_lfm3_omni_tts("hi")
    assert len(calls) == 2 and all(call["creationflags"] == NO_WINDOW for call in calls)


@pytest.mark.skipif(sys.platform != "win32" or shutil.which("powershell") is None, reason="Windows SAPI")
def test_sapi_worker_reads_utf8_text_and_non_ascii_paths(tmp_path, monkeypatch):
    tts_module._cleanup_sapi_worker()
    try:
        if RobotTTSEngine._try_windows_sapi("hello") is None:
            pytest.skip("System.Speech synthesis is unavailable on this machine")
        tts_module._cleanup_sapi_worker()
        # Typical for Turkish/European Windows accounts: %TEMP% under C:\Users\Çağrı\...
        target = tmp_path / "\u00c7a\u011fr\u0131 ses"
        target.mkdir()
        original = tempfile.TemporaryDirectory
        monkeypatch.setattr(
            tts_module.tempfile, "TemporaryDirectory", lambda prefix=None: original(prefix=prefix, dir=target)
        )
        result = RobotTTSEngine._try_windows_sapi("I\u2019m caf\u00e9 ready")
        assert result is not None and len(result[0]) > 0
    finally:
        tts_module._cleanup_sapi_worker()


# --- face ----------------------------------------------------------------------------------------


class _SlowFaceProcess:
    def __init__(self):
        self.exited = False
        self.wait_threads: list[int] = []

    def poll(self):
        return 0 if self.exited else None

    def wait(self, timeout=None):
        self.wait_threads.append(threading.get_ident())
        time.sleep(0.2)
        self.exited = True
        return 0

    def terminate(self):
        pass

    def kill(self):
        self.exited = True


async def test_face_bridge_stop_never_blocks_the_event_loop():
    bus = EventBus()
    bridge = FaceProcessBridge(bus, udp_port=_free_udp_port())
    await bus.start()
    await bridge.start()
    proc = _SlowFaceProcess()
    bridge._proc = proc
    await bridge.stop()
    await bus.stop()
    assert proc.exited and bridge._proc is None
    assert proc.wait_threads and threading.get_ident() not in proc.wait_threads


def test_face_window_process_imports_the_same_checkout_as_the_brain(tmp_path, monkeypatch):
    spawned = {}

    def fake_popen(cmd, **kwargs):
        spawned.update(cmd=cmd, **kwargs)
        return SimpleNamespace(poll=lambda: 0)

    monkeypatch.setattr(face_process, "subprocess", SimpleNamespace(Popen=fake_popen))
    monkeypatch.chdir(tmp_path)  # e.g. a launcher script started from another directory
    FaceProcessBridge(EventBus(), udp_port=_free_udp_port())._spawn_subprocess()
    assert spawned["cmd"][1:3] == ["-m", "robot.face.process"]
    # Resolve the module exactly as the child would: from its cwd, without the parent's sys.path tweaks.
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    probe = subprocess.run(
        [sys.executable, "-c", "import robot.face.process as m; print(m.__file__)"],
        cwd=spawned.get("cwd"), env=env, capture_output=True, text=True, timeout=60,
    )
    assert probe.returncode == 0, probe.stderr
    assert Path(probe.stdout.strip()).resolve() == Path(face_process.__file__).resolve()


def test_face_window_survives_an_oversized_datagram():
    port = _free_udp_port()
    outcome: dict[str, object] = {}

    def run():
        try:
            outcome["frames"] = run_standalone_face_loop(port=port, fps=200, max_frames=5000, headless=True)
        except BaseException as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        deadline = time.monotonic() + 10
        while worker.is_alive() and time.monotonic() < deadline:
            sender.sendto(b"x" * 5000, ("127.0.0.1", port))
            sender.sendto(json.dumps({"theme": "green"}).encode(), ("127.0.0.1", port))
            sender.sendto(json.dumps({"cmd": "quit"}).encode(), ("127.0.0.1", port))
            worker.join(0.05)
    worker.join(5)
    assert "error" not in outcome, outcome.get("error")
    assert 0 < outcome["frames"] < 5000


# --- console -------------------------------------------------------------------------------------


async def test_console_survives_bad_arguments_and_reads_on_a_daemon_thread(monkeypatch, capsys):
    bus = EventBus()
    harness = SimpleNamespace(bus=bus, config=SimpleNamespace(name="Test"), dashboard=None)
    console = ConsoleInterface(harness, verbose_events=False)
    lines = ["/emotion happy high", "/observe box far", "/quit"]
    daemon_flags = []

    def fake_input(_prompt=""):
        daemon_flags.append(threading.current_thread().daemon)
        return lines.pop(0)

    monkeypatch.setattr(builtins, "input", fake_input)
    await asyncio.wait_for(console.run_interactive(), 3)
    assert lines == []
    # A non-daemon executor thread stuck in input() keeps the process alive after Ctrl+C.
    assert daemon_flags == [True, True, True]
    assert capsys.readouterr().out.count("[console] Command failed") == 2


# --- configuration -------------------------------------------------------------------------------


def test_config_tolerates_empty_sections_and_rejects_ambiguous_values(tmp_path):
    path = tmp_path / "robot.yaml"
    path.write_text(
        'robot:\n  wake_words: "sprocket"\nruntime:\nagent:\nautonomy:\naudio:\n'
        'bus:\n  record_events: "false"\nllm:\n  backend: "null"\n',
        encoding="utf-8",
    )
    cfg = RobotConfig.from_yaml(path, project_root=PROJECT_ROOT)
    assert cfg.wake_words == ["sprocket"]
    assert cfg.record_events is False
    assert cfg.agent.max_steps == AgentConfig().max_steps and cfg.autonomy.enabled is True

    path.write_text('llm:\n  auto_start_server: "sometimes"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="auto_start_server"):
        RobotConfig.from_yaml(path, project_root=PROJECT_ROOT)
    path.write_text("agent: 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="agent"):
        RobotConfig.from_yaml(path, project_root=PROJECT_ROOT)


# --- scripts -------------------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32" or shutil.which("powershell") is None, reason="PowerShell script")
def test_start_llm_always_enables_jinja_and_the_vision_projector(tmp_path):
    server = tmp_path / "fake-server.cmd"
    server.write_text("@echo ARGS: %*\r\n", encoding="ascii")
    model, mmproj = tmp_path / "model.gguf", tmp_path / "mmproj.gguf"
    model.write_bytes(b"x")
    mmproj.write_bytes(b"x")
    script = PROJECT_ROOT / "scripts" / "start_llm.ps1"
    # Point the default paths at an empty profile: a script that ignored -ServerBin must fail,
    # never launch the real llama-server.
    env = {**os.environ, "USERPROFILE": str(tmp_path)}

    def run(*extra: str) -> str:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
             "-ServerBin", str(server), "-ModelPath", str(model), *extra],
            capture_output=True, text=True, timeout=60, env=env,
        )
        assert proc.returncode == 0, proc.stderr
        return next(line for line in proc.stdout.splitlines() if line.startswith("ARGS:"))

    args = run("-MmprojPath", str(mmproj))
    assert "--jinja" in args and f"--mmproj {mmproj}" in args
    assert f"--mmproj {mmproj}" in run("-MmprojPath", str(mmproj), "-IncludeProjector", "-Port", "8099")
    assert "--mmproj" not in run("-MmprojPath", str(tmp_path / "missing.gguf"))
    # Existing callers pass Port, GpuLayers and CtxSize positionally.
    positional = run("8099", "50", "2048", "-MmprojPath", str(mmproj))
    assert all(part in positional for part in ("--port 8099", "-ngl 50", "-c 2048", "--jinja", "--mmproj"))


@pytest.mark.skipif(sys.platform != "win32" or shutil.which("powershell") is None, reason="PowerShell script")
def test_run_ps1_starts_the_llm_from_a_path_with_spaces_and_stops_its_whole_tree(tmp_path):
    psutil = pytest.importorskip("psutil")
    project = tmp_path / "ai robot"  # The real checkout lives under "...\ai robot".
    (project / "scripts").mkdir(parents=True)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    original = (PROJECT_ROOT / "run.ps1").read_text(encoding="utf-8")
    default_url = "http://127.0.0.1:8080/health"
    assert original.count(default_url) == 1
    # Port 8080 belongs to the user's own llama-server; probe a private port instead.
    (project / "run.ps1").write_text(
        original.replace(default_url, f"http://127.0.0.1:{port}/health"), encoding="utf-8"
    )
    served, fake_bin = tmp_path / "served", tmp_path / "bin"
    served.mkdir()
    fake_bin.mkdir()
    (served / "health").write_text("ok", encoding="ascii")
    marker, pid_file, args_file = tmp_path / "started.txt", tmp_path / "server.pid", tmp_path / "args.txt"
    # Stand-in for start_llm.ps1: a wrapper shell whose child answers /health, like llama-server does.
    (project / "scripts" / "start_llm.ps1").write_text(
        f"Set-Content -LiteralPath '{marker}' -Value started\n"
        f"$s = Start-Process -FilePath '{sys.executable}' -NoNewWindow -PassThru -ArgumentList "
        f"@('-m', 'http.server', '{port}', '--bind', '127.0.0.1', '--directory', '{served}')\n"
        f"Set-Content -LiteralPath '{pid_file}' -Value $s.Id\n"
        "$s.WaitForExit()\n",
        encoding="utf-8",
    )
    (fake_bin / "python.cmd").write_text(f'@echo %* > "{args_file}"\r\n@exit /b 7\r\n', encoding="ascii")
    env = {**os.environ, "USERPROFILE": str(tmp_path), "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(project / "run.ps1"),
             "-StartLLM", "-FaceWindow", "--once", "hi"],
            capture_output=True, text=True, timeout=120, env=env,
        )
        output = proc.stdout + proc.stderr
        assert marker.exists(), output  # Start-Process used to split "-File ...\ai robot\..." at the space.
        assert proc.returncode == 7, output  # The robot's exit code reaches the caller.
        assert args_file.read_text().split() == ["-m", "robot", "--face-window", "--once", "hi"]
        server_pid = int(pid_file.read_text().strip())
        deadline = time.monotonic() + 10
        while psutil.pid_exists(server_pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        # Stop-Process killed only the wrapper shell and orphaned llama-server with its VRAM and port.
        assert not psutil.pid_exists(server_pid)
    finally:
        if pid_file.exists():
            try:
                psutil.Process(int(pid_file.read_text().strip())).kill()
            except (psutil.Error, ValueError):
                pass
