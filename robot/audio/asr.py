"""Local speech-to-text: whisper.cpp on the robot, Windows offline recognition on a development laptop."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
# whisper.cpp transcribes silence and noise as tags such as "[BLANK_AUDIO]" or "(wind blowing)".
_NON_SPEECH = re.compile(r"\[[^\]]*\]|\([^)]*\)|[♪♫]")


@dataclass(frozen=True)
class ASRResult:
    """Transcription result returned by a SpeechRecognizer."""

    text: str
    confidence: float = 1.0
    latency_ms: float = 0.0
    backend_name: str = "unknown"


class SpeechRecognizer:
    """Transcribes mono float32 audio utterances into text offline."""

    def __init__(
        self,
        *,
        mode: str = "auto",  # "auto" | "whisper_cpp" | "windows_offline"
        whisper_bin: Path | None = None,
        whisper_model: Path | None = None,
    ) -> None:
        self.mode = mode
        self.whisper_bin = (
            whisper_bin or Path("~/robot-runtime/whisper.cpp/build/bin/whisper-cli").expanduser()
        )
        self.whisper_model = whisper_model or Path("~/robot-runtime/models/ggml-tiny.en.bin").expanduser()

    def transcribe_samples(self, samples: np.ndarray, sample_rate: int = 16_000) -> ASRResult:
        """Write `samples` to a temporary WAV file and transcribe offline."""
        if len(samples) == 0:
            return ASRResult(text="", confidence=0.0, backend_name="empty")

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "utterance.wav"
            pcm16 = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm16.tobytes())
            return self.transcribe_wav(wav_path)

    def transcribe_wav(self, wav_path: Path | str) -> ASRResult:
        path = Path(wav_path).expanduser().resolve()
        if not path.exists():
            return ASRResult(text="", confidence=0.0, backend_name="missing_file")
        if self.mode in ("auto", "whisper_cpp") and self.whisper_bin.exists() and self.whisper_model.exists():
            result = self._try_whisper_cpp(path)
            if result is not None:
                return result

        if self.mode in ("auto", "windows", "windows_offline") and sys.platform == "win32":
            res = self._try_windows_offline_asr(path)
            if res is not None:
                return res

        return ASRResult(text="", confidence=0.0, backend_name="unavailable")

    def _try_whisper_cpp(self, wav_path: Path) -> ASRResult | None:
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [
                    str(self.whisper_bin),
                    "-m",
                    str(self.whisper_model),
                    "-f",
                    str(wav_path),
                    "-nt",
                    "-np",
                    "-t",
                    "2",
                ],
                check=True,
                capture_output=True,
                # whisper.cpp prints UTF-8; the locale code page (cp1252 on Windows) garbles or rejects it.
                encoding="utf-8",
                errors="replace",
                timeout=15.0,
                creationflags=_NO_WINDOW,
            )
            text = " ".join(_NON_SPEECH.sub(" ", proc.stdout).split())[:4096]
            return ASRResult(text, 0.8 if text else 0.0, (time.perf_counter() - start) * 1000, "whisper_cpp")
        except (OSError, subprocess.SubprocessError):
            return None

    @staticmethod
    def _try_windows_offline_asr(wav_path: Path) -> ASRResult | None:
        t0 = time.perf_counter()
        quoted_path = str(wav_path).replace("'", "''")
        ps_script = (
            "Add-Type -AssemblyName System.Speech; "
            "$r = New-Object System.Speech.Recognition.SpeechRecognitionEngine("
            "[System.Globalization.CultureInfo]::GetCultureInfo('en-US')); "
            "$r.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar)); "
            f"$r.SetInputToWaveFile('{quoted_path}'); "
            "$res = $r.Recognize(); "
            "$r.Dispose(); "
            "if ($res) { Write-Output ($res.Confidence.ToString('0.00') + '|' + $res.Text) }"
        )
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                check=True,
                capture_output=True,
                text=True,
                timeout=6.0,
                creationflags=_NO_WINDOW,
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            out = proc.stdout.strip()
            if "|" in out:
                conf_str, txt = out.split("|", 1)
                try:
                    conf = float(conf_str.replace(",", "."))
                except ValueError:
                    conf = 0.8
                return ASRResult(
                    text=txt.strip(),
                    confidence=conf,
                    latency_ms=round(dt_ms, 1),
                    backend_name="windows_offline_asr",
                )
            return ASRResult(
                text="", confidence=0.0, latency_ms=round(dt_ms, 1), backend_name="windows_offline_asr"
            )
        except Exception:
            return None
