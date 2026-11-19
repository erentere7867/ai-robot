"""Is this machine ready to run and test Sprocket? Checks each piece the way the robot uses it, says what is wrong and
how to fix it, starts the two servers when asked, and with --3d runs one task in the 3D apartment end to end.

    . tools/linux/env.sh
    python -m robot.eval.doctor                   # check
    python -m robot.eval.doctor --start --3d      # also start the model and vision servers, and run task M1 in 3D

On the Windows development PC the two servers are Windows programs (tools/brain.ps1, tools/win/vision.ps1); start them
there first, and the doctor finds them.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from robot import paths

TOOLS = paths.REPO / "tools" / "linux"
SETUP = "bash tools/linux/setup.sh"


class Problem(Exception):
    """What is wrong, and how to fix it."""

    def __init__(self, what: str, fix: str) -> None:
        super().__init__(f"{what}\n         fix: {fix}")


def python_packages() -> str:
    for name in ("numpy", "cv2", "httpx", "onnxruntime", "ultralytics"):
        try:
            importlib.import_module(name)
        except ImportError as exc:
            raise Problem(f"Python package {name} is missing ({exc})", SETUP) from None
    return "numpy, opencv, httpx, onnxruntime, ultralytics"


def ros() -> str:
    try:
        import rclpy  # noqa: F401
    except ImportError as exc:
        raise Problem(f"ROS 2 does not import ({exc})",
                      "source tools/linux/env.sh first (or run " + SETUP + ")") from None
    import os

    if "fastdds_shm.xml" not in os.environ.get("FASTDDS_DEFAULT_PROFILES_FILE", ""):
        raise Problem("ROS would talk over UDP, which delivers no data in WSL", "source tools/linux/env.sh")
    return "rclpy, shared-memory transport"


def habitat() -> str:
    from robot.sim.habitat import HABITAT_PYTHON

    data = Path("~/habitat-data/replica_cad").expanduser()
    if not Path(HABITAT_PYTHON).exists() or not data.is_dir():
        raise Problem("Habitat-Sim or the apartment's data is missing", "bash tools/linux/setup_habitat.sh")
    out = subprocess.run([HABITAT_PYTHON, "-c", "import habitat_sim; print(habitat_sim.__version__)"],
                         capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise Problem(f"Habitat-Sim does not import: {out.stderr.strip()[-300:]}", "bash tools/linux/setup_habitat.sh")
    return f"Habitat-Sim {out.stdout.strip()}, ReplicaCAD"


def models() -> str:
    out = subprocess.run(["bash", str(TOOLS / "setup_models.sh"), "--check"], capture_output=True, text=True)
    if out.returncode != 0:
        missing = [line.split("]")[1].strip() for line in out.stdout.splitlines() if "[MISS]" in line]
        raise Problem(f"models missing: {', '.join(missing)}", "bash tools/linux/setup_models.sh")
    return f"all present in {paths.MODELS}"


def server(name: str, url: str, start: str, started: bool) -> Callable[[], str]:
    def check() -> str:
        import httpx

        def up() -> bool:
            try:
                return httpx.get(f"{url}/health", timeout=3, trust_env=False).status_code == 200
            except httpx.HTTPError:
                return False

        if not up() and started:
            subprocess.run(["bash", str(TOOLS / start)], check=False)
            for _ in range(90):  # the language model takes a while to load
                if up():
                    break
                time.sleep(2)
        if not up():
            raise Problem(f"the {name} is not answering on {url}", f"bash tools/linux/{start} (or --start)")
        return f"answering on {url}"
    return check


def model_sees() -> str:
    import numpy as np

    from robot.mind.brain import Brain
    from robot.perception.camera import data_url

    picture = np.zeros((288, 512, 3), np.uint8)
    picture[:, :, 0] = 220  # all red
    t = time.perf_counter()
    yes, _ = Brain().check(data_url(picture), "Is this picture mostly red?")
    if not yes:
        raise Problem("the model answered no to an all-red picture: is its vision projector (mmproj) loaded?",
                      "check models/llm/mmproj-LFM3-VL-2B-F16.gguf and tools/linux/brain.sh")
    return f"reads a picture and answers in {time.perf_counter() - t:.1f} s"


def vision_finds() -> str:
    import numpy as np

    from robot.perception.detector import Detector

    detector = Detector()
    detector.watch("ball")
    t = time.perf_counter()
    detector(np.zeros((720, 1280, 3), np.uint8))
    return f"YOLOE answers a frame in {time.perf_counter() - t:.2f} s on {detector.server.info['providers'][0]}"


def apartment(rtf: float) -> str:
    """Task M1 ("turn left 90 degrees") on the whole stack: Habitat, ROS 2, Nav2, perception, the agent."""
    t = time.perf_counter()
    out = subprocess.run([sys.executable, "-u", "-m", "robot.eval.home3d", "--tag", "doctor", "--scenes", "apt_0",
                          "--only", "M1", "--rtf", str(rtf)], capture_output=True, text=True, timeout=7200)
    report = paths.RUNS / "doctor" / "apt_0" / "M1" / "report.html"
    if "[ok]" not in out.stdout:
        tail = "\n".join((out.stdout + out.stderr).strip().splitlines()[-12:])
        raise Problem(f"task M1 did not pass in the apartment:\n{tail}",
                      f"read {report} and runs/home_apt_0/robot.log")
    return f"task M1 passed in {(time.perf_counter() - t) / 60:.0f} min; report: {report}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check that this machine can run and test Sprocket.")
    parser.add_argument("--start", action="store_true", help="start the model and vision servers if they are not up")
    parser.add_argument("--3d", dest="three_d", action="store_true", help="also run task M1 in the 3D apartment")
    parser.add_argument("--rtf", type=float, default=0.33, help="for --3d: simulated seconds per second (0.1 on "
                                                                "a machine without a GPU)")
    args = parser.parse_args()
    checks: list[tuple[str, Callable[[], str]]] = [
        ("Python packages", python_packages),
        ("ROS 2", ros),
        ("Habitat", habitat),
        ("models", models),
        ("language model server", server("language model server", "http://127.0.0.1:8081", "brain.sh", args.start)),
        ("vision server", server("vision server", "http://127.0.0.1:8090", "vision.sh", args.start)),
        ("the model sees", model_sees),
        ("YOLOE finds", vision_finds),
    ]
    if args.three_d:
        checks.append(("3D apartment", lambda: apartment(args.rtf)))
    failed = 0
    for name, check in checks:
        try:
            print(f"[ok]   {name}: {check()}", flush=True)
        except Exception as exc:  # noqa: BLE001 - every check reports; one failure does not hide the others
            failed += 1
            print(f"[FAIL] {name}: {exc}", flush=True)
    print("\nready" if not failed else f"\n{failed} problem(s): fix them in order, then run the doctor again")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
