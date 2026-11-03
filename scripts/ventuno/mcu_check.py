"""Bring-up check for the VENTUNO Q's real-time controller: is the STM32H5 talking, and does it drive?

    python scripts/ventuno/mcu_check.py              # 5 s of telemetry: rate, pose, range, faults, battery
    python scripts/ventuno/mcu_check.py --wiggle     # also turn left then right for a second each (wheels up!)

Uses the same link as the robot (robot/body/ventuno_body.py: msgpack-RPC over the Arduino Router socket).
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robot.body.ventuno_body import ROUTER_SOCKET, RouterLink  # noqa: E402

FLAGS = {1: "bumper", 2: "cliff", 4: "estop", 8: "picked up", 16: "charging", 32: "forward blocked",
         64: "wheel stall", 128: "IMU fault", 256: "range sensor fault", 512: "battery low"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--socket", default=ROUTER_SOCKET)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--wiggle", action="store_true", help="turn in place briefly (lift the wheels first)")
    args = parser.parse_args()

    samples: list[tuple[float, tuple]] = []
    lock = threading.Lock()

    def on_telemetry(*values: object) -> None:
        with lock:
            samples.append((time.monotonic(), values))

    link = RouterLink(args.socket)
    try:
        link.open()
    except OSError as exc:
        print(f"Cannot reach the Arduino Router at {args.socket}: {exc}")
        return 2
    link.provide("telemetry", on_telemetry)
    link.notify("stop")
    end = time.monotonic() + args.seconds
    phases = [(0.0, 0, 0)]
    if args.wiggle:
        phases = [(1.0, 0, 60000), (2.0, 0, -60000), (3.0, 0, 0)]
    start = time.monotonic()
    while time.monotonic() < end:
        t = time.monotonic() - start
        v, w = next(((pv, pw) for until, pv, pw in phases if t < until), (0, 0))
        link.notify("cmd_vel", v, w)  # refreshed at 20 Hz: the watchdog stops the wheels 300 ms after the last
        time.sleep(0.05)
    link.notify("stop")
    link.close()

    with lock:
        got = list(samples)
    if not got:
        print("No telemetry: is firmware/ventuno_mcu flashed and running?")
        return 1
    span = got[-1][0] - got[0][0]
    rate = (len(got) - 1) / span if span > 0 else 0.0
    gaps = [b[0] - a[0] for a, b in zip(got, got[1:], strict=False)]
    ms, x, y, th, v, w, tl, tr, rng, flags, batt, head = got[-1][1]
    print(f"telemetry: {len(got)} messages, {rate:.1f} Hz (want 50), worst gap {1000 * max(gaps, default=0):.0f} ms")
    print(f"pose: x {x} mm, y {y} mm, heading {th / 1000:.1f} deg; speed {v} mm/s, {w / 1000:.1f} deg/s")
    print(f"encoders: {tl} / {tr}; range {rng} mm; battery {batt} mV; head {head / 10:.1f} deg")
    print("flags:", ", ".join(name for bit, name in FLAGS.items() if flags & bit) or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
