"""Run the full stack in the 3D simulation: Sprocket in Gazebo, Nav2, YOLOE-26 on its three cameras, the tracker.

    python -m robot.ros.run3d --perception             # how well the robot sees the room: tracked vs real objects
    python -m robot.ros.run3d --cases O1 O3 S1         # hand-written suite programs on real perception, judged
    python -m robot.ros.run3d --interactive --gui --brain http://127.0.0.1:8081   # Gazebo and the page

Run inside WSL with ROS sourced and the YOLOE environment active, from ~/sprocket-models (where the weights are).
Imports stay inside functions: the runner's sandbox process re-imports this module and must stay small.
"""

from __future__ import annotations

import argparse
import math
import tempfile
import time
from pathlib import Path

ROOMS = ("living room", "two rooms", "office")


def truth_scene(name: str):  # noqa: ANN201
    from robot.eval.suite import living_room, office, two_rooms
    from robot.ros.fuel import furnish

    return furnish({"living room": living_room, "two rooms": two_rooms, "office": office}[name]())


def start(scene_factory, gui: bool, brain_url: str | None):  # noqa: ANN001, ANN201
    """Gazebo and Nav2 with Sprocket in the scene, and a RealRobot on them."""
    import rclpy

    from robot.mind.brain import Brain
    from robot.perception.detector import Detector
    from robot.ros.real import RealRobot
    from robot.ros.stage import launch, set_initial_pose
    from robot.sim.world import SimRobot

    truth = scene_factory()
    sim = launch(SimRobot(truth), Path(tempfile.mkdtemp(prefix="sprocket-3d-")), gui=gui)
    rclpy.init()
    helper = rclpy.create_node("sprocket_run3d")
    print("starting Gazebo and Nav2...", flush=True)
    time.sleep(30.0 if gui else 25.0)
    set_initial_pose(helper, SimRobot(truth))
    print("loading YOLOE-26...", flush=True)
    detector = Detector(threads=3)  # the rest of the laptop's CPU runs Gazebo, Nav2 and the model
    robot = RealRobot(scene_factory(), detector=detector, brain=Brain(brain_url) if brain_url else None,
                      nudge=lambda: set_initial_pose(helper, SimRobot(truth)))
    return robot, truth, sim


def report(robot, truth) -> None:  # noqa: ANN001
    """Each real object against the nearest tracked one with its label; and tracks matching nothing real."""
    from robot.perception.tracker import canonical

    robot._perceive_objects()
    tracks = list(robot.things.values())
    used, errors = set(), []
    print(f"\n{'real object':<16} {'tracked as':<16} {'error':>7}")
    for t in sorted(truth.things, key=lambda t: t.label):
        if t.label == "door":
            continue
        same = [k for k in tracks if canonical(k.label) == canonical(t.label) and k.id not in used]
        best = min(same, key=lambda k: math.dist((k.x, k.y), (t.x, t.y)), default=None)
        if best is not None and math.dist((best.x, best.y), (t.x, t.y)) < 1.0:
            used.add(best.id)
            err = math.dist((best.x, best.y), (t.x, t.y))
            errors.append(err)
            print(f"{t.color + ' ' + t.label:<16} {best.color + ' ' + best.label + ' ' + str(best.id):<16} {err:6.2f} m")
        else:
            near = min((k for k in tracks if k.id not in used), key=lambda k: math.dist((k.x, k.y), (t.x, t.y)),
                       default=None)
            other = (f"(a {near.label} {near.id} is {math.dist((near.x, near.y), (t.x, t.y)):.2f} m away)"
                     if near is not None and math.dist((near.x, near.y), (t.x, t.y)) < 1.0 else "")
            print(f"{t.color + ' ' + t.label:<16} {'(not seen)':<16} {other}")
    extra = [k for k in tracks if k.id not in used]
    print(f"\nfound {len(errors)} of {sum(t.label != 'door' for t in truth.things)} real objects; median position error "
          f"{sorted(errors)[len(errors) // 2] if errors else float('nan'):.2f} m; tracks matching nothing real: "
          + (", ".join(f"{k.label} {k.id}" for k in extra) or "none"))
    dropped = [t for t in robot.perception.tracker.tracks.values() if t.rejected]
    if dropped:
        print("the vision model saw nothing in: " + ", ".join(
            f"{t.votes.most_common(1)[0][0]} {t.id} at ({t.x:.1f}, {t.y:.1f})" for t in dropped))
    timings = robot.perception.timings
    if timings:
        print(f"detector: median {1000 * sorted(timings)[len(timings) // 2]:.0f} ms per frame on this CPU")


def annotated(robot, name: str, width: int = 0):  # noqa: ANN001, ANN201
    """A camera's latest frame with each tracked object's box and name drawn on it (RGB array), or None."""
    import numpy as np
    from PIL import Image, ImageDraw

    frame = robot.perception.frames.get(name)
    if frame is None:
        return None
    image = Image.fromarray(np.ascontiguousarray(frame[1]))
    tracks = robot.perception.tracker.tracks
    draw = ImageDraw.Draw(image)
    for (x1, y1, x2, y2), tid in robot.perception.boxes.get(name, []):
        track = tracks.get(tid)
        if track is None or track.rejected:
            continue
        colour = (255, 0, 255) if track.confirmed else (160, 160, 160)
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=3)
        draw.text((x1 + 4, max(0, y1 - 14)), f"{track.label} {track.id}", fill=colour)
    if width and image.width > width:
        image = image.resize((width, round(image.height * width / image.width)))
    return np.asarray(image)


def snapshots(robot, folder: Path) -> None:  # noqa: ANN001
    """Each camera's latest frame with the tracked objects drawn on it, as PNG files."""
    from robot.perception.camera import encode_png

    folder.mkdir(parents=True, exist_ok=True)
    for name in list(robot.perception.frames):
        (folder / f"{name}.png").write_bytes(encode_png(annotated(robot, name)))
    print(f"camera pictures with what was tracked: {folder}")


def scan_points(robot, every: int = 3) -> list[tuple[float, float]]:  # noqa: ANN001
    """The lidar's hits on the map (the lidar sits over the robot's centre)."""
    scan = robot.perception.scan
    if scan is None:
        return []
    ranges, angle_min, step = scan
    points = []
    for k in range(0, len(ranges), every):
        r = float(ranges[k])
        if math.isfinite(r) and 0.05 < r < 12.0:
            a = robot.yaw + angle_min + k * step
            points.append((round(robot.x + r * math.cos(a), 2), round(robot.y + r * math.sin(a), 2)))
    return points


def interactive(robot, room: str, brain_url: str, port: int) -> None:  # noqa: ANN001
    """The live page on the 3D run: type an instruction, then watch the Gazebo window, the map with the lidar, the
    three cameras with what the robot tracks, and the program as it runs."""
    import threading

    from robot.agent.agent import Agent
    from robot.agent.planner import LocalPlanner
    from robot.interfaces.demo import Holder, drive
    from robot.interfaces.live import Live
    from robot.mind.brain import Brain
    from robot.perception.camera import encode_png
    from robot.ros.perceive import TOPICS

    def camera(name: str) -> bytes | None:
        image = annotated(robot, name, 480)
        return encode_png(image) if image is not None else None

    live = Live()
    live.serve(port)
    live.camera = camera
    live.update(scene=room, scenes=[room], cameras=list(TOPICS))

    def refresh() -> None:
        while True:
            try:
                robot._perceive_objects()
                live.show_world(robot)
                live.update(scan=scan_points(robot))
            except RuntimeError:  # a collection changed while being copied; take the next frame
                pass
            time.sleep(0.3)

    threading.Thread(target=refresh, daemon=True).start()
    brain = Brain(brain_url)
    planner = LocalPlanner(brain)
    if not brain.healthy():
        print(f"no model answering at {brain_url}: start llama-server there first", flush=True)
    else:
        print("the planner is reading its rules and examples...", flush=True)
        started = time.monotonic()
        planner.warm()
        print(f"...done in {time.monotonic() - started:.0f} s", flush=True)
    print(f"Open http://localhost:{port} and type an instruction. Ctrl+C to quit.", flush=True)
    drive(live, Agent(planner, live=live), Holder(robot))


def run_cases(ids: list[str], gui: bool, brain_url: str | None, look: float) -> None:
    """Hand-written suite programs on the full stack: each case's room in Gazebo, the robot seeing it through its
    cameras for `look` seconds first, the gate's dry run in the room as perceived, and the suite's check judged
    against the room as it really is."""
    from robot.agent.gate import check
    from robot.agent.runner import Runner
    from robot.eval.programs import PROGRAMS
    from robot.eval.suite import CASES
    from robot.ros.fuel import furnish
    from robot.ros.real import judge

    passed = 0
    cases = [c for c in CASES if c.id in ids and c.id in PROGRAMS]
    for case in cases:
        robot = sim = None
        try:
            robot, truth, sim = start(lambda c=case: furnish(c.scene()), gui, brain_url)
            time.sleep(look)
            robot._perceive_objects()
            robot.begin()
            gate = check(PROGRAMS[case.id], robot)
            if not gate.ok:
                outcome = f"gate: {gate}"
            else:
                result = Runner(robot).run(PROGRAMS[case.id])
                ok = result.status != "error" and bool(case.check(judge(robot, truth)))
                passed += ok
                outcome = "passed" if ok else (f"run: {result.error} (line {result.line})"
                                               if result.status == "error" else "check failed")
        except Exception as exc:  # noqa: BLE001 - one case's failure must not stop the others
            outcome = f"crashed: {exc}"
        finally:
            stop(robot, sim)
        print(f"{case.id:<4} {outcome:<44} {case.say}", flush=True)
    print(f"\n{passed} of {len(cases)} passed on the full stack")


def stop(robot, sim) -> None:  # noqa: ANN001
    import rclpy

    if robot is not None:
        robot.perception.stop()
        robot.shutdown()
    if rclpy.ok():
        rclpy.shutdown()
    if sim is not None:
        sim.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", choices=ROOMS, default="living room")
    parser.add_argument("--gui", action="store_true", help="open the Gazebo window")
    parser.add_argument("--perception", action="store_true", help="watch for a while and report what was seen")
    parser.add_argument("--watch", type=float, default=20.0, help="seconds of looking before reporting or a case")
    parser.add_argument("--interactive", action="store_true", help="the instruction page on --port")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cases", nargs="+", help="suite cases whose hand-written programs to run")
    parser.add_argument("--brain", default="", help="llama-server URL, e.g. http://127.0.0.1:8081: names new objects, "
                                                    "answers vision questions, and plans with --interactive")
    args = parser.parse_args()
    if args.cases:
        run_cases(args.cases, args.gui, args.brain or None, args.watch)
        return
    robot, truth, sim = start(lambda: truth_scene(args.room), args.gui, args.brain or None)
    try:
        if args.interactive:
            interactive(robot, args.room, args.brain or "http://127.0.0.1:8081", args.port)
        elif args.perception:
            print(f"watching for {args.watch:.0f} s...", flush=True)
            time.sleep(args.watch)
            report(robot, truth)
            print(f"frames processed per camera: {robot.perception.counts}")
            snapshots(robot, Path("/mnt/c/Users/FixAccount/robot-runtime/runs/3d"))
    finally:
        stop(robot, sim)


if __name__ == "__main__":
    main()
