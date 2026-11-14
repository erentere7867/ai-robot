"""Run the full stack in the 3D simulation: Sprocket in Gazebo, Nav2, YOLOE-26 on its three cameras, the tracker.

    python -m robot.ros.run3d --perception             # how well the robot sees the room: tracked vs real objects
    python -m robot.ros.run3d --cases O1 O3 S1         # hand-written suite programs on real perception, judged
    python -m robot.ros.run3d --cases O1 O3 --plan --brain http://127.0.0.1:8081  # the planner's programs, judged
    python -m robot.ros.run3d --interactive --gui --brain http://127.0.0.1:8081   # Gazebo and the page

All of them serve the live page on --port (8765): the map with the lidar, the cameras and the program as it runs.

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


def start(scene_factory, gui: bool, brain_url: str | None, memory: str = "", slam: bool = False,  # noqa: ANN001, ANN201
          map_file: Path | None = None):
    """Gazebo and Nav2 with Sprocket in the scene, and a RealRobot on them (remembering in the file `memory`; mapping
    as it goes with `slam`; localising in its own `map_file`)."""
    import rclpy

    from robot.mind.brain import Brain
    from robot.mind.memory import Memory
    from robot.perception.detector import Detector
    from robot.ros.real import RealRobot
    from robot.ros.stage import launch, set_initial_pose
    from robot.sim.world import SimRobot

    truth = scene_factory()
    sim = launch(SimRobot(truth), Path(tempfile.mkdtemp(prefix="sprocket-3d-")), gui=gui, slam=slam,
                 map_file=map_file)
    try:
        if not rclpy.ok():
            rclpy.init()
        helper = rclpy.create_node("sprocket_run3d")
        print("starting Gazebo and Nav2...", flush=True)
        time.sleep(30.0 if gui else 25.0)
        set_initial_pose(helper, SimRobot(truth))
        print("loading YOLOE-26...", flush=True)
        detector = Detector(threads=3)  # the rest of the laptop's CPU runs Gazebo, Nav2 and the model
        grid = None
        if map_file is not None:  # plan and dry-run in the map the robot built, not the simulator's
            from robot.ros.explore import load_grid

            grid = load_grid(map_file, SimRobot(truth).occupied.shape, (truth.width, truth.height))
        robot = RealRobot(scene_factory(), detector=detector, brain=Brain(brain_url) if brain_url else None,
                          nudge=lambda: set_initial_pose(helper, SimRobot(truth)),
                          store=Memory(memory) if memory else None, grid=grid)
        if grid is not None:  # the guided tour, done once at setup: "this is the kitchen", in the middle of each room
            from robot.agent.api import Pose

            robot.places.update({name: Pose((x0 + x1) / 2, (y0 + y1) / 2, 0.0)
                                 for name, (x0, y0, x1, y1) in truth.rooms.items()})
    except BaseException:  # a simulation that failed to come up must not keep running under the next one
        stop(None, sim)
        raise
    return robot, truth, sim


def report(robot, truth, shift: tuple[float, float] = (0.0, 0.0)) -> None:  # noqa: ANN001
    """Each real object against the nearest tracked one with its label; and tracks matching nothing real. `shift` takes
    the robot's map frame to the house's (a map being built starts where the robot started)."""
    from dataclasses import replace

    from robot.perception.tracker import canonical

    robot._perceive_objects()
    tracks = [replace(k, x=k.x + shift[0], y=k.y + shift[1]) for k in robot.things.values()]
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
          + (", ".join(f"{k.label} {k.id} at ({k.x:.1f}, {k.y:.1f}) r {k.radius:.2f}" for k in extra) or "none"))
    dropped = [t for t in robot.perception.tracker.tracks.values() if t.rejected]
    if dropped:
        print("the vision model saw nothing in: " + ", ".join(
            f"{t.votes.most_common(1)[0][0]} {t.id} at ({t.x:.1f}, {t.y:.1f})" for t in dropped))
    pairs = list(robot.perception.depth_pairs)
    if pairs:
        misses = sorted(abs(g) for g in pairs if g)
        print(f"front frames with their own depth image: {len(pairs) - len(misses)} of {len(pairs)}"
              + (f"; when not, the nearest was {1000 * misses[len(misses) // 2]:.0f} ms off (median)" if misses else ""))
    timings = robot.perception.timings
    if timings:
        print(f"detector: median {1000 * sorted(timings)[len(timings) // 2]:.0f} ms per frame")


def steadiness(robot, samples: list) -> None:  # noqa: ANN001
    """How much each tracked object moves on the map while it stands still (its wobble: the spread of its place over
    the second half of the watch), and where each camera puts it, from the raw sightings."""
    import statistics

    if len(samples) < 4:
        return
    late = samples[len(samples) // 2:]
    since = late[0][0]
    print(f"\n{'object':<14} {'wobble':>7} {'largest jump':>13}   sightings by camera: mean place, spread")
    for track in robot.perception.objects():
        path = [(t.x, t.y) for _, tracks in late for t in tracks if t.id == track.id]
        if len(path) < 3:
            continue
        wobble = math.hypot(statistics.pstdev(p[0] for p in path), statistics.pstdev(p[1] for p in path))
        jump = max(math.dist(a, b) for a, b in zip(path, path[1:], strict=False))
        cams = {}
        for t, camera, tid, x, y, *_ in list(robot.perception.sightings):
            if tid == track.id and t >= since:
                cams.setdefault(camera, []).append((x, y))
        by_camera = "; ".join(
            f"{c.split('_')[0] if c == 'front_camera' else c.split('_')[1]} ({statistics.fmean(p[0] for p in ps):.2f}, "
            f"{statistics.fmean(p[1] for p in ps):.2f}) ±{math.hypot(statistics.pstdev(p[0] for p in ps), statistics.pstdev(p[1] for p in ps)):.2f} x{len(ps)}"
            for c, ps in sorted(cams.items()))
        print(f"{track.label + ' ' + str(track.id):<14} {wobble:6.2f} m {jump:10.2f} m   {by_camera}")
    print(f"frames placed with the latest pose instead of the frame's own: {getattr(robot, 'stale_poses', 0)}")
    low, high = robot.perception.pitch_range
    print(f"body pitch seen by the IMU: {math.degrees(low):+.2f} to {math.degrees(high):+.2f} deg "
          f"({len(robot.perception.tilts)} recent readings)")


def range_errors(robot, truth) -> None:  # noqa: ANN001
    """For each camera and way of finding range: the sighting's distance from the robot over the distance of the real
    object on the nearest bearing (1.00 is right), the share of sightings more than 30% long, and the depth image's
    lag behind the colour picture for those."""
    import statistics

    real = [(t.x, t.y, t.radius) for t in truth.things if t.label != "door"]
    groups: dict[tuple[str, str], list[tuple[float, float]]] = {}
    def off(p: tuple, rx: float, ry: float, heading: float) -> float:
        d = math.atan2(p[1] - ry, p[0] - rx) - heading
        return abs(math.atan2(math.sin(d), math.cos(d)))

    for _, camera, _, x, y, how, rx, ry, gap in list(robot.perception.sightings):
        heading = math.atan2(y - ry, x - rx)
        near = min(real, key=lambda p, rx=rx, ry=ry, h=heading: off(p, rx, ry, h))
        reach = math.dist(near[:2], (rx, ry))
        if off(near, rx, ry, heading) < math.radians(8) and reach > 0.3:
            groups.setdefault((camera, how), []).append((math.dist((x, y), (rx, ry)) / reach, gap))
    print(f"\n{'camera':<18} {'range from':<6} {'sightings':>9} {'median ratio':>13} {'middle half':>15} "
          f"{'>30% long':>10} {'their depth lag':>16}")
    for (camera, how), rows in sorted(groups.items()):
        ratios = sorted(r for r, _ in rows)
        q1, q3 = ratios[len(ratios) // 4], ratios[3 * len(ratios) // 4]
        long = [g for r, g in rows if r > 1.3]
        lag = f"{1000 * statistics.median(long):.0f} ms" if long else "-"
        print(f"{camera:<18} {how:<6} {len(rows):9d} {statistics.median(ratios):13.2f} {q1:7.2f}–{q3:<7.2f} "
              f"{len(long) / len(rows):9.0%} {lag:>16}")


def tour(robot, truth) -> None:  # noqa: ANN001
    """Drive to a free spot near each corner of the room and back to the start, the way a robot moves about a home."""
    import numpy as np

    from robot.sim.world import SimRobot

    sim = SimRobot(truth)
    nx, ny = sim.inflated.shape
    free = np.argwhere(~sim.inflated)
    cell = truth.width / nx
    for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        want = (sx * (truth.width / 2 - 1.2), sy * (truth.height / 2 - 1.0))
        i, j = min(free, key=lambda c: math.dist(want, (-truth.width / 2 + (c[0] + 0.5) * cell,
                                                        -truth.height / 2 + (c[1] + 0.5) * cell)))
        goal = (-truth.width / 2 + (i + 0.5) * cell, -truth.height / 2 + (j + 0.5) * cell)
        print(f"touring: to ({goal[0]:.1f}, {goal[1]:.1f})", flush=True)
        try:
            robot.go_to(goal)
        except Exception as exc:  # noqa: BLE001 - one unreachable corner does not end the tour
            print(f"touring: could not reach it: {exc}", flush=True)
    robot.go_to("start")


def annotated(robot, name: str, width: int = 0):  # noqa: ANN001, ANN201
    """A camera's latest frame with each tracked object's box and name drawn on it (RGB array), or None."""
    import numpy as np
    from PIL import Image, ImageDraw

    picture = robot.perception.detected.get(name)  # the frame the boxes were found in, not a later one
    if picture is None:
        return None
    image = Image.fromarray(np.ascontiguousarray(picture))
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
    """The lidar's hits on the map (the lidar sits LIDAR_X ahead of the robot's centre)."""
    from robot.ros.model import LIDAR_X

    scan = robot.perception.scan
    if scan is None:
        return []
    ranges, angle_min, step = scan
    points = []
    for k in range(0, len(ranges), every):
        r = float(ranges[k])
        if math.isfinite(r) and 0.05 < r < 12.0:
            a = robot.yaw + angle_min + k * step
            points.append((round(robot.x + LIDAR_X * math.cos(robot.yaw) + r * math.cos(a), 2),
                           round(robot.y + LIDAR_X * math.sin(robot.yaw) + r * math.sin(a), 2)))
    return points


def page(holder, port: int):  # noqa: ANN001, ANN201
    """The live page for whatever robot `holder.sim` is now: the map with the lidar, the three cameras with what the
    robot tracks, the program as it runs, and a box for instructions (they arrive in the returned Live's inbox)."""
    import threading

    from robot.interfaces.live import Live
    from robot.perception.camera import encode_png
    from robot.ros.perceive import TOPICS

    def camera(name: str) -> bytes | None:
        image = annotated(holder.sim, name, 480) if holder.sim is not None else None
        return encode_png(image) if image is not None else None

    live = Live()
    live.serve(port)
    live.camera = camera
    live.update(cameras=list(TOPICS))
    from robot.interfaces.voice import attach

    if attach(live):
        print("the page can hear (hold to talk) and speak (tick 'speak aloud')", flush=True)

    def refresh() -> None:
        while True:
            robot = holder.sim
            try:
                if robot is not None and robot.perception is not None:
                    robot._perceive_objects()
                    live.show_world(robot)
                    live.update(scan=scan_points(robot))
            except (RuntimeError, AttributeError):  # changing while copied, or between cases; take the next frame
                pass
            time.sleep(0.3)

    threading.Thread(target=refresh, daemon=True).start()
    print(f"watch it at http://localhost:{port}", flush=True)
    return live


def planner_for(brain_url: str):  # noqa: ANN201
    """The on-board planner on the model at `brain_url`, its rules and examples already read."""
    from robot.agent.planner import LocalPlanner
    from robot.mind.brain import Brain

    brain = Brain(brain_url)
    planner = LocalPlanner(brain)
    if not brain.healthy():
        print(f"no model answering at {brain_url}: start llama-server there first", flush=True)
        return planner
    started = time.monotonic()
    planner.warm()
    print(f"the planner read its rules and examples in {time.monotonic() - started:.0f} s", flush=True)
    return planner


def interactive(robot, room: str, brain_url: str, port: int) -> None:  # noqa: ANN001
    """Type an instruction on the live page, then watch the Gazebo window and the page as the robot does it."""
    from robot.agent.agent import Agent
    from robot.interfaces.demo import Holder, drive

    holder = Holder(robot)
    live = page(holder, port)
    live.update(scene=room, scenes=[room])
    planner = planner_for(brain_url)
    print("type an instruction on the page. Ctrl+C to quit.", flush=True)
    drive(live, Agent(planner, live=live), holder)


def run_cases(ids: list[str], gui: bool, brain_url: str | None, look: float, port: int, plan: bool,
              own_map: bool = False) -> None:
    """Suite cases on the full stack: each case's room in Gazebo, the robot seeing it through its cameras for `look`
    seconds first, then the hand-written program (or, with `plan`, the planner's own from the case's words) through
    the gate's dry run in the room as perceived, and the suite's check judged against the room as it really is."""
    from robot.agent.agent import Agent
    from robot.agent.gate import check
    from robot.agent.runner import Runner
    from robot.eval.programs import PROGRAMS
    from robot.eval.suite import CASES
    from robot.interfaces.demo import Holder
    from robot.ros.fuel import furnish
    from robot.ros.real import judge

    holder = Holder(None)
    live = page(holder, port)
    agent = Agent(planner_for(brain_url or "http://127.0.0.1:8081"), live=live) if plan else None
    passed = 0
    cases = [c for c in CASES if c.id in ids and (plan or c.id in PROGRAMS)]
    for case in cases:
        robot = sim = None
        started = time.monotonic()
        print(f"\n== {case.id}: {case.say}", flush=True)
        try:
            mapped = Path(f"~/sprocket-maps/{getattr(case.scene, '__name__', '')}.yaml").expanduser()
            robot, truth, sim = start(lambda c=case: furnish(c.scene()), gui, brain_url,
                                      map_file=mapped if own_map and mapped.exists() else None)
            holder.sim = robot
            live.update(scene=case.id, scenes=[case.id], status="idle", instruction=case.say, program="", plan="",
                        readback="", problems=[], line=0)
            time.sleep(look)
            robot._perceive_objects()
            begun = time.monotonic()
            if agent is not None:
                agent.earlier.clear()  # each case is a conversation of its own
                record = agent.do(case.say, robot)
                ok = record.outcome == "done" and bool(case.check(judge(robot, truth)))
                outcome = ("passed" if ok else "ran, check failed" if record.outcome == "done" else
                           f"{record.outcome}: {record.problems[-1][:60] if record.problems else ''}")
                if not ok:
                    print("  the planner's last program:\n" + "\n".join(
                        "    " + line for line in (record.programs[-1] if record.programs else "").splitlines()))
            else:
                robot.begin()
                gate = check(PROGRAMS[case.id], robot)
                result = Runner(robot).run(PROGRAMS[case.id]) if gate.ok else None
                ok = result is not None and result.status != "error" and bool(case.check(judge(robot, truth)))
                outcome = ("passed" if ok else f"gate: {gate}" if result is None else
                           f"run: {result.error} (line {result.line})" if result.status == "error" else "check failed")
            if not ok:
                print("  the robot knew: " + ", ".join(sorted(f"{t.color} {t.label} {t.id} at ({t.x:.1f}, {t.y:.1f})"
                                                             for t in robot.things.values())))
                print("  really there:   " + ", ".join(sorted(f"{t.color} {t.label} at ({t.x:.1f}, {t.y:.1f})"
                                                             for t in truth.things)))
            passed += ok
        except Exception as exc:  # noqa: BLE001 - one case's failure must not stop the others
            import traceback

            outcome, begun = f"crashed: {type(exc).__name__}: {exc}", time.monotonic()
            traceback.print_exc()
        finally:
            holder.sim = None
            stop(robot, sim)
        print(f"{case.id:<4} {outcome:<50} {time.monotonic() - begun:5.0f} s task, {time.monotonic() - started:4.0f} s "
              f"in all | {case.say}", flush=True)
    print(f"\n{passed} of {len(cases)} passed on the full stack ({'planner' if plan else 'hand-written programs'})")


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
    parser.add_argument("--turn", action="store_true", help="with --perception: turn once, slowly, while watching")
    parser.add_argument("--tour", action="store_true", help="with --perception: drive round the room while watching")
    parser.add_argument("--interactive", action="store_true", help="the instruction page on --port")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cases", nargs="+", help="suite cases to run on the full stack")
    parser.add_argument("--plan", action="store_true", help="with --cases: the planner writes each program")
    parser.add_argument("--brain", default="", help="llama-server URL, e.g. http://127.0.0.1:8081: names new objects, "
                                                    "answers vision questions, and plans")
    parser.add_argument("--memory", default="", help="the memory file; --interactive uses ~/sprocket-memory/<room>.db")
    parser.add_argument("--forget", action="store_true", help="start with an empty memory")
    parser.add_argument("--explore", action="store_true", help="map the room from scratch (slam_toolbox, frontier "
                                                               "exploration) and save the map to ~/sprocket-maps")
    parser.add_argument("--own-map", action="store_true", help="localise in the map --explore saved, not the "
                                                               "simulator's")
    args = parser.parse_args()
    if args.cases:
        run_cases(args.cases, args.gui, args.brain or None, args.watch, args.port, args.plan, args.own_map)
        return
    memory = args.memory or (f"~/sprocket-memory/{args.room}.db" if args.interactive else "")
    if memory and args.forget:
        Path(memory).expanduser().unlink(missing_ok=True)
    own = Path(f"~/sprocket-maps/{args.room.replace(' ', '_')}.yaml").expanduser()
    robot, truth, sim = start(lambda: truth_scene(args.room), args.gui, args.brain or None, memory,
                              slam=args.explore, map_file=own if args.own_map and own.exists() else None)
    try:
        if args.explore:
            from robot.ros.explore import Explorer, coverage
            from robot.sim.world import SimRobot

            explorer = Explorer(robot)
            stats = explorer.explore(max(args.watch, 60.0))
            saved = explorer.save(own.with_suffix(""), (truth.start.x, truth.start.y))
            print(f"explored in {stats['seconds']:.0f} s: {stats['goals']} frontiers visited, {stats['unreachable']} "
                  f"unreachable, {stats['frontiers left']} left; the map shows {coverage(saved, SimRobot(truth)):.0%} "
                  f"of the free floor; saved to {saved}", flush=True)
            report(robot, truth, (truth.start.x, truth.start.y))
        elif args.interactive:
            interactive(robot, args.room, args.brain or "http://127.0.0.1:8081", args.port)
        elif args.perception:
            import threading

            from nav_msgs.msg import Odometry

            what = "driving round the room" if args.tour else "turning slowly" if args.turn else "standing still"
            print(f"watching, {what}...", flush=True)
            samples: list[tuple[float, list]] = []
            corrections: list[tuple[float, float, float]] = []  # map -> odom: how localisation corrects odometry
            lost: list[tuple[float, float]] = []  # localisation error: metres and degrees from the true pose
            really: dict[str, tuple[float, float, float]] = {}
            done = threading.Event()

            def yaw_of(q: object) -> float:
                return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

            def truth_pose(msg: object) -> None:
                p = msg.pose.pose
                really["pose"] = (p.position.x, p.position.y, yaw_of(p.orientation))

            robot.node.create_subscription(Odometry, "/ground_truth", truth_pose, 10)

            def sample() -> None:
                from rclpy.time import Time

                while not done.is_set():
                    samples.append((time.monotonic(), robot.perception.objects()))
                    try:
                        tr = robot.tf.lookup_transform("map", "odom", Time()).transform
                        corrections.append((tr.translation.x, tr.translation.y, math.degrees(yaw_of(tr.rotation))))
                        me = robot.tf.lookup_transform("map", "base_link", Time()).transform
                        if "pose" in really:
                            x, y, yaw = really["pose"]
                            off = math.degrees(yaw_of(me.rotation) - yaw)
                            lost.append((math.dist((me.translation.x, me.translation.y), (x, y)),
                                         abs((off + 180) % 360 - 180)))
                    except Exception:  # noqa: BLE001 - not yet localised
                        pass
                    time.sleep(0.5)

            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            if args.tour:
                tour(robot, truth)
            elif args.turn:
                robot.spin(turns=1.0, seconds=args.watch, direction="left")
            else:
                time.sleep(args.watch)
            done.set()
            sampler.join()
            if lost:
                metres, degrees = sorted(m for m, _ in lost), sorted(d for _, d in lost)
                print(f"localisation error against the true pose: median {metres[len(metres) // 2]:.2f} m and "
                      f"{degrees[len(degrees) // 2]:.1f} deg, worst {metres[-1]:.2f} m and {degrees[-1]:.1f} deg "
                      f"({len(lost)} samples)", flush=True)
            report(robot, truth)
            steadiness(robot, samples)
            range_errors(robot, truth)
            if len(corrections) > 1:
                steps = [(math.dist(a[:2], b[:2]), abs(b[2] - a[2])) for a, b in zip(corrections, corrections[1:],
                                                                                     strict=False)]
                print(f"localisation corrections (map -> odom) over the watch: moved {math.dist(corrections[0][:2], corrections[-1][:2]):.2f} m and "
                      f"{corrections[-1][2] - corrections[0][2]:+.1f} deg in all; largest half-second step "
                      f"{max(s[0] for s in steps):.2f} m, {max(s[1] for s in steps):.1f} deg")
            print(f"frames processed per camera: {robot.perception.counts}")
            snapshots(robot, Path("/mnt/c/Users/FixAccount/robot-runtime/runs/3d"))
    finally:
        stop(robot, sim)


if __name__ == "__main__":
    main()
