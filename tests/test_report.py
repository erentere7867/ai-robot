"""The 3D eval's task page (robot/eval/report.py) from a made-up run: it draws the map and marks a believed thing
with no real one of its name as false."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot.agent.agent import Step, TaskRecord
from robot.eval import report
from robot.eval.home3d import TASKS, Referee, Task, keep_pictures, write_report
from robot.perception.tracker import Track
from robot.sim.world import Thing, Trace


def test_nearest_table_task_requires_ending_near_the_table_nearest_the_start():
    task = next(task for task in TASKS if task.id == "T1")
    truth = {"objects": [{"id": i, "label": "table", "x": x, "y": 0.0, "radius": 0.4}
                         for i, x in enumerate((2.0, 4.0), 1)]}
    home = SimpleNamespace(truth=truth, path=[(0.0, 0.0, 0.0, 0.0)])
    for x, passed in [(0.0, False), (1.5, True), (3.5, False)]:
        home.path[1:] = [(5.0, x, 0.0, 0.0)]
        assert task.check(Referee(home), {"xy": (0.0, 0.0)}) is passed


def test_nearest_table_task_uses_the_footprint_instead_of_its_longest_half_side():
    task = next(task for task in TASKS if task.id == "T1")
    table = {"id": 1, "label": "table", "x": 0.0, "y": 0.0, "radius": 1.0,
             "footprint": ((-0.2, -1.0), (0.2, -1.0), (0.2, 1.0), (-0.2, 1.0))}
    home = SimpleNamespace(truth={"objects": [table]}, path=[(0.0, 2.0, 0.0, 0.0)])
    for (x, y), passed in [((1.1, 0.0), False), ((0.9, 0.0), True), ((1.0, 1.3), False), ((0.0, 0.0), True)]:
        home.path[1:] = [(5.0, x, y, 0.0)]
        assert task.check(Referee(home), {"xy": (2.0, 0.0)}) is passed


@pytest.mark.parametrize(("task_id", "label"), [("T1", "table"), ("N3", "chair")])
def test_nearest_object_tasks_choose_the_closest_edge_not_the_closest_centre(task_id, label):
    task = next(task for task in TASKS if task.id == task_id)
    objects = [
        {"id": 1, "label": label, "x": 0.0, "y": 1.0, "radius": 0.1,
         "footprint": ((-0.1, 0.9), (0.1, 0.9), (0.1, 1.1), (-0.1, 1.1))},
        {"id": 2, "label": label, "x": 1.5, "y": 0.0, "radius": 1.3,
         "footprint": ((0.2, -0.2), (2.8, -0.2), (2.8, 0.2), (0.2, 0.2))},
    ]
    home = SimpleNamespace(truth={"objects": objects}, path=[(0.0, 0.0, 0.0, 0.0)])
    for (x, y), passed in [((0.0, 1.0), False), ((1.0, 0.0), True)]:
        home.path[1:] = [(5.0, x, y, 0.0)]
        assert task.check(Referee(home), {"xy": (0.0, 0.0)}) is passed


def _rect(i, label, x, y, half_x, half_y):
    return {"id": i, "label": label, "x": x, "y": y, "radius": max(half_x, half_y),
            "footprint": ((x - half_x, y - half_y), (x + half_x, y - half_y),
                          (x + half_x, y + half_y), (x - half_x, y + half_y))}


@pytest.mark.parametrize("task_id", ["N4", "P1"])
def test_nearest_table_to_sofa_uses_both_footprints(task_id):
    task = next(task for task in TASKS if task.id == task_id)
    objects = [_rect(1, "sofa", 0.0, 0.0, 2.0, 0.2), _rect(2, "table", 0.0, 1.5, 0.2, 0.2),
               _rect(3, "table", 2.5, 0.0, 0.2, 0.2)]
    home = SimpleNamespace(truth={"objects": objects}, path=[(0.0, 0.0, 0.0, 0.0)])
    for (x, y), passed in [((0.0, 0.8), False), ((2.2, 0.0), True)]:
        home.path[1:] = [(5.0, x, y, 0.0)]
        assert task.check(Referee(home), {"xy": (0.0, 0.0)}) is passed


@pytest.mark.parametrize("task_id", ["Q4", "C2"])
def test_sofa_vs_tv_uses_footprints_not_centres(task_id):
    task = next(task for task in TASKS if task.id == task_id)
    objects = [_rect(1, "sofa", 2.3, 0.0, 1.9, 0.2), _rect(2, "tv stand", 0.0, 1.0, 0.2, 0.1)]
    home = SimpleNamespace(truth={"objects": objects}, path=[(0.0, 0.0, 0.0, 0.0)])
    for label, (x, y), passed in [("TV", (0.0, 1.0), False), ("sofa", (2.3, 0.0), True)]:
        home.path[1:] = [(5.0, x, y, 0.0)]
        referee = Referee(home, SimpleNamespace(trace=Trace(said=[(5.0, f"The {label} is closer.")])))
        assert task.check(referee, {"xy": (0.0, 0.0)}) is passed


def test_closer_tv_task_accepts_arrival_at_tv_when_stand_is_closest_at_start():
    task = next(task for task in TASKS if task.id == "C2")
    objects = [_rect(1, "tv stand", 1.0, 0.0, 0.1, 0.1), _rect(2, "tv", 2.0, 0.0, 0.1, 0.1),
               _rect(3, "sofa", 4.0, 0.0, 0.1, 0.1)]
    home = SimpleNamespace(truth={"objects": objects}, path=[(0.0, 0.0, 0.0, 0.0)])
    for x, passed in [(1.0, True), (2.0, True), (4.0, False)]:
        home.path[1:] = [(5.0, x, 0.0, 0.0)]
        assert task.check(Referee(home), {"xy": (0.0, 0.0)}) is passed


def test_a_task_page_shows_the_map_the_turns_and_what_was_false(tmp_path):
    nx, ny = 40, 20  # 4 x 2 m at 0.1 m
    truth = {"width": 4.0, "height": 2.0, "res": 0.1, "shape": (nx, ny),
             "free": np.packbits(np.ones(nx * ny, bool)).tobytes(),
             "objects": [{"label": "sofa", "x": 1.0, "y": 0.0, "radius": 0.5}]}
    home = SimpleNamespace(truth=truth, path=[(0.0, -1.0, 0.0, 0.0), (5.0, 0.3, 0.0, 0.0)], dock=lambda: (-1.0, 0.0, 0.0))
    trace = Trace(poses=[(0.0, 0.0, 0.0, 0.0), (5.0, 1.3, 0.0, 0.0)])
    trace.asked.append((2.0, "search sofa: which way", "front 80%: went front"))
    real, false = Thing("sofa", 2.0, 0.0, 0.5, id=3), Thing("tv", 0.5, 0.8, 0.3, id=4)  # map frame: dock at (-1, 0)
    robot = SimpleNamespace(trace=trace, things={3: real, 4: false},
                            perception=SimpleNamespace(objects=lambda: [Track(3, "sofa", 2.0, 0.0, 0.5, 0, 0, hits=5),
                                                                        Track(4, "tv", 0.5, 0.8, 0.3, 0, 0, hits=2)]))
    rec = TaskRecord("Go to the sofa.", steps=[Step("", calls=[("go_to", {"target": "sofa"}, "Arrived.", True)],
                                                    thought="The sofa is ahead.", t=1.0), Step("Done!", t=6.0)])
    task = SimpleNamespace(id="N1", says=("Go to the sofa.",))
    row = {"scene": "apt_0", "id": "N1", "says": ("Go to the sofa.",), "passed": True, "problems": [],
           "simulated_s": 5.0, "bumps": 0}
    (tmp_path / "N1").mkdir()
    page = report.write(tmp_path / "N1", task, row, [rec], robot, home, 0, [(3.0, "001_front_camera.jpg")],
                        "ended at (0.3, 0.0)")
    text = page.read_text()
    assert (tmp_path / "N1" / "map.png").stat().st_size > 1000
    assert "The sofa is ahead." in text and "go_to(target=&#x27;sofa&#x27;)" in text and "001_front_camera.jpg" in text
    assert "sofa 3: a real one is there" in text and "tv 4: no real one there; nearest real thing: sofa" in text
    assert text.index("The sofa is ahead.") < text.index("search sofa") < text.index("001_front_camera.jpg")
    assert "apt_0/N1/report.html" in report.index(tmp_path, [row]).read_text()


def test_same_mission_keeps_each_layout_report_and_photos(tmp_path):
    import cv2

    task = Task("M1", ("Move forward.",), lambda referee, start: True)
    rows = []
    for scene, passed, x, shade in [("apt_0", True, 0.2, 40), ("apt_4", False, 0.8, 210)]:
        home = SimpleNamespace(scene=scene, dock=lambda: (0.0, 0.0, 0.0), path=[(1.0, x, 0.0, 0.0)],
                               truth={"width": 1.0, "height": 1.0, "res": 0.5, "shape": (2, 2),
                                      "free": np.packbits(np.ones(4, bool)).tobytes(), "objects": []})
        robot = SimpleNamespace(t=1.0, things={}, trace=Trace(),
                                camera_view=lambda camera, width, height, shade=shade: np.full((8, 8, 3), shade, np.uint8))
        folder = tmp_path / scene / task.id
        kept = keep_pictures(robot, folder)
        robot.camera_view()
        rows.append({"scene": scene, "id": task.id, "says": task.says, "passed": passed,
                     "problems": [] if passed else ["ReadTimeout: timed out"], "simulated_s": 1.0, "bumps": 0})
        write_report(tmp_path, task, rows, [], robot, home, 0, kept, Referee(home))

    index = (tmp_path / "index.html").read_text()
    for scene, verdict, x, shade in [("apt_0", "passed", "0.2", 40), ("apt_4", "FAILED", "0.8", 210)]:
        folder = tmp_path / scene / task.id
        page = (folder / "report.html").read_text()
        assert f"{verdict}</span>" in page
        assert ("ReadTimeout: timed out" in page) == (verdict == "FAILED")  # why it failed, on the failed one only
        assert f"ended at ({x}, 0.0)" in page
        assert "001_front_camera.jpg" in page
        assert abs(float(cv2.imread(str(folder / "001_front_camera.jpg")).mean()) - shade) < 2
        assert (folder / "map.png").is_file()
        assert f"{scene}/M1/report.html" in index
        assert f"{scene}" in index.split(f"{scene}/M1/report.html'>", 1)[1].split("</a>", 1)[0]


def test_views_score_counts_found_and_false_against_the_truth(tmp_path, capsys):
    from robot.eval.views import score

    px = np.array([[900, 0], [500, 0], [0, 0], [0, 40]])  # columns: chair, tv stand; four pictures
    np.savez(tmp_path / "v.npz", rgb=np.zeros((4, 2, 2, 3), np.uint8), labels=np.array(["chair", "tv stand"]), px=px)
    answers = iter([True, False, True, False] + [False, False, True, False])  # chair asked first, then tv
    score(tmp_path / "v.npz", ["chair", "tv"], lambda rgb, word: next(answers), clear=300)
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].split() == ["chair", "2", "2", "50%", "50%"]  # in pictures 1-2 (one yes), absent in 3-4 (one yes)
    assert lines[2].split() == ["tv", "0", "3", "-", "33%"]  # the tv stand only 40 px: never plainly in view
