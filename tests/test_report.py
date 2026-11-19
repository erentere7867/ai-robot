"""The 3D eval's task page (robot/eval/report.py) from a made-up run: it draws the map and marks a believed thing
with no real one of its name as false."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from robot.agent.agent import Step, TaskRecord
from robot.eval import report
from robot.perception.tracker import Track
from robot.sim.world import Thing, Trace


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
    row = {"id": "N1", "says": ("Go to the sofa.",), "passed": True, "simulated_s": 5.0, "bumps": 0}
    (tmp_path / "N1").mkdir()
    page = report.write(tmp_path / "N1", task, row, [rec], robot, home, 0, [(3.0, "001_front_camera.jpg")],
                        "ended at (0.3, 0.0)")
    text = page.read_text()
    assert (tmp_path / "N1" / "map.png").stat().st_size > 1000
    assert "The sofa is ahead." in text and "go_to(target=&#x27;sofa&#x27;)" in text and "001_front_camera.jpg" in text
    assert "sofa 3: a real one is there" in text and "tv 4: no real one there; nearest real thing: sofa" in text
    assert text.index("The sofa is ahead.") < text.index("search sofa") < text.index("001_front_camera.jpg")
    assert "N1/report.html" in report.index(tmp_path, [row]).read_text()
