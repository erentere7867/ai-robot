"""The GPU vision path without a GPU: the server's decoding of YOLOE's raw output, the detector's word slots, masks and
feet, and the tools that point at a floor spot or look back."""

from __future__ import annotations

import importlib.util
import threading
from contextlib import closing
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from robot.agent.agent import Agent
from robot.eval.suite import living_room
from robot.perception.detector import _box_mask, _foot
from robot.perception.gpu import VisionServer
from robot.sim.world import SimRobot
from tests.perception_fakes import make_detector
from tests.test_agent import Scripted, call, say, tool_results

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("vision_server", ROOT / "tools" / "vision_server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


def test_streamed_pixels_round_trip_through_the_real_http_handler():
    models = Mock(slots=8, times={})
    models.yoloe.get_providers.return_value = []
    models.run_depth.side_effect = lambda rgb: {"inverse": rgb[..., 0].astype(np.float16)}
    with ThreadingHTTPServer(("127.0.0.1", 0), server.handler(models)) as http:
        threading.Thread(target=http.serve_forever, daemon=True).start()
        try:
            client = VisionServer(f"http://127.0.0.1:{http.server_port}")
            with closing(client.http):
                image = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
                assert np.array_equal(client.depth(image), image[..., 0])
                models.run_depth.side_effect = ValueError("bad depth input")
                with pytest.raises(RuntimeError, match="bad depth input"):
                    client.depth(image)
        finally:
            http.shutdown()


def _raw(boxes: list[tuple[float, float, float, float]], scores: list[list[float]], slots: int = 3):
    """YOLOE's raw output for some boxes (frame pixels): (4 + slots + 32, boxes), and prototypes that are all 'inside'."""
    pred = np.zeros((4 + slots + 32, len(boxes)), np.float32)
    for k, (x1, y1, x2, y2) in enumerate(boxes):
        pred[:4, k] = ((x1 + x2) / 2, (y1 + y2) / 2 + server.PAD, x2 - x1, y2 - y1)
        pred[4:4 + slots, k] = scores[k]
        pred[4 + slots, k] = 5.0  # one mask coefficient, on a prototype that is 1 everywhere
    protos = np.zeros((32, 184, 320), np.float32)
    protos[0] = 1.0
    return pred, protos


def test_each_real_word_keeps_its_own_boxes_and_padding_slots_are_ignored():
    pred, protos = _raw([(100, 100, 200, 300), (104, 102, 204, 298), (600, 50, 700, 150)],
                        [[0.9, 0.3, 0.95], [0.8, 0.0, 0.95], [0.1, 0.6, 0.95]])
    out = server.decode(pred, protos, n=2, conf=0.22)  # slot 3 is padding: its 0.95s must not appear
    got = sorted((int(w), round(float(s), 2)) for w, s in zip(out["word"], out["scores"], strict=True))
    assert got == [(0, 0.9), (1, 0.3), (1, 0.6)]  # word 0: the overlapping pair suppressed to one; word 1: both boxes
    assert len(out["mask_shape"]) == 3 and out["mask"].size == int(np.prod(out["mask_shape"], axis=1).sum())


class FakeServer:
    slots = 4

    def __init__(self, box=(100.0, 100.0, 200.0, 300.0)) -> None:
        self.box, self.calls = box, []

    def detect(self, rgb, words, n, conf):  # noqa: ANN001, ANN201
        self.calls.append((words.shape, n))
        pred, protos = _raw([self.box], [[0.0, 0.7, 0.0, 0.0]], slots=self.slots)
        return server.decode(pred, protos, n, conf)


def test_the_newest_words_fill_the_slots_and_a_box_takes_its_word_and_feet():
    fake = FakeServer()
    det = make_detector(fake, ("ball", "cup", "keys", "cat"))
    det.watch("ball", "cup", "keys", "cat")  # 4 slots, one kept for people: the oldest asked word gives way
    assert det.vocabulary == ["person", "cup", "keys", "cat"]
    det.watch("ball")
    assert det.vocabulary == ["person", "keys", "cat", "ball"]
    seen = det(np.zeros((720, 1280, 3), np.uint8))
    assert fake.calls[-1] == ((1, 4, 512), 4)
    assert [s.label for s in seen] == ["keys"]  # the box scored for word 1
    assert seen[0].mask.shape == (200, 100) and seen[0].mask.all()
    assert abs(seen[0].foot[0] - 150.0) < 1.0 and abs(seen[0].foot[1] - 300.0) < 1.0  # bottom middle of the box


def test_a_mask_is_cut_to_its_box_and_its_foot_is_its_lowest_middle():
    patch = np.zeros((10, 10), np.float32)
    patch[2:8, 3:6] = 1.0  # quarter resolution, covering frame pixels 0..40
    mask = _box_mask(patch, (0, 0, 40, 40), (8.0, 4.0, 32.0, 36.0))
    assert mask.shape == (32, 24)
    foot = _foot(mask, (8.0, 4.0, 32.0, 36.0))
    assert 16 <= foot[0] <= 20 and 30 <= foot[1] <= 33


class SpotRobot(SimRobot):
    """The 2D simulator with pictures from before, as RealRobot keeps them."""

    def past_views(self, seconds: float, count: int = 4):  # noqa: ANN201
        return [(4.0, np.zeros((288, 512, 3), np.uint8)), (0.0, np.zeros((288, 512, 3), np.uint8))]


def test_a_spot_described_as_a_relation_is_sent_to_go_toward():
    sim = SimRobot(living_room())
    brain = Scripted(call("go_to", target="between the sofa and the table"), say("Hm."), say("Hm."))
    Agent(brain, cameras="none").do("Go between the sofa and the table.", sim)
    assert "use go_toward" in tool_results(brain.seen[1])[0]


def test_look_back_shows_the_pictures_from_before_with_how_long_ago():
    sim = SpotRobot(living_room())
    brain = Scripted(call("look_back", seconds=5), say("The cat left."), say("The cat left."))
    Agent(brain, cameras="none").do("Did the cat just leave?", sim)
    texts = [p.get("text", "") for p in brain.seen[1][-1]["content"]]
    assert "front camera 4 s ago:" in texts and "front camera 0 s ago:" in texts



def test_go_toward_drives_that_way_and_stops_short_of_what_is_in_the_way():
    sim = SimRobot(living_room())
    start = sim.pose()
    brain = Scripted(call("go_toward", direction="left", metres=1.0), say("Done."), say("Done."))
    Agent(brain, cameras="none").do("Go a metre to your left.", sim)
    assert tool_results(brain.seen[1])[0].startswith(("Drove 0.9 m left", "Drove 1.0 m left"))
    end = sim.pose()
    import math
    left = (end.x - start.x) * -math.sin(math.radians(start.yaw)) + (end.y - start.y) * math.cos(math.radians(start.yaw))
    assert 0.8 < left < 1.2
    far = sim.toward("front", 50.0)  # through the wall: the last free floor before it
    assert not sim._occ(*far, sim.inflated)


def test_search_can_start_beside_a_thing_the_model_names():
    sim = SimRobot(living_room())
    near = sorted(sim.things.values(), key=lambda t: -t.radius)[0]
    brain = Scripted(call("search", what="ball", near=near.label), say("Found."), say("Found."))
    Agent(brain, cameras="none").do("Look for the ball by the big thing.", sim)
    names = [e.name for e in sim.trace.events]
    assert names.index("go_to") < names.index("search")  # beside it first, then the search


def test_a_flickering_thing_is_not_gone_until_unseen_for_a_while():
    from robot.perception.tracker import GONE_AFTER, GONE_S, Detection, Tracker

    tr = Tracker()
    tr.update([Detection("plant", 1.0, 0.0, 0.3)], 9.5)
    tr.update([Detection("plant", 1.0, 0.0, 0.3)], 10.0)  # seen twice: a thing, not a one-frame blip
    plant = tr.objects()[0]
    # two cameras missing it for a few frames, a second after it was seen
    assert not any([tr.missed(plant, 11.0) for _ in range(GONE_AFTER + 2)])
    assert tr.missed(plant, 10.0 + GONE_S + 0.1)


def test_depth_is_normalised_into_the_kept_buffer_as_before():
    models = server.Models.__new__(server.Models)
    models.gpu, models.times, models.depth_input = threading.Lock(), {"depth": []}, "pixel_values"
    models._depth_buf = np.empty((1, 3, 364, 644), np.float32)
    seen = []
    models.depth = Mock(run=lambda _, feeds: seen.append(feeds["pixel_values"].copy()) or [np.zeros((1, 364, 644))])
    pictures = np.random.default_rng(0).integers(0, 256, (2, 364, 644, 3), dtype=np.uint8)
    for picture in pictures:
        models.run_depth(picture)
    for picture, fed in zip(pictures, seen, strict=True):
        normalised = (picture * server._DEPTH_SCALE + server._DEPTH_BIAS).transpose(2, 0, 1)[None]
        assert np.allclose(fed, normalised, atol=1e-5)
    with pytest.raises(ValueError):
        models.run_depth(pictures[0, :1])
