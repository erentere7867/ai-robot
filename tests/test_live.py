import json
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot.eval.suite import office
from robot.interfaces.demo import Holder, attach_3d
from robot.interfaces.live import PAGE, Live
from robot.sim.world import SimRobot

NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="Node.js is needed to run the dashboard JavaScript")
def test_dashboard_connection_and_command_delivery():
    result = subprocess.run(
        [NODE, str(Path(__file__).with_name("live_dashboard.cjs"))],
        input=PAGE,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def dashboard():
    live = Live()
    server = live.serve(port=0)
    yield live, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def post(url: str, body: bytes) -> tuple[int, dict]:
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


@pytest.mark.parametrize("path", ["/instruct", "/stop", "/scene"])
@pytest.mark.parametrize("body", [b"[]", b'"go"', b"5", b"null", b"{", b"\xff", b'{"text": null}', b'{"text": 5}'])
def test_a_command_that_is_not_a_json_object_with_text_is_refused(dashboard, path, body):
    live, url = dashboard
    status, reply = post(url + path, body)
    assert status == 400
    assert reply["ok"] is False and '{"text":' in reply["error"]
    assert live.inbox.empty()


@pytest.mark.parametrize("path,body,queued", [
    ("/instruct", b'{"text": "go to the kitchen"}', ("instruct", "go to the kitchen")),
    ("/stop", b'{"text": ""}', ("stop", "")),
    ("/stop", b"", ("stop", "")),
    ("/scene", b'{"text": "office"}', ("scene", "office")),
])
def test_a_command_with_text_is_queued(dashboard, path, body, queued):
    live, url = dashboard
    assert post(url + path, body) == (200, {"ok": True})
    assert live.inbox.get_nowait() == queued


@pytest.mark.skipif(NODE is None, reason="Node.js is needed to syntax-check the 3D viewer JS")
def test_viewer3d_js_is_served_and_valid_js(dashboard):
    _, url = dashboard
    with urllib.request.urlopen(f"{url}/viewer3d.js", timeout=5) as response:
        assert response.status == 200
        js = response.read().decode("utf-8")
    assert "sprocket3d" in js
    check = subprocess.run([NODE, "--check"], input=js, text=True, capture_output=True, timeout=10)
    assert check.returncode == 0, check.stderr


def test_scene3d_and_whitelisted_assets(dashboard, tmp_path):
    live, url = dashboard
    with urllib.request.urlopen(f"{url}/scene3d", timeout=5) as response:
        assert json.load(response) == {"mode": "2d"}

    glb = tmp_path / "room.glb"
    glb.write_bytes(b"glTF\x02\x00\x00\x00\x0c\x00\x00\x00")
    live.world3d = lambda: {
        "mode": "3d",
        "cx": 1.0,
        "cy": -0.5,
        "ground_y": 0.0,
        "stage": str(glb),
        "instances": [{"path": str(glb), "mat": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]}],
    }
    with urllib.request.urlopen(f"{url}/scene3d", timeout=5) as response:
        manifest = json.load(response)
    assert manifest["mode"] == "3d"
    assert manifest["stage"].startswith("/assets/")
    assert manifest["instances"][0]["url"] == manifest["stage"]

    with urllib.request.urlopen(f"{url}{manifest['stage']}", timeout=5) as response:
        assert response.status == 200
        assert response.headers["Content-Type"] == "model/gltf-binary"
        assert response.read() == glb.read_bytes()

    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(f"{url}/assets/../../etc/passwd", timeout=5)
    assert err.value.code == 404


def test_show_world_and_attach_3d_include_truth_and_slam_overlay():
    live = Live()
    sim = SimRobot(office())
    sim.walls = np.zeros((10, 10), dtype=bool)
    sim.walls[2, 3] = True
    home = SimpleNamespace(
        truth={"start": (1.5, -2.0, 0.5), "objects": [{"label": "sofa", "x": 2.0, "y": 1.0, "radius": 0.6, "height": 0.8}]},
        path=[(0.0, 1.5, -2.0, 0.5), (1.0, 1.8, -1.7, 0.6)],
        dock=lambda: (1.5, -2.0, 0.5),
        world=SimpleNamespace(call=lambda **kw: {"cx": 0.0, "cy": 0.0, "ground_y": 0.0, "stage": "", "instances": []}),
    )
    holder = Holder(sim, home)
    attach_3d(live, holder)
    live.show_world(sim, home=home)
    assert live.state.true_pose == (1.8, -1.7, 0.6)
    assert live.state.true_trail == [(1.5, -2.0)]
    assert live.state.dock == (1.5, -2.0, 0.5)
    assert live.state.true_objects == [{"label": "sofa", "x": 2.0, "y": 1.0, "r": 0.6, "height": 0.8}]

    payload = live.scene_3d_payload()
    assert payload["mode"] == "3d"
    assert payload["slam"]["png"]
