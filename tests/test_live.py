import json
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from robot.interfaces.live import PAGE, Live

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
