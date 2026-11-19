"""Starting the Habitat world (robot/sim/habitat_server.py, in Habitat's own Python) and talking to it.

The hardware node (robot/sim/habitat_ros.py) drives the body and reads the sensors through one connection; the judge
(robot/eval/home3d.py) reads the truth through another.
"""

from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

HABITAT_PYTHON = os.environ.get("SPROCKET_HABITAT_PYTHON", str(Path("~/micromamba/envs/habitat/bin/python").expanduser()))
SERVER = Path(__file__).with_name("habitat_server.py")
AUTH = b"sprocket"


def start(scene: str) -> tuple[subprocess.Popen, int]:
    """Load a home in a new Habitat process; returns it and the port it answers on (loading takes a few seconds)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(os.environ)
    if Path("/dev/dxg").exists():  # WSL: OpenGL on the GPU through Direct3D 12; Mesa draws on the CPU otherwise
        env.setdefault("GALLIUM_DRIVER", "d3d12")
    proc = subprocess.Popen([HABITAT_PYTHON, str(SERVER), "--port", str(port), "--scene", scene],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, text=True, cwd="/tmp")
    for line in proc.stdout or ():  # Habitat prints its renderer details first
        if line.strip() == "ready":
            break
    else:
        raise RuntimeError(f"the Habitat server did not start ({HABITAT_PYTHON}); is the habitat env installed?")
    proc.stdout.close()  # nothing more is read; an unread pipe could fill and block the server
    return proc, port


class Habitat:
    """One connection to a running Habitat server."""

    def __init__(self, port: int) -> None:
        self.lock = threading.Lock()
        for _ in range(100):
            try:
                self.conn = Client(("127.0.0.1", port), authkey=AUTH)
                break
            except ConnectionRefusedError:
                time.sleep(0.1)
        else:
            raise ConnectionRefusedError(f"no Habitat server on port {port}")

    def call(self, **request: Any) -> Any:
        with self.lock:
            self.conn.send(request)
            reply = self.conn.recv()
        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    def close(self) -> None:
        self.conn.close()
