"""The live page: what the robot is doing, as it does it.

One small web page (standard library only) shows the instruction, the planner's first sentence and plan, the program
with the line now running highlighted, the readback of its dry run, what the robot has said, and a map of the room
with the robot's path. It is the debugging view and the demo screen at once.

    live = Live()
    live.serve(port=8765)          # then open http://<robot>:8765
    agent = Agent(planner, live=live)
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass
class State:
    status: str = "idle"  # idle, planning, checking, running, done, failed, stopped
    instruction: str = ""
    say_first: str = ""
    plan: str = ""
    program: str = ""
    line: int = 0
    readback: str = ""
    problems: list[str] = field(default_factory=list)
    said: list[str] = field(default_factory=list)
    path: list[tuple[float, float]] = field(default_factory=list)
    robot: tuple[float, float, float] = (0.0, 0.0, 0.0)
    things: list[dict[str, Any]] = field(default_factory=list)
    walls: list[tuple[tuple[float, float], tuple[float, float]]] = field(default_factory=list)
    size: tuple[float, float] = (8.0, 6.0)


class Live:
    """Shared state, updated by the agent and the runner, read by the page twice a second."""

    def __init__(self) -> None:
        self.state = State()
        self._lock = threading.Lock()

    def update(self, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(self.state, key, value)

    def snapshot(self) -> str:
        with self._lock:
            return json.dumps(asdict(self.state))

    def show_world(self, sim: Any) -> None:
        """Copy the robot, its path, the things it knows of and the walls from a SimRobot (or RosRobot)."""
        scene = sim.scene
        w, h = scene.width, scene.height
        walls = [((-w / 2, -h / 2), (w / 2, -h / 2)), ((w / 2, -h / 2), (w / 2, h / 2)),
                 ((w / 2, h / 2), (-w / 2, h / 2)), ((-w / 2, h / 2), (-w / 2, -h / 2)), *scene.walls]
        known = set(sim.memory)
        things = [{"label": t.label, "x": t.x, "y": t.y, "r": t.radius, "color": t.color, "known": t.id in known,
                   "person": t.person} for t in sim.things.values()]
        path = [(round(x, 2), round(y, 2)) for _, x, y, _ in sim.trace.poses[::5]]
        self.update(robot=(sim.x, sim.y, sim.yaw), path=path, things=things, walls=walls, size=(w, h),
                    said=[text for _, text in sim.trace.said][-6:])

    def serve(self, port: int = 8765) -> ThreadingHTTPServer:
        live = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - the http.server name
                body, kind = (live.snapshot().encode(), "application/json") if self.path.startswith("/state") \
                    else (PAGE.encode(), "text/html; charset=utf-8")
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:  # quiet
                pass

        server = ThreadingHTTPServer(("0.0.0.0", port), Handler)  # noqa: S104 - the robot's own debug page
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sprocket live</title>
<style>
:root{--bg:#f6f5f2;--panel:#fff;--ink:#1d1d1b;--muted:#6b6a66;--line:#e4e2dc;--accent:#2f6fde;--hi:#fff2c2;
--ok:#1f8a4c;--bad:#c2410c}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--panel:#1d1d1b;--ink:#ecebe7;--muted:#9a988f;--line:#2e2d2a;
--accent:#7aa7ff;--hi:#3a3420;--ok:#5cc98a;--bad:#f08a5d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding:16px;display:grid;gap:16px;grid-template-columns:1fr 1fr}
@media (max-width:860px){main{grid-template-columns:1fr}}
section{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;min-width:0}
h1{font-size:20px;margin:0 0 4px}h2{font-size:13px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);
margin:0 0 8px}.status{display:inline-block;padding:2px 10px;border-radius:99px;background:var(--line);font-size:13px}
.status.running{background:var(--accent);color:#fff}.status.done{background:var(--ok);color:#fff}
.status.failed,.status.stopped{background:var(--bad);color:#fff}
pre{margin:0;font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;overflow-x:auto}
.ln{display:block;padding:0 6px;border-radius:4px;white-space:pre}.ln.now{background:var(--hi)}
.ln i{color:var(--muted);font-style:normal;display:inline-block;width:2.2em}
canvas{width:100%;aspect-ratio:4/3;border-radius:8px;background:var(--bg)}
ul{margin:0;padding-left:18px}.muted{color:var(--muted)}.wide{grid-column:1/-1}
</style></head><body><main>
<section class="wide"><h1 id="instruction">Waiting for an instruction</h1>
<span id="status" class="status">idle</span> <span id="sayfirst" class="muted"></span></section>
<section><h2>Program</h2><div id="plan" class="muted"></div><pre id="program"></pre></section>
<section><h2>Map</h2><canvas id="map" width="800" height="600"></canvas></section>
<section><h2>Readback of the dry run</h2><div id="readback"></div></section>
<section><h2>Said</h2><ul id="said"></ul><h2 style="margin-top:12px">Problems</h2><ul id="problems"></ul></section>
</main><script>
const $=id=>document.getElementById(id);
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function draw(s){const c=$('map'),g=c.getContext('2d'),[W,H]=s.size,k=Math.min(c.width/W,c.height/H)*.92;
const css=getComputedStyle(document.documentElement),ink=css.getPropertyValue('--ink'),mut=css.getPropertyValue('--muted'),
acc=css.getPropertyValue('--accent');const X=x=>c.width/2+x*k,Y=y=>c.height/2-y*k;g.clearRect(0,0,c.width,c.height);
g.strokeStyle=ink;g.lineWidth=3;for(const[a,b]of s.walls){g.beginPath();g.moveTo(X(a[0]),Y(a[1]));g.lineTo(X(b[0]),Y(b[1]));g.stroke()}
for(const t of s.things){g.beginPath();g.arc(X(t.x),Y(t.y),Math.max(3,t.r*k),0,7);g.fillStyle=t.person?acc:(t.color||mut);
g.globalAlpha=t.known?.85:.25;g.fill();g.globalAlpha=1;g.fillStyle=ink;g.font='13px system-ui';
g.fillText(t.person?'person':t.label,X(t.x)+t.r*k+3,Y(t.y)+4)}
g.strokeStyle=acc;g.lineWidth=2;g.beginPath();s.path.forEach(([x,y],i)=>i?g.lineTo(X(x),Y(y)):g.moveTo(X(x),Y(y)));g.stroke();
const[x,y,a]=s.robot;g.beginPath();g.arc(X(x),Y(y),.15*k,0,7);g.fillStyle=acc;g.fill();g.beginPath();g.moveTo(X(x),Y(y));
g.lineTo(X(x+.3*Math.cos(a)),Y(y+.3*Math.sin(a)));g.strokeStyle=ink;g.stroke()}
async function tick(){try{const s=await(await fetch('/state',{cache:'no-store'})).json();
$('instruction').textContent=s.instruction||'Waiting for an instruction';$('status').textContent=s.status;
$('status').className='status '+s.status;$('sayfirst').textContent=s.say_first?'“'+s.say_first+'”':'';
$('plan').textContent=s.plan;$('program').innerHTML=s.program.split('\\n').filter((l,i,a)=>i<a.length-1||l)
.map((l,i)=>`<span class="ln${i+1===s.line?' now':''}"><i>${i+1}</i>${esc(l)}</span>`).join('');
$('readback').textContent=s.readback||'—';$('said').innerHTML=s.said.map(t=>`<li>${esc(t)}</li>`).join('')||'<li class="muted">nothing yet</li>';
$('problems').innerHTML=s.problems.map(t=>`<li>${esc(t)}</li>`).join('')||'<li class="muted">none</li>';draw(s)}catch(e){}}
setInterval(tick,500);tick();
</script></body></html>"""
