"""The live page: what the robot is doing, as it does it, and a box to tell it what to do.

One small web page (standard library only) shows the instruction, the planner's first sentence and plan, the program
with the line now running highlighted, the readback of its dry run, what the robot has said, a map of the room with
the robot's path, and the history of instructions. Instructions typed on the page, the Stop button and the room picker
arrive in `Live.inbox` for whoever drives the robot (robot/interfaces/demo.py).

    live = Live()
    live.serve(port=8765)          # then open http://localhost:8765
    agent = Agent(planner, live=live)

It listens on this computer only by default: the page can command the robot.
"""

from __future__ import annotations

import json
import queue
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
    rooms: list[str] = field(default_factory=list)  # room names in this scene
    scene: str = ""
    scenes: list[str] = field(default_factory=list)  # what the room picker offers
    history: list[dict[str, Any]] = field(default_factory=list)  # past instructions and how they went
    cameras: list[str] = field(default_factory=list)  # camera images served at /camera/<name>.png
    scan: list[tuple[float, float]] = field(default_factory=list)  # lidar hits on the map


class Live:
    """Shared state, updated by the agent and the runner, read by the page twice a second."""

    def __init__(self) -> None:
        self.state = State()
        self.inbox: queue.Queue[tuple[str, str]] = queue.Queue()  # ("instruct", text), ("stop", ""), ("scene", name)
        self.camera: Any = None  # name -> PNG bytes or None, for the camera panel
        self._lock = threading.Lock()

    def update(self, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(self.state, key, value)

    def snapshot(self) -> str:
        with self._lock:
            return json.dumps(asdict(self.state))

    def show_world(self, sim: Any) -> None:
        """Copy the robot, its path, the things in the room and the walls from a SimRobot (or RosRobot)."""
        scene = sim.scene
        w, h = scene.width, scene.height
        walls = [((-w / 2, -h / 2), (w / 2, -h / 2)), ((w / 2, -h / 2), (w / 2, h / 2)),
                 ((w / 2, h / 2), (-w / 2, h / 2)), ((-w / 2, h / 2), (-w / 2, -h / 2)), *scene.walls]
        known = set(sim.memory)
        things = []
        for t in list(sim.things.values()):
            p = t.at(sim.t)
            if p is not None:
                things.append({"label": t.label, "x": p[0], "y": p[1], "r": t.radius, "color": t.color,
                               "known": t.id in known, "person": t.person, "id": t.id})
        path = [(round(x, 2), round(y, 2)) for _, x, y, _ in sim.trace.poses[-1500::5]]
        self.update(robot=(sim.x, sim.y, sim.yaw), path=path, things=things, walls=walls, size=(w, h),
                    said=[text for _, text in sim.trace.said][-6:], rooms=list(scene.rooms))

    def serve(self, port: int = 8765, host: str = "127.0.0.1") -> ThreadingHTTPServer:
        live = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, body: bytes, kind: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 - the http.server name
                if self.path.startswith("/camera/") and live.camera is not None:
                    png = live.camera(self.path.split("/")[2].split(".")[0])
                    if png:
                        self._send(png, "image/png")
                        return
                if self.path.startswith("/state"):
                    self._send(live.snapshot().encode(), "application/json")
                else:
                    self._send(PAGE.encode(), "text/html; charset=utf-8")

            def do_POST(self) -> None:  # noqa: N802
                length = min(int(self.headers.get("Content-Length") or 0), 4096)
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError:
                    body = {}
                kind = self.path.strip("/")
                if kind in ("instruct", "stop", "scene"):
                    live.inbox.put((kind, str(body.get("text", ""))[:300]))
                self._send(b"{}", "application/json")

            def log_message(self, *args: Any) -> None:  # quiet
                pass

        server = ThreadingHTTPServer((host, port), Handler)
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
.status.running,.status.planning,.status.checking{background:var(--accent);color:#fff}
.status.done{background:var(--ok);color:#fff}.status.failed,.status.stopped{background:var(--bad);color:#fff}
pre{margin:0;font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;overflow-x:auto}
.ln{display:block;padding:0 6px;border-radius:4px;white-space:pre}.ln.now{background:var(--hi)}
.ln i{color:var(--muted);font-style:normal;display:inline-block;width:2.2em}
canvas{width:100%;aspect-ratio:4/3;border-radius:8px;background:var(--bg)}
ul{margin:0;padding-left:18px}.muted{color:var(--muted)}.wide{grid-column:1/-1}
form{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
input,select,button{font:inherit;border-radius:8px;border:1px solid var(--line);padding:8px 12px;background:var(--bg);
color:var(--ink)}input{flex:1;min-width:200px}button{cursor:pointer;background:var(--accent);color:#fff;border:0}
button.stop{background:var(--bad)}.cams{display:grid;gap:8px;grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}
.cams figure{margin:0}.cams img{width:100%;border-radius:8px;background:var(--bg)}.cams figcaption{font-size:13px;color:var(--muted)}.hint{font-size:13px;color:var(--muted);margin-top:6px}
</style></head><body><main>
<section class="wide"><h1 id="instruction">Tell the robot what to do</h1>
<span id="status" class="status">idle</span> <span id="sayfirst" class="muted"></span>
<form id="form"><input id="cmd" placeholder="e.g. Circle the table twice, then come back to me" autocomplete="off">
<button type="submit">Send</button><button type="button" class="stop" id="stop">Stop</button>
<select id="scene" title="Room"></select></form>
<div class="hint" id="hint">The robot is the blue dot with a line showing where it faces; you are the blue
person. Faded things have not been seen yet.</div></section>
<section><h2>Program</h2><div id="plan" class="muted"></div><pre id="program"></pre></section>
<section><h2>Map</h2><canvas id="map" width="800" height="600"></canvas></section>
<section><h2>What it will do (from the dry run)</h2><div id="readback"></div>
<h2 style="margin-top:12px">Problems</h2><ul id="problems"></ul></section>
<section><h2>Said</h2><ul id="said"></ul><h2 style="margin-top:12px">History</h2><ul id="history"></ul></section>
<section class="wide" id="camsec" hidden><h2>What the cameras see</h2><div id="cams" class="cams"></div></section>
</main><script>
const $=id=>document.getElementById(id);
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
const post=(p,text)=>fetch(p,{method:'POST',body:JSON.stringify({text})});
$('form').onsubmit=e=>{e.preventDefault();const t=$('cmd').value.trim();if(t){post('/instruct',t);$('cmd').value=''}};
$('stop').onclick=()=>post('/stop','');
$('scene').onchange=()=>post('/scene',$('scene').value);
function draw(s){const c=$('map'),g=c.getContext('2d'),[W,H]=s.size,k=Math.min(c.width/W,c.height/H)*.92;
const css=getComputedStyle(document.documentElement),ink=css.getPropertyValue('--ink'),mut=css.getPropertyValue('--muted'),
acc=css.getPropertyValue('--accent');const X=x=>c.width/2+x*k,Y=y=>c.height/2-y*k;g.clearRect(0,0,c.width,c.height);
g.strokeStyle=ink;g.lineWidth=3;for(const[a,b]of s.walls){g.beginPath();g.moveTo(X(a[0]),Y(a[1]));g.lineTo(X(b[0]),Y(b[1]));g.stroke()}
for(const t of s.things){g.beginPath();g.arc(X(t.x),Y(t.y),Math.max(3,t.r*k),0,7);g.fillStyle=t.person?acc:(t.color||mut);
g.globalAlpha=t.known?.85:.25;g.fill();g.globalAlpha=1;g.strokeStyle=mut;g.lineWidth=1;g.stroke();g.fillStyle=ink;
g.font='13px system-ui';g.fillText((t.person?'you':t.label)+(t.person?'':' '+t.id),X(t.x)+t.r*k+3,Y(t.y)+4)}
g.fillStyle='#e0457b';for(const[x,y]of s.scan){g.fillRect(X(x)-1.5,Y(y)-1.5,3,3)}
g.strokeStyle=acc;g.lineWidth=2;g.beginPath();s.path.forEach(([x,y],i)=>i?g.lineTo(X(x),Y(y)):g.moveTo(X(x),Y(y)));g.stroke();
const[x,y,a]=s.robot;g.beginPath();g.arc(X(x),Y(y),.15*k,0,7);g.fillStyle=acc;g.fill();g.beginPath();g.moveTo(X(x),Y(y));
g.lineTo(X(x+.35*Math.cos(a)),Y(y+.35*Math.sin(a)));g.strokeStyle=ink;g.lineWidth=3;g.stroke()}
let scenes='';
async function tick(){try{const s=await(await fetch('/state',{cache:'no-store'})).json();
$('instruction').textContent=s.instruction||'Tell the robot what to do';$('status').textContent=s.status;
$('status').className='status '+s.status;$('sayfirst').textContent=s.say_first?'“'+s.say_first+'”':'';
$('plan').textContent=s.plan;$('program').innerHTML=s.program.split('\\n').filter((l,i,a)=>i<a.length-1||l)
.map((l,i)=>`<span class="ln${i+1===s.line?' now':''}"><i>${i+1}</i>${esc(l)}</span>`).join('');
$('readback').textContent=s.readback||'—';$('said').innerHTML=s.said.map(t=>`<li>${esc(t)}</li>`).join('')||'<li class="muted">nothing yet</li>';
$('problems').innerHTML=s.problems.map(t=>`<li>${esc(t)}</li>`).join('')||'<li class="muted">none</li>';
$('history').innerHTML=s.history.slice().reverse().map(h=>`<li><b>${esc(h.outcome)}</b> ${esc(h.instruction)}</li>`).join('')||'<li class="muted">nothing yet</li>';
const key=s.scenes.join('|')+'#'+s.scene;if(key!==scenes){scenes=key;$('scene').innerHTML=s.scenes.map(n=>
`<option${n===s.scene?' selected':''}>${esc(n)}</option>`).join('')}draw(s)}catch(e){}}
let camtick=0;function cams(s){if(!s.cameras.length)return;$('camsec').hidden=false;if(camtick++%3)return;
$('cams').innerHTML=s.cameras.map(n=>`<figure><img src="/camera/${n}.png?t=${Date.now()}" alt="${n}">
<figcaption>${n.replace(/_/g,' ')}</figcaption></figure>`).join('')}
setInterval(async()=>{try{cams(await(await fetch('/state',{cache:'no-store'})).json())}catch(e){}},1000);
setInterval(tick,400);tick();
</script></body></html>"""
