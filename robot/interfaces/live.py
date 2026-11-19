"""The live page: what Sprocket v2 is thinking and doing, and a box to command it.

One small web page (standard library only) shows the instruction, the agent's to-do list and every tool call
with its result, the model's reasoning, a 2D map with the robot's pose, trail, objects and rooms,
what the robot has said, and the history of instructions. Commands typed on the
page, the Stop button, and the room picker arrive in `Live.inbox` for whoever drives the robot
(`robot/interfaces/demo.py`).

    live = Live()
    live.serve(port=8765)          # then open http://localhost:8765
    agent = Agent(brain, live=live)

It listens on this computer only by default: the page can command the robot.
"""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


@dataclass
class State:
    status: str = "idle"  # idle, thinking, acting, done, failed, stopped
    instruction: str = ""
    posture: str = "neutral"
    chime: str = ""
    steps: list[str] = field(default_factory=list)  # every tool call and its result, in order
    thought: str = ""  # the model's latest reasoning, in full
    searching: list[str] = field(default_factory=list)  # the last search's stops: where the model thought it was
    todo: str = ""  # the model's to-do list
    problems: list[str] = field(default_factory=list)
    said: list[str] = field(default_factory=list)
    pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    poses: list[tuple[float, float, float]] = field(default_factory=list)
    trail: list[tuple[float, float]] = field(default_factory=list)
    objects: list[dict[str, Any]] = field(default_factory=list)
    walls: list[tuple[tuple[float, float], tuple[float, float]]] = field(default_factory=list)
    size: tuple[float, float] = (8.0, 6.0)
    rooms: list[str] = field(default_factory=list)
    room_bounds: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    scene: str = ""
    scenes: list[str] = field(default_factory=list)
    scene_loading: str = ""  # the room picked on the page while it loads; `scene` stays the room the robot is in
    history: list[dict[str, Any]] = field(default_factory=list)
    cameras: list[str] = field(default_factory=list)
    voice: bool = False
    heard: str = ""
    said_total: int = 0


class Live:
    """Shared Sprocket v2 state, updated by the agent/driver and polled by the live web UI."""

    def __init__(self) -> None:
        self.state = State()
        self.inbox: queue.Queue[tuple[str, str]] = queue.Queue()  # ("instruct", text), ("stop", ""), ("scene", name)
        self.camera: Any = None  # name -> PNG bytes or None, for multi-camera panel
        self.files: Any = None  # a folder served at /runs/ (the 3D eval's reports), or None
        self.ears: Any = None
        self.mouth: Any = None
        self._lock = threading.Lock()

    def update(self, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                if not hasattr(self.state, key):
                    continue
                setattr(self.state, key, value)

    def snapshot(self) -> str:
        with self._lock:
            return json.dumps(asdict(self.state), default=str)

    def show_world(self, sim: Any) -> None:
        """Copy the robot pose, trail, objects, walls, and speech from a SimRobot (or RosRobot)."""
        scene = sim.scene
        w, h = scene.width, scene.height
        walls = [
            ((-w / 2, -h / 2), (w / 2, -h / 2)),
            ((w / 2, -h / 2), (w / 2, h / 2)),
            ((w / 2, h / 2), (-w / 2, h / 2)),
            ((-w / 2, h / 2), (-w / 2, -h / 2)),
            *scene.walls,
        ]
        known = set(sim.memory)
        objects = []
        for t in list(sim.things.values()):
            p = t.at(sim.t)
            if p is not None:
                objects.append({
                    "label": t.label,
                    "x": p[0],
                    "y": p[1],
                    "r": t.radius,
                    "color": t.color,
                    "known": t.id in known,
                    "person": t.person,
                    "user": bool(getattr(t, "user", False)),
                    "id": t.id,
                })
        sampled = sim.trace.poses[-1500::5]
        trail = [(round(x, 2), round(y, 2)) for _, x, y, _ in sampled]
        poses = [(round(x, 2), round(y, 2), round(yaw, 3)) for _, x, y, yaw in sampled[-120:]]
        room_bounds = (
            {
                str(name): tuple(round(float(v), 3) for v in rect)
                for name, rect in scene.rooms.items()
                if isinstance(rect, (tuple, list)) and len(rect) == 4
            }
            if isinstance(scene.rooms, dict)
            else {}
        )
        changes: dict[str, Any] = {
            "pose": (round(sim.x, 3), round(sim.y, 3), round(sim.yaw, 4)),
            "poses": poses,
            "trail": trail,
            "objects": objects,
            "walls": walls,
            "size": (w, h),
            "said": [text for _, text in sim.trace.said][-6:],
            "said_total": len(sim.trace.said),
            "rooms": list(scene.rooms),
            "room_bounds": room_bounds,
        }
        if hasattr(sim, "posture"):
            changes["posture"] = sim.posture
        if hasattr(sim, "search_log"):  # a search running now: where the model thinks the thing is, stop by stop
            changes["searching"] = list(sim.search_log)
        self.update(**changes)

    def hear(self, audio: bytes) -> None:
        """Speech from the talk button: transcribed, then handled like a typed instruction."""
        try:
            text = self.ears.hear(audio)
        except Exception as exc:  # noqa: BLE001 - a bad recording must not stop the page
            text = ""
            self.update(heard=f"(could not hear: {exc})")
        if text:
            self.update(heard=text)
            self.inbox.put(("instruct", text[:300]))

    def serve(self, port: int = 8765, host: str = "127.0.0.1") -> ThreadingHTTPServer:
        live = self

        class _LiveHTTPServer(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True
            request_queue_size = 64

        class Handler(BaseHTTPRequestHandler):
            timeout = 5.0

            def _send(self, body: bytes, kind: str, status: int = 200) -> None:
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", kind)
                    if status != 204:
                        self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    if body and status != 204:
                        self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                    return

            def _read_body(self, limit: int) -> bytes:
                try:
                    length = min(max(0, int(self.headers.get("Content-Length") or 0)), limit)
                    return self.rfile.read(length) if length else b""
                except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError, ValueError):
                    return b""

            def do_GET(self) -> None:  # noqa: N802 - http.server method name
                if self.path.startswith("/runs/") and live.files is not None:
                    base = Path(live.files).resolve()
                    want = (base / urlparse(self.path).path.removeprefix("/runs/")).resolve()
                    if base in want.parents and want.is_file():
                        kind = {".html": "text/html; charset=utf-8", ".png": "image/png",
                                ".jpg": "image/jpeg"}.get(want.suffix, "application/octet-stream")
                        return self._send(want.read_bytes(), kind)
                    return self._send(b"not found", "text/plain", 404)
                if self.path.startswith("/camera/"):
                    png = None
                    if live.camera is not None:
                        try:
                            cam_name = self.path.split("/")[2].split("?")[0].split(".")[0]
                            png = live.camera(cam_name)
                        except Exception:  # noqa: BLE001 - a camera callback error must not serve HTML
                            png = None
                    if png:
                        self._send(png, "image/png")
                    else:
                        self._send(b"", "image/png", status=204)
                    return
                if self.path.startswith("/speech?") and live.mouth is not None:
                    text = parse_qs(urlparse(self.path).query).get("text", [""])[0][:300]
                    if text:
                        self._send(live.mouth.say(text), "audio/wav")
                        return
                if self.path.startswith("/state"):
                    self._send(live.snapshot().encode("utf-8"), "application/json")
                else:
                    self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")

            def do_POST(self) -> None:  # noqa: N802
                if self.path.startswith("/voice") and live.ears is not None:
                    audio = self._read_body(8_000_000)
                    if audio:
                        threading.Thread(target=live.hear, args=(audio,), daemon=True).start()
                    self._send(b'{"ok":true}', "application/json")
                    return
                raw = self._read_body(4096)
                kind = self.path.split("?")[0].strip("/")
                if kind in ("instruct", "stop", "scene"):
                    text = command_text(raw)
                    if text is None:
                        self._send(REFUSED, "application/json", 400)
                        return
                    if kind == "scene" and text in live.state.scenes:
                        live.update(scene_loading=text)  # before replying, so the page's next poll sees the pick
                    live.inbox.put((kind, text))
                self._send(b'{"ok":true}', "application/json")

            def log_message(self, *args: Any) -> None:  # quiet
                pass

        server = _LiveHTTPServer((host, port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server


REFUSED = json.dumps({"ok": False, "error": 'Send a JSON object like {"text": "go to the kitchen"}.'}).encode()


def command_text(raw: bytes) -> str | None:
    """The text of a posted command, or None when the body is not a JSON object whose text is a string."""
    try:
        body = json.loads(raw or b"{}")
    except ValueError:  # malformed JSON, or bytes that are not UTF-8
        return None
    text = body.get("text", "") if isinstance(body, dict) else None
    return text[:300] if isinstance(text, str) else None


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sprocket v2 live</title>
<style>
:root{--bg:#f6f5f2;--panel:#fff;--ink:#1d1d1b;--muted:#6b6a66;--line:#e4e2dc;--accent:#2f6fde;--hi:#fff2c2;
--ok:#1f8a4c;--bad:#c2410c;--warn:#9a6700}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--panel:#1d1d1b;--ink:#ecebe7;--muted:#9a988f;--line:#2e2d2a;
--accent:#7aa7ff;--hi:#3a3420;--ok:#5cc98a;--bad:#f08a5d;--warn:#e3b341}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main{max-width:1200px;margin:0 auto;padding:16px;display:grid;gap:16px;grid-template-columns:1fr 1fr}
@media (max-width:860px){main{grid-template-columns:1fr}}
section{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;min-width:0}
h1{font-size:20px;margin:0 0 6px}h2{font-size:12px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);
margin:0 0 8px}h3{font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin:12px 0 6px}
.badges{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.status,.badge{display:inline-block;padding:2px 10px;border-radius:99px;background:var(--line);font-size:13px}
.status.running,.status.planning{background:var(--accent);color:#fff}
.status.done{background:var(--ok);color:#fff}.status.failed,.status.stopped{background:var(--bad);color:#fff}
.status.refused{background:var(--warn);color:#fff}
.status.stale,.send-error:not(:empty){background:color-mix(in srgb,var(--bad) 8%,var(--panel));color:var(--bad)}
.send-error:not(:empty){margin-top:8px;padding:8px 12px;border-left:3px solid var(--bad);border-radius:8px;font-size:14px}
pre{margin:0;padding:8px 10px;border-radius:8px;background:var(--bg);font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
overflow-x:auto;white-space:pre-wrap}
.steps{margin:0;padding-left:20px;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
.steps li{padding:3px 6px;border-radius:6px;margin-bottom:4px}.steps li.now{background:var(--hi)}
.steps li.thought{color:var(--muted);font-style:italic}
canvas{width:100%;aspect-ratio:4/3;border-radius:8px;background:var(--bg)}
ul{margin:0;padding-left:18px}.muted{color:var(--muted)}.wide{grid-column:1/-1}
.mind-kv{display:flex;gap:12px;flex-wrap:wrap;font-size:14px;margin-bottom:6px}
form{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
input,select,button{font:inherit;border-radius:8px;border:1px solid var(--line);padding:8px 12px;background:var(--bg);
color:var(--ink)}input{flex:1;min-width:200px}button{cursor:pointer;background:var(--accent);color:#fff;border:0}
#send:disabled{opacity:.65;cursor:wait}
button.stop{background:var(--bad)}.cams{display:grid;gap:8px;grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}
.cams figure{margin:0}.cams img{width:100%;border-radius:8px;background:var(--bg)}.cams figcaption{font-size:13px;color:var(--muted)}
.hint{font-size:13px;color:var(--muted);margin-top:6px}
.reply{display:flex;align-items:center;gap:10px;margin-top:10px;padding:10px 14px;background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:8px;font-size:15px;color:var(--ink)}
.reply-tag{display:inline-block;padding:2px 8px;border-radius:4px;background:var(--accent);color:#fff;font-size:11px;font-weight:700;letter-spacing:.05em}
.reply[hidden],.badge[hidden]{display:none}
#reply-text{font-weight:500;line-height:1.4}
</style></head><body><main>
<section class="wide">
<h1 id="instruction">Tell the robot what to do</h1>
<div class="badges">
<span id="status" class="status" role="status">Connecting…</span>
<span id="state-stale" class="muted" hidden>Displayed robot data may be out of date.</span>
<span id="posture" class="badge">posture: neutral</span>
<span id="chime" class="badge" hidden></span>
<span id="rooms" class="muted"></span>
</div>
<form id="form">
<input id="cmd" placeholder="e.g. Circle the table twice, then come back to me" autocomplete="off"
 aria-label="Robot command" aria-describedby="send-error">
<button type="submit" id="send">Send</button>
<button type="button" class="stop" id="stop">Stop</button>
<button type="button" id="talk" hidden title="Hold and speak">Hold to talk</button>
<label id="speaklabel" class="muted" hidden><input type="checkbox" id="speak" checked onchange="toggleVoice()" style="min-width:0;flex:0"> speak aloud</label>
<select id="scene" title="Room"></select>
</form>
<div id="send-error" class="send-error" role="alert"></div>
<div class="reply" id="reply" hidden><span class="reply-tag">SPROCKET</span><span id="reply-text"></span></div>
<div class="hint" id="hint">The robot is the blue dot with a heading indicator; you are the blue person.
Faded objects are remembered.</div>
<div class="hint" id="heard"></div>
</section>

<section>
<h2>Agent</h2>
<h3>To-do</h3>
<pre id="todo" class="muted">(no to-do list)</pre>
<h3>Tool calls</h3>
<ol id="steps" class="steps"></ol>
<h3>Latest reasoning</h3>
<pre id="thought" class="muted" style="white-space:pre-wrap">(none)</pre>
<h3>Last search: where the model looked</h3>
<ol id="searching" class="steps"></ol>
</section>

<section>
<h2>Map</h2>
<canvas id="map" width="800" height="600"></canvas>
</section>


<section>
<h2>Said</h2>
<ul id="said"></ul>
<h3>Problems</h3>
<ul id="problems"></ul>
</section>

<section>
<h2>History</h2>
<ul id="history"></ul>
</section>

<section class="wide" id="camsec" hidden>
<h2>Live cameras</h2>
<div id="cams" class="cams"></div>
</section>
</main><script>
const $=id=>document.getElementById(id);
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
const post=(p,text,signal)=>fetch(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text}),signal});
$('form').onsubmit=async e=>{
e.preventDefault();
const input=$('cmd'),send=$('send'),error=$('send-error'),draft=input.value,text=draft.trim();
if(!text||send.disabled)return;
const ctrl=new AbortController(),timer=setTimeout(()=>ctrl.abort(),5000);
send.disabled=true;
send.textContent='Sending…';
error.textContent='';
try{
const r=await post('/instruct',text,ctrl.signal);
if(!r.ok)throw new Error('Command was not acknowledged');
if(input.value===draft)input.value='';
}catch{
error.textContent='Couldn’t confirm the command was sent. Your draft is still here.';
}finally{
clearTimeout(timer);
send.textContent='Send';
send.disabled=false;
}
};
$('stop').onclick=()=>post('/stop','').catch(()=>{});
let picksSent=0,picksTaken=0;
$('scene').onchange=()=>{const n=++picksSent;post('/scene',$('scene').value).catch(()=>{}).finally(()=>{picksTaken=Math.max(picksTaken,n)})};

function draw(s){
const c=$('map'),g=c.getContext('2d'),[W,H]=s.size||[8,6];
const rEntries=(s.room_bounds&&Object.keys(s.room_bounds).length)?Object.entries(s.room_bounds):(s.rooms&&!Array.isArray(s.rooms)?Object.entries(s.rooms):[]);
let hasB=false,minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
const addPt=(px,py)=>{
if(!Number.isFinite(px)||!Number.isFinite(py))return;
hasB=true;
if(px<minX)minX=px;if(px>maxX)maxX=px;
if(py<minY)minY=py;if(py>maxY)maxY=py;
};
const activeRoom=(s.scene&&s.room_bounds&&Array.isArray(s.room_bounds[s.scene]))?s.room_bounds[s.scene]:null;
if(activeRoom&&activeRoom.length>=4){
addPt(activeRoom[0],activeRoom[1]);addPt(activeRoom[2],activeRoom[3]);
}else{
for(const[,rect]of rEntries){
if(Array.isArray(rect)&&rect.length>=4){addPt(rect[0],rect[1]);addPt(rect[2],rect[3])}
}
}
for(const[a,b]of(s.walls||[])){
if(Array.isArray(a))addPt(a[0],a[1]);
if(Array.isArray(b))addPt(b[0],b[1]);
}
for(const t of(s.objects||[])){
const r=Number(t.r)||0;
addPt(t.x-r,t.y-r);addPt(t.x+r,t.y+r);
}
for(const pt of(s.trail||[])){if(Array.isArray(pt))addPt(pt[0],pt[1])}
for(const pt of(s.poses||[])){if(Array.isArray(pt))addPt(pt[0],pt[1])}
if(Array.isArray(s.pose)&&s.pose.length>=2){addPt(s.pose[0]-0.25,s.pose[1]-0.25);addPt(s.pose[0]+0.25,s.pose[1]+0.25)}
if(!hasB){minX=-W/2;maxX=W/2;minY=-H/2;maxY=H/2}
else{const pad=0.4;minX-=pad;maxX+=pad;minY-=pad;maxY+=pad}
const spanX=Math.max(1,maxX-minX),spanY=Math.max(1,maxY-minY),k=Math.min(c.width/spanX,c.height/spanY)*.92,cx=(minX+maxX)/2,cy=(minY+maxY)/2;
const css=getComputedStyle(document.documentElement),ink=css.getPropertyValue('--ink'),mut=css.getPropertyValue('--muted'),
acc=css.getPropertyValue('--accent'),line=css.getPropertyValue('--line')||'#e4e2dc',panel=css.getPropertyValue('--panel')||'#fff';
const X=x=>c.width/2+(x-cx)*k,Y=y=>c.height/2-(y-cy)*k;
g.clearRect(0,0,c.width,c.height);
g.save();
g.strokeStyle=line;g.lineWidth=1;g.globalAlpha=.75;
for(let gx=Math.ceil(minX);gx<=Math.floor(maxX);gx++){g.beginPath();g.moveTo(X(gx),Y(maxY));g.lineTo(X(gx),Y(minY));g.stroke()}
for(let gy=Math.ceil(minY);gy<=Math.floor(maxY);gy++){g.beginPath();g.moveTo(X(minX),Y(gy));g.lineTo(X(maxX),Y(gy));g.stroke()}
g.restore();
if(rEntries.length){
g.save();
for(const[rname,rect]of rEntries){
if(!Array.isArray(rect)||rect.length<4)continue;
const[x0,y0,x1,y1]=rect,rx=X(Math.min(x0,x1)),ry=Y(Math.max(y0,y1)),rw=Math.abs(x1-x0)*k,rh=Math.abs(y1-y0)*k;
g.strokeStyle=mut;g.lineWidth=1;g.setLineDash([4,4]);g.globalAlpha=.35;g.strokeRect(rx,ry,rw,rh);
g.setLineDash([]);g.fillStyle=mut;g.globalAlpha=.55;g.font='600 11px system-ui';
g.fillText(String(rname),rx+8,ry+15);
}
g.restore();
}else if(Array.isArray(s.rooms)&&s.rooms.length){
g.save();g.fillStyle=mut;g.globalAlpha=.55;g.font='600 11px system-ui';
g.fillText(s.rooms.join(' / '),X(minX)+8,Y(maxY)+15);g.restore();
}
g.strokeStyle=ink;g.lineWidth=3;g.lineCap='round';
for(const[a,b]of(s.walls||[])){g.beginPath();g.moveTo(X(a[0]),Y(a[1]));g.lineTo(X(b[0]),Y(b[1]));g.stroke()}
const placed=[];
const placeLabel=(txt,cx,cy,rr,bold,col)=>{
g.font=bold?'600 13px system-ui':'12px system-ui';
const tw=g.measureText(txt).width,th=14;
const cands=[[cx+rr+5,cy+4],[cx-tw/2,cy-rr-6],[cx-tw/2,cy+rr+th],[cx-rr-tw-5,cy+4]];
for(let dy=16;dy<=48;dy+=16){cands.push([cx+rr+5,cy+4+dy],[cx+rr+5,cy+4-dy])}
let chosen=cands[0],found=false;
for(const[bx,by]of cands){
const lx=Math.max(4,Math.min(c.width-tw-4,bx)),ly=Math.max(th+2,Math.min(c.height-4,by));
const box=[lx-2,ly-th,lx+tw+2,ly+3];
if(!placed.some(p=>!(box[2]<p[0]||box[0]>p[2]||box[3]<p[1]||box[1]>p[3]))){chosen=[lx,ly];placed.push(box);found=true;break}
}
const lx=Math.max(4,Math.min(c.width-tw-4,chosen[0])),ly=Math.max(th+2,Math.min(c.height-4,chosen[1]));
if(!found){placed.push([lx-2,ly-th,lx+tw+2,ly+3])}
g.lineWidth=3;g.strokeStyle=panel;g.strokeText(txt,lx,ly);
g.fillStyle=col;g.fillText(txt,lx,ly);
};
for(const t of(s.objects||[])){
const cx=X(t.x),cy=Y(t.y),rr=Math.max(4,(t.r||.25)*k),isUser=Boolean(t.user);
if(isUser){g.beginPath();g.arc(cx,cy,rr+4,0,7);g.strokeStyle=acc;g.lineWidth=2;g.globalAlpha=.85;g.stroke()}
g.beginPath();g.arc(cx,cy,rr,0,7);
g.fillStyle=isUser?acc:(t.color||mut);
g.globalAlpha=isUser?.9:(t.known?.42:.16);g.fill();
g.globalAlpha=isUser?1:(t.known?.95:.45);
g.strokeStyle=(isUser||t.known)?ink:mut;g.lineWidth=isUser?2:(t.known?1.8:1);g.stroke();
g.globalAlpha=1;
const lbl=isUser?'you':(t.label+(t.id!==undefined?' '+t.id:''));
placeLabel(lbl,cx,cy,rr,isUser,(isUser||t.known)?ink:mut);
}
g.fillStyle=acc;g.globalAlpha=.35;
for(const p of(s.poses||[])){g.beginPath();g.arc(X(p[0]),Y(p[1]),2.5,0,7);g.fill()}
g.globalAlpha=1;
const trail=s.trail||[];
if(trail.length){
g.strokeStyle=acc;g.lineWidth=2;g.beginPath();
trail.forEach(([x,y],i)=>i?g.lineTo(X(x),Y(y)):g.moveTo(X(x),Y(y)));
g.stroke();
}
const[x,y,a]=s.pose||[0,0,0];
g.beginPath();g.arc(X(x),Y(y),.15*k,0,7);g.fillStyle=acc;g.fill();
g.beginPath();g.moveTo(X(x),Y(y));g.lineTo(X(x+.35*Math.cos(a)),Y(y+.35*Math.sin(a)));
g.strokeStyle=ink;g.lineWidth=3;g.stroke();
}

let scenesKey='';
function showStatus(text,style=text){
const status=$('status');
if(status.textContent!==text)status.textContent=text;
status.className='status '+style;
}
function render(s,picksSeen){
$('instruction').textContent=s.instruction||'Tell the robot what to do';
showStatus(s.status||'idle');
$('posture').textContent='posture: '+(s.posture||'neutral');
if(s.chime){$('chime').hidden=false;$('chime').textContent='chime: '+s.chime}else{$('chime').hidden=true}
$('rooms').textContent=(s.rooms&&s.rooms.length)?'rooms: '+s.rooms.join(', '):'';
$('todo').textContent=s.todo||'(no to-do list)';
$('todo').className=s.todo?'':'muted';
$('thought').textContent=s.thought||'(none)';
$('searching').innerHTML=(s.searching||[]).map(t=>`<li>${esc(t)}</li>`).join('')||'<li class="muted">no search yet</li>';
const steps=s.steps||[];
$('steps').innerHTML=steps.map((it,idx)=>`<li class="${idx===steps.length-1?'now':''}${String(it).startsWith('thought:')?' thought':''}">${esc(it)}</li>`).join('')||'<li class="muted">none yet</li>';
$('said').innerHTML=(s.said||[]).map(t=>`<li>${esc(t)}</li>`).join('')||'<li class="muted">nothing yet</li>';
const saidArr=s.said||[];
if(saidArr.length>0){
let lastSaid=saidArr[saidArr.length-1]||'';
lastSaid=lastSaid.replace(/^["'“]/,'').replace(/["'”]$/,'').trim();
if(lastSaid){$('reply').hidden=false;$('reply-text').textContent=lastSaid}
else{$('reply').hidden=true}
}else{$('reply').hidden=true}
$('problems').innerHTML=(s.problems||[]).map(t=>`<li>${esc(t)}</li>`).join('')||'<li class="muted">none</li>';
$('history').innerHTML=(s.history||[]).slice().reverse().map(h=>`<li><b>${esc(h.outcome)}</b> ${esc(h.instruction)}</li>`).join('')||'<li class="muted">nothing yet</li>';
const key=(s.scenes||[]).join('|');
if(key!==scenesKey){
scenesKey=key;
$('scene').innerHTML=(s.scenes||[]).map(n=>`<option>${esc(n)}</option>`).join('');
}
if(picksSeen===picksSent){
const room=s.scene_loading||s.scene||'';
if($('scene').value!==room)$('scene').value=room;
$('scene').disabled=!!s.scene_loading;
}
draw(s);
voice(s);
cams(s);
}

let camsKey='',lastCamAt=0,camsVisible=false;
const camObserver=new IntersectionObserver(([entry])=>{
camsVisible=entry.isIntersecting;
if(camsVisible){lastCamAt=0;refreshCams()}
});
camObserver.observe($('camsec'));
document.addEventListener('visibilitychange',()=>{
if(!document.hidden){lastCamAt=0;refreshCams()}
});
function cams(s){
const list=s.cameras||[];
if(!list.length){$('camsec').hidden=true;return}
$('camsec').hidden=false;
const key=list.join('|');
if(key!==camsKey){
camsKey=key;
$('cams').innerHTML=list.map(n=>`<figure><img data-cam="${esc(n)}" alt="${esc(n)}"><figcaption>${esc(n.replace(/_/g,' '))}</figcaption></figure>`).join('');
$('cams').querySelectorAll('img[data-cam]').forEach(img=>{
img.onload=()=>{img.dataset.loading='';img.dataset.loadingAt='0'};
img.onerror=()=>{img.dataset.loading='';img.dataset.loadingAt='0'};
});
lastCamAt=0;
}
refreshCams();
}
function refreshCams(){
if(!camsVisible||document.hidden||$('camsec').hidden)return;
const now=Date.now();
if(now-lastCamAt<200)return;
lastCamAt=now;
$('cams').querySelectorAll('img[data-cam]').forEach(img=>{
const started=Number(img.dataset.loadingAt||0);
if(img.dataset.loading&&now-started<=4000)return;
img.dataset.loading='1';
img.dataset.loadingAt=String(now);
img.src=`/camera/${encodeURIComponent(img.dataset.cam)}.png?t=${now}`;
});
}

let rec=null,pendingMic=null,spoken=-1;
async function talk(){
if(rec||pendingMic)return;
const request={};pendingMic=request;
try{
const stream=await navigator.mediaDevices.getUserMedia({audio:true});
if(pendingMic!==request){
stream.getTracks().forEach(t=>t.stop());
if(!rec)$('talk').textContent='Hold to talk';
return;
}
pendingMic=null;
const ctx=new AudioContext(),src=ctx.createMediaStreamSource(stream),node=ctx.createScriptProcessor(4096,1,1),chunks=[];
node.onaudioprocess=e=>chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
src.connect(node);node.connect(ctx.destination);
rec={stream,ctx,node,chunks};$('talk').textContent='Listening… release to send';
}catch(e){if(pendingMic===request)pendingMic=null;$('heard').textContent='No microphone: '+e.message}
}
function wav(chunks,rate){
const n=chunks.reduce((a,c)=>a+c.length,0),pcm=new Float32Array(n);let o=0;
for(const c of chunks){pcm.set(c,o);o+=c.length}
const step=rate/16000,m=Math.floor(n/step),v=new DataView(new ArrayBuffer(44+m*2));
const w=(p,s)=>[...s].forEach((ch,i)=>v.setUint8(p+i,ch.charCodeAt(0)));
w(0,'RIFF');v.setUint32(4,36+m*2,true);w(8,'WAVE');w(12,'fmt ');v.setUint32(16,16,true);
v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,16000,true);v.setUint32(28,32000,true);
v.setUint16(32,2,true);v.setUint16(34,16,true);w(36,'data');v.setUint32(40,m*2,true);
for(let i=0;i<m;i++){const x=Math.max(-1,Math.min(1,pcm[Math.floor(i*step)]));v.setInt16(44+i*2,x*32767,true)}
return v.buffer;
}
function release(){
pendingMic=null;
if(!rec)return;
const{stream,ctx,node,chunks}=rec;rec=null;node.disconnect();
stream.getTracks().forEach(t=>t.stop());
const body=wav(chunks,ctx.sampleRate);ctx.close();
$('talk').textContent='Hold to talk';$('heard').textContent='Hearing…';
fetch('/voice',{method:'POST',body}).catch(()=>{});
}
$('talk').onpointerdown=talk;$('talk').onpointerup=release;$('talk').onpointerleave=release;
$('talk').onpointercancel=release;
function toggleVoice(){
if(!window.speechSynthesis)return;
if(!$('speak').checked)window.speechSynthesis.cancel();
}
function voice(s){
if(!s.voice)return;
$('talk').hidden=false;$('speaklabel').hidden=false;
if(s.heard)$('heard').textContent='Heard: “'+s.heard+'”';
if(spoken<0||s.said_total<spoken){spoken=s.said_total;return}
const fresh=(s.said||[]).slice(Math.max(0,(s.said||[]).length-(s.said_total-spoken)));
spoken=s.said_total;
if($('speak').checked){
if(window.speechSynthesis){
fresh.forEach(t=>{const u=new SpeechSynthesisUtterance(t);window.speechSynthesis.speak(u)});
}else{
fresh.reduce((p,text)=>p.then(()=>new Promise(done=>{
const a=new Audio('/speech?text='+encodeURIComponent(text));a.onended=done;a.onerror=done;a.play().catch(done);
})),Promise.resolve());
}
}
}
async function pollLoop(){
const ctrl=new AbortController();
const timer=setTimeout(()=>ctrl.abort(),2500);
// a poll sent before the server took the latest pick still shows the room it replaced
const picksSeen=picksTaken;
try{
const r=await fetch('/state',{cache:'no-store',signal:ctrl.signal});
if(!r.ok)throw new Error('State request failed');
render(await r.json(),picksSeen);
$('state-stale').hidden=true;
}catch{
showStatus('Live updates unavailable','stale');
$('state-stale').hidden=false;
}finally{
clearTimeout(timer);
setTimeout(pollLoop,350);
}
}
pollLoop();
</script></body></html>"""
