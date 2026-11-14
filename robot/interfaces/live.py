"""The live page: what Sprocket v2 is thinking and doing, and a box to command it.

One small web page (standard library only) shows the instruction, current action, subgoal, posture, chime,
working-memory scratchpad, intent step trace, optional camera/Set-of-Mark view, a 2D map with the robot's
pose/poses/trail/objects/rooms, what the robot has said, and the history of instructions. Commands typed on the
page, the Stop button, and the room picker arrive in `Live.inbox` for whoever drives the robot
(`robot/interfaces/demo.py`).

    live = Live()
    live.serve(port=8765)          # then open http://localhost:8765
    agent = Agent(planner, live=live)

It listens on this computer only by default: the page can command the robot.
"""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import asdict, dataclass, field, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse


@dataclass
class State:
    status: str = "idle"  # idle, planning, running, done, failed, stopped, refused
    instruction: str = ""
    subgoal: str = ""
    action: str = ""
    posture: str = "neutral"
    chime: str = ""
    intents: list[dict[str, Any] | str] = field(default_factory=list)
    scratchpad: str = ""
    problems: list[str] = field(default_factory=list)
    said: list[str] = field(default_factory=list)
    pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    poses: list[tuple[float, float, float]] = field(default_factory=list)
    trail: list[tuple[float, float]] = field(default_factory=list)
    objects: list[dict[str, Any]] = field(default_factory=list)
    walls: list[tuple[tuple[float, float], tuple[float, float]]] = field(default_factory=list)
    size: tuple[float, float] = (8.0, 6.0)
    rooms: list[str] = field(default_factory=list)
    scene: str = ""
    scenes: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    cameras: list[str] = field(default_factory=list)
    scan: list[tuple[float, float]] = field(default_factory=list)
    view: str = ""  # optional base64 camera or Set-of-Mark image data URL
    voice: bool = False
    heard: str = ""
    said_total: int = 0


def _serialize_intent(item: Any) -> dict[str, Any] | str:
    """Convert an Intent dataclass, dict, or string into a JSON-safe representation."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        out = dict(item)
    elif is_dataclass(item) and not isinstance(item, type):
        out = asdict(item)
    else:
        return str(item)
    target = out.get("target")
    if target is not None and not isinstance(target, (str, int, float, bool)):
        out["target"] = str(target)
    return out


class Live:
    """Shared Sprocket v2 state, updated by the agent/driver and polled by the live web UI."""

    def __init__(self) -> None:
        self.state = State()
        self.inbox: queue.Queue[tuple[str, str]] = queue.Queue()  # ("instruct", text), ("stop", ""), ("scene", name)
        self.camera: Any = None  # name -> PNG bytes or None, for multi-camera panel
        self.ears: Any = None
        self.mouth: Any = None
        self._lock = threading.Lock()

    def update(self, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                if not hasattr(self.state, key):
                    continue
                if key == "intents" and isinstance(value, (list, tuple)):
                    value = [_serialize_intent(item) for item in value]
                setattr(self.state, key, value)

    def snapshot(self) -> str:
        with self._lock:
            return json.dumps(asdict(self.state), default=str)

    def show_world(self, sim: Any, *, include_view: bool = False) -> None:
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
                    "id": t.id,
                })
        sampled = sim.trace.poses[-1500::5]
        trail = [(round(x, 2), round(y, 2)) for _, x, y, _ in sampled]
        poses = [(round(x, 2), round(y, 2), round(yaw, 3)) for _, x, y, yaw in sampled[-120:]]
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
        }
        if hasattr(sim, "posture"):
            changes["posture"] = sim.posture
        if getattr(sim, "scratchpad_summary", ""):
            changes["scratchpad"] = sim.scratchpad_summary
        if include_view and hasattr(sim, "render_view"):
            try:
                from robot.perception.camera import data_url

                changes["view"] = data_url(sim.render_view())
            except Exception:  # noqa: BLE001 - rendering is best-effort for live visualization
                pass
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

        class Handler(BaseHTTPRequestHandler):
            def _send(self, body: bytes, kind: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 - http.server method name
                if self.path.startswith("/camera/") and live.camera is not None:
                    png = live.camera(self.path.split("/")[2].split(".")[0])
                    if png:
                        self._send(png, "image/png")
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
                    audio = self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 8_000_000))
                    threading.Thread(target=live.hear, args=(audio,), daemon=True).start()
                    self._send(b"{}", "application/json")
                    return
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
pre{margin:0;padding:8px 10px;border-radius:8px;background:var(--bg);font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
overflow-x:auto;white-space:pre-wrap}
.intents{margin:0;padding-left:20px;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
.intents li{padding:3px 6px;border-radius:6px;margin-bottom:4px}.intents li.now{background:var(--hi)}
canvas{width:100%;aspect-ratio:4/3;border-radius:8px;background:var(--bg)}
.viewframe{width:100%;max-height:320px;object-fit:contain;border-radius:8px;background:var(--bg);display:block}
ul{margin:0;padding-left:18px}.muted{color:var(--muted)}.wide{grid-column:1/-1}
.mind-kv{display:flex;gap:12px;flex-wrap:wrap;font-size:14px;margin-bottom:6px}
form{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
input,select,button{font:inherit;border-radius:8px;border:1px solid var(--line);padding:8px 12px;background:var(--bg);
color:var(--ink)}input{flex:1;min-width:200px}button{cursor:pointer;background:var(--accent);color:#fff;border:0}
button.stop{background:var(--bad)}.cams{display:grid;gap:8px;grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}
.cams figure{margin:0}.cams img{width:100%;border-radius:8px;background:var(--bg)}.cams figcaption{font-size:13px;color:var(--muted)}
.hint{font-size:13px;color:var(--muted);margin-top:6px}
</style></head><body><main>
<section class="wide">
<h1 id="instruction">Tell the robot what to do</h1>
<div class="badges">
<span id="status" class="status">idle</span>
<span id="posture" class="badge">posture: neutral</span>
<span id="chime" class="badge" hidden></span>
<span id="rooms" class="muted"></span>
</div>
<form id="form">
<input id="cmd" placeholder="e.g. Circle the table twice, then come back to me" autocomplete="off">
<button type="submit">Send</button>
<button type="button" class="stop" id="stop">Stop</button>
<button type="button" id="talk" hidden title="Hold and speak">Hold to talk</button>
<label id="speaklabel" class="muted" hidden><input type="checkbox" id="speak" style="min-width:0;flex:0"> speak aloud</label>
<select id="scene" title="Room"></select>
</form>
<div class="hint" id="hint">The robot is the blue dot with a heading indicator; you are the blue person.
Faded objects are remembered but not currently seen.</div>
<div class="hint" id="heard"></div>
</section>

<section>
<h2>Closed-Loop Mind</h2>
<div class="mind-kv">
<span>Action: <strong id="action">—</strong></span>
<span>Subgoal: <span id="subgoal" class="muted">—</span></span>
</div>
<h3>Scratchpad</h3>
<pre id="scratchpad" class="muted">(empty)</pre>
<h3>Intents</h3>
<ol id="intents" class="intents"></ol>
</section>

<section>
<h2>Map</h2>
<canvas id="map" width="800" height="600"></canvas>
</section>

<section id="viewsec" hidden>
<h2>Camera / Set-of-Mark View</h2>
<img id="view" class="viewframe" alt="Visual observation">
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
<h2>What the cameras see</h2>
<div id="cams" class="cams"></div>
</section>
</main><script>
const $=id=>document.getElementById(id);
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
const post=(p,text)=>fetch(p,{method:'POST',body:JSON.stringify({text})});
$('form').onsubmit=e=>{e.preventDefault();const t=$('cmd').value.trim();if(t){post('/instruct',t);$('cmd').value=''}};
$('stop').onclick=()=>post('/stop','');
$('scene').onchange=()=>post('/scene',$('scene').value);

function formatIntent(item){
if(typeof item==='string')return esc(item);
if(!item||typeof item!=='object')return esc(String(item));
const act=esc(item.action||'done');
const tgt=item.target!==undefined&&item.target!==null&&item.target!==''?'('+esc(item.target)+')':'';
const sg=item.subgoal?' — '+esc(item.subgoal):'';
const meta=[item.posture&&item.posture!=='neutral'?'posture:'+item.posture:'',item.chime?'chime:'+item.chime:''].filter(Boolean).join(', ');
const suffix=meta?' ['+esc(meta)+']':'';
return `<b>${act}</b>${tgt}${sg}${suffix}`;
}

function draw(s){
const c=$('map'),g=c.getContext('2d'),[W,H]=s.size||[8,6],k=Math.min(c.width/W,c.height/H)*.92;
const css=getComputedStyle(document.documentElement),ink=css.getPropertyValue('--ink'),mut=css.getPropertyValue('--muted'),
acc=css.getPropertyValue('--accent');
const X=x=>c.width/2+x*k,Y=y=>c.height/2-y*k;
g.clearRect(0,0,c.width,c.height);
g.strokeStyle=ink;g.lineWidth=3;
for(const[a,b]of(s.walls||[])){g.beginPath();g.moveTo(X(a[0]),Y(a[1]));g.lineTo(X(b[0]),Y(b[1]));g.stroke()}
for(const t of(s.objects||[])){
g.beginPath();g.arc(X(t.x),Y(t.y),Math.max(3,t.r*k),0,7);
g.fillStyle=t.person?acc:(t.color||mut);
g.globalAlpha=t.known?.85:.25;g.fill();g.globalAlpha=1;
g.strokeStyle=mut;g.lineWidth=1;g.stroke();
g.fillStyle=ink;g.font='13px system-ui';
g.fillText((t.person?'you':t.label)+(t.person?'':' '+t.id),X(t.x)+t.r*k+3,Y(t.y)+4);
}
g.fillStyle='#e0457b';
for(const[sx,sy]of(s.scan||[])){g.fillRect(X(sx)-1.5,Y(sy)-1.5,3,3)}
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
async function tick(){
try{
const s=await(await fetch('/state',{cache:'no-store'})).json();
$('instruction').textContent=s.instruction||'Tell the robot what to do';
$('status').textContent=s.status||'idle';
$('status').className='status '+(s.status||'idle');
$('posture').textContent='posture: '+(s.posture||'neutral');
if(s.chime){$('chime').hidden=false;$('chime').textContent='chime: '+s.chime}else{$('chime').hidden=true}
$('rooms').textContent=(s.rooms&&s.rooms.length)?'rooms: '+s.rooms.join(', '):'';
$('action').textContent=s.action||'—';
$('subgoal').textContent=s.subgoal||'—';
$('scratchpad').textContent=s.scratchpad||'(empty)';
$('scratchpad').className=s.scratchpad?'':'muted';
const intents=s.intents||[];
$('intents').innerHTML=intents.map((it,idx)=>`<li class="${idx===intents.length-1?'now':''}">${formatIntent(it)}</li>`).join('')||'<li class="muted">none yet</li>';
if(s.view){$('viewsec').hidden=false;$('view').src=s.view}else{$('viewsec').hidden=true}
$('said').innerHTML=(s.said||[]).map(t=>`<li>${esc(t)}</li>`).join('')||'<li class="muted">nothing yet</li>';
$('problems').innerHTML=(s.problems||[]).map(t=>`<li>${esc(t)}</li>`).join('')||'<li class="muted">none</li>';
$('history').innerHTML=(s.history||[]).slice().reverse().map(h=>`<li><b>${esc(h.outcome)}</b> ${esc(h.instruction)}</li>`).join('')||'<li class="muted">nothing yet</li>';
const key=(s.scenes||[]).join('|')+'#'+(s.scene||'');
if(key!==scenesKey){
scenesKey=key;
$('scene').innerHTML=(s.scenes||[]).map(n=>`<option${n===s.scene?' selected':''}>${esc(n)}</option>`).join('');
}
draw(s);
}catch(e){}
}

let camtick=0;
function cams(s){
if(!s.cameras||!s.cameras.length)return;
$('camsec').hidden=false;
if(camtick++%3)return;
$('cams').innerHTML=s.cameras.map(n=>`<figure><img src="/camera/${n}.png?t=${Date.now()}" alt="${n}"><figcaption>${n.replace(/_/g,' ')}</figcaption></figure>`).join('');
}
setInterval(async()=>{try{cams(await(await fetch('/state',{cache:'no-store'})).json())}catch(e){}},1000);

let rec=null,spoken=-1;
async function talk(){
if(rec)return;
try{
const stream=await navigator.mediaDevices.getUserMedia({audio:true});
const ctx=new AudioContext(),src=ctx.createMediaStreamSource(stream),node=ctx.createScriptProcessor(4096,1,1),chunks=[];
node.onaudioprocess=e=>chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
src.connect(node);node.connect(ctx.destination);
rec={stream,ctx,node,chunks};$('talk').textContent='Listening… release to send';
}catch(e){$('heard').textContent='No microphone: '+e.message}
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
if(!rec)return;
const{stream,ctx,node,chunks}=rec;rec=null;node.disconnect();
stream.getTracks().forEach(t=>t.stop());
const body=wav(chunks,ctx.sampleRate);ctx.close();
$('talk').textContent='Hold to talk';$('heard').textContent='Hearing…';
fetch('/voice',{method:'POST',body});
}
$('talk').onpointerdown=talk;$('talk').onpointerup=release;$('talk').onpointerleave=release;
function voice(s){
if(!s.voice)return;
$('talk').hidden=false;$('speaklabel').hidden=false;
if(s.heard)$('heard').textContent='Heard: “'+s.heard+'”';
if(spoken<0||s.said_total<spoken){spoken=s.said_total;return}
const fresh=(s.said||[]).slice(Math.max(0,(s.said||[]).length-(s.said_total-spoken)));
spoken=s.said_total;
if($('speak').checked)fresh.reduce((p,text)=>p.then(()=>new Promise(done=>{
const a=new Audio('/speech?text='+encodeURIComponent(text));a.onended=done;a.onerror=done;a.play().catch(done);
})),Promise.resolve());
}
setInterval(async()=>{try{voice(await(await fetch('/state',{cache:'no-store'})).json())}catch(e){}},700);
setInterval(tick,400);tick();
</script></body></html>"""
