"""The agent: an agentic tool-calling loop, as in coding agents, around the robot's vision-language model.

One task is one running conversation. The model sees the cameras and the object list, calls tools in its native
format, and reads every result before it decides the next step. Pictures come back after anything that moves the
robot or after it asks to look. A reply with no tool call is spoken and ends the task. The user can speak while a
step runs: the step is interrupted and the words join the conversation, so the model can change course.
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from robot.agent.api import Preempted, SafetyStop, side, wrap
from robot.agent.instant import instant
from robot.agent.tools import CAMERA_NAMES, ActionResult, Tools, recalled_facts, speakable
from robot.perception.camera import data_url

MAX_TURNS = 16  # model replies per task
MAX_OBJECTS = 8  # nearest objects listed in full; keep context lean and let look/search/recall fetch specifics
MAX_HIDDEN_NAMES = 12  # the rest by name only, a few tokens each
MAX_IMAGES = 3  # pictures kept in the conversation (older ones replaced by a note; saves ~1100 tokens/image)
IMAGE_WIDTH = 768  # pixels across each picture; at 512 the 2B model misjudged colours of small things (a red bike)
THINK = {"off": 0, "low": 128, "medium": 512, "high": 2048}  # most tokens of reasoning before each turn
EARLIER = 4  # earlier tasks summarised at the start of a new one
REPEATS = 2  # the same call (same arguments) this many times in a row runs; the next is refused with a note
FRONT_CHECK = {"go_toward", "drive", "move", "turn"}

SYSTEM = """You are Sprocket, a friendly, helpful wheeled robot. Act only through tools.
Stay on open floor. Never drive under or inside objects or climb stairs. You cannot pick things up.
For unsafe or unsupported requests, stay put, explain why, and ask before substituting a nearby spot.

- Your cameras face front, rear-left, and rear-right. For conditions about what is visible from here, use look
  without moving; search can drive away.
- The object list holds people and things you've looked for. Numbers on pictures are object IDs; prefer IDs to
  names. Detector labels are guesses. Verify features in the picture before driving or claiming you found a target;
  if you can't distinguish it, say so rather than guessing. Compare measured ranges, not apparent sizes.
- Use go_to for named things, surfaces, areas, and saved places (such as "where you were before"), even if unlisted.
  Use go_toward only for an unnamed spot. If a target is out of view, search where it likely is before saying
  it isn't there: give a room, known landmark, or promising direction.
- Read tool results before choosing what to do next. If a tool fails, try another way or ask the user.
- Keep the user's conditions, order, and limits. Check conditions before acting. Ask rather than guessing when
  a missing detail would change what you do, and follow any later correction.
- For multi-part requests, use todo. Use todo_done when results show a step is complete, or todo_blocked when
  you cannot finish it. Tell the user what remains.
- End with one or two short, warm spoken sentences and no tool call."""


@dataclass
class Step:
    """One model turn: what it said aloud, the calls it made and what each returned."""

    said: str
    calls: list[tuple[str, dict[str, Any], str, bool]] = field(default_factory=list)
    model_s: float = 0.0
    tokens: int = 0
    thought: str = ""  # what the model reasoned before acting (thinking on)
    t: float = 0.0  # the robot's clock when the turn began (simulated seconds in the simulator)


@dataclass
class TaskRecord:
    instruction: str
    outcome: str = ""  # done, preempted or failed
    steps: list[Step] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def model_s(self) -> float:
        return sum(s.model_s for s in self.steps)

    def lines(self, thoughts: bool = False) -> list[str]:
        """Each call and its result, and each plain reply; with `thoughts`, the start of each turn's reasoning."""
        out = []
        for s in self.steps:
            if thoughts and s.thought:
                out.append(f"thought: {' '.join(s.thought.split())}")
            for name, args, result, ok in s.calls:
                arg_text = ", ".join(f"{k}={v!r}" for k, v in args.items())
                out.append(f"{name}({arg_text}) -> {result.splitlines()[0][:120] if result else ''}"
                           + ("" if ok else "  [failed]"))
            if s.said and not s.calls:
                out.append(f'reply: "{s.said}"')
        return out


def _image_part(url: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url}}


class Agent:
    def __init__(self, brain: Any, *, memory: Any = None, live: Any = None, cameras: str = "all",
                 see_after_moves: bool | str = "front", max_turns: int = MAX_TURNS, think: str = "low",
                 image_width: int = IMAGE_WIDTH, review: bool = False, think_when: str = "always") -> None:
        """`cameras`: the pictures sent with each request, "front", "all" three, or "none" (the object list only);
        the model asks for others with look. `see_after_moves`: "front" (default: send one fresh front_camera frame
        after moves for closed-loop sight without 3-camera bloat), True (all cameras + full scene), or False.
        `think`: how long the model may reason before each turn: off, low, medium or high (see THINK).
        `image_width`: pixels across each picture the model gets (16:9). `review`: before a task ends, show the
        model the request and what it did once, for a last check (it costs a model turn on every task).
        `think_when`: "always", or "news": reason only before a call that follows something the model did not ask
        for (the request, the user's words, a call that failed); after calls that went as asked, and for the
        review, it carries on without."""
        self.brain, self.memory, self.live = brain, memory, live
        self.cameras = {"all": ("front_camera", "rear_left_camera", "rear_right_camera"), "front": ("front_camera",),
                        "none": ()}[cameras]
        self.see_after_moves = see_after_moves
        self.max_turns, self.think, self.image_width, self.review = max_turns, THINK[think], image_width, review
        self.think_when = think_when
        self.earlier: list[str] = []  # one line per task before the last one
        self.last = ""  # the last task step by step: each call with its result, and the reply
        self._heard: queue.Queue[str] = queue.Queue()
        self._acting = threading.Event()
        self._robot: Any = None
        self._stopped = threading.Event()

    # ---- from other threads (the live page, the microphone) ------------------------------------------------------
    def interject(self, text: str) -> None:
        """The user spoke while a task runs: interrupt the current step and give the words to the model."""
        self._heard.put(text)
        if self._acting.is_set() and self._robot is not None:
            self._robot.cancel()

    def stop(self) -> None:
        """End the task now."""
        self._stopped.set()
        if self._robot is not None:
            self._robot.cancel()

    # ---- what the model sees -----------------------------------------------------------------------------------
    def _scene_text(self, robot: Any, marked_ages: dict[int, float] | None, full: bool) -> str:
        snap, lines = robot.snapshot(), []
        if full:
            rooms = [r for r in snap.get("rooms", []) if r]
            lines.append(f"You are in the {snap.get('room') or 'room'}"
                         + (f" (rooms you know: {', '.join(rooms)})" if len(rooms) > 1 else "")
                         + f". Battery {snap.get('battery', 1.0):.0%}.")
        objects = self._objects_text(snap, marked_ages, full)
        lines.extend([objects, *self._spatial_lines(robot, snap)])
        if full and (places := self._places_text(robot, snap.get("pose"))):
            lines.append(places)
        return "\n".join(lines)

    @staticmethod
    def _objects_text(snap: dict[str, Any], marked_ages: dict[int, float] | None, full: bool) -> str:
        user_id = snap.get("user")
        marks = marked_ages or {}
        everything = sorted(snap.get("objects", []), key=lambda o: (o.id not in marks, o.distance))
        objects = everything[:MAX_OBJECTS]
        if user_id is not None and not any(o.id == user_id for o in objects):
            user_obj = next((o for o in everything if o.id == user_id), None)
            if user_obj is not None:
                objects = [*objects[:MAX_OBJECTS - 1], user_obj]
        rest = [o for o in everything if o not in objects]
        if not objects:
            return "Objects: none seen yet."
        groups: dict[str, list[str]] = {}
        for o in objects:
            who = "person (the user)" if o.id == user_id else o.label
            row = f"{o.id} {who}"
            if full:
                if o.color and o.id != user_id:
                    row += f", {o.color}"
                row += f", {o.distance:.1f} m {o.side}"
                if o.room not in ("", "room", snap.get("room")):
                    row += f", in the {o.room}"
            age = marks.get(o.id, o.last_seen)
            if marked_ages is not None or age >= 2:
                row += ", seen <1 s ago" if age < 1 else f", seen {age:.0f} s ago"
            group = ("Objects" if marked_ages is None else "Marked at capture" if o.id in marks
                     else "Last known (not marked at capture)")
            groups.setdefault(group, []).append(row)
        sep = "\n  " if full else "; "
        lines = [f"{group}:{sep}{sep.join(rows)}" for group, rows in groups.items()]
        if rest:  # names only: "how many chairs" was answered 4 when the second of 2 chairs was cut off
            names = ", ".join(f"{o.id} {o.label}" for o in rest[:MAX_HIDDEN_NAMES])
            lines.append(f"(+{len(rest)} more: {names}{', ...' if len(rest) > MAX_HIDDEN_NAMES else ''})")
        return "\n".join(lines)

    @staticmethod
    def _spatial_lines(robot: Any, snap: dict[str, Any]) -> list[str]:
        lines = []
        if (pose := snap.get("pose")) is not None:
            lines.append(f"Map pose: ({pose.x:.1f}, {pose.y:.1f}) m, heading {pose.yaw:.0f}°.")
        if spatial := getattr(robot, "spatial_observation", None):
            sigma, scan = spatial()
            if sigma is not None:
                lines.append(f"AMCL uncertainty (1 sigma): {sigma[0]:.2f} m, {math.degrees(sigma[1]):.0f}° heading.")
            if isinstance(scan, str):
                lines.append(f"Scan: {scan}.")
            else:
                sectors = "; ".join(f"{name} {r:.1f} m" if r is not None else f"{name} unknown"
                                    for name, r in scan.items())
                lines.append(f"Scan (nearest sensor return): {sectors}.")
        return lines

    @staticmethod
    def _places_text(robot: Any, pose: Any) -> str:
        """The named places with how far and which way each is from here."""
        places = getattr(robot, "places", {})
        if pose is None or not places:
            return ""
        about = {"start": "where this request began", "where you were before": "where the request before began"}
        prev = places.get("where you were before")
        at_before = prev is not None and math.dist((prev.x, prev.y), (pose.x, pose.y)) <= 0.3
        here = []
        # List "where you were before" first, then "start", then mapped areas nearest-first.
        def order(kv: tuple[str, Any]) -> tuple[int, float, str]:
            name, p = kv
            rank = 0 if name == "where you were before" else 1 if name == "start" else 2
            return (rank, math.dist((p.x, p.y), (pose.x, pose.y)), name)

        for name, p in sorted(places.items(), key=order):
            d = math.dist((p.x, p.y), (pose.x, pose.y))
            if name == "start" and (d <= 0.3 or at_before):  # don't lure the robot back after reaching where it was
                continue
            bearing = math.degrees(wrap(math.atan2(p.y - pose.y, p.x - pose.x) - math.radians(pose.yaw)))
            where = f"{d:.1f} m {side(bearing)}" if d > 0.3 else "here"
            here.append(f"{name} ({about[name]}, {where})" if name in about else f"{name} ({where})")
        has_areas = any(n not in about for n in places)
        prefix = "Places you can go_to by name (or search near): " if has_areas else "Places you can go_to by name: "
        return prefix + "; ".join(here) if here else ""

    def _observation(self, robot: Any, tools: Tools, cameras: list[str], header: str, *,
                     include_scene: bool = True, camera_limit: int | None = None) -> dict[str, Any]:
        observe = getattr(robot, "camera_observation", None)
        view_fn = getattr(robot, "camera_view", None)
        width, height = self.image_width, round(self.image_width * 9 / 16)
        views = []
        for cam in cameras:
            if len(views) == camera_limit:
                break
            if observe:
                view = observe(cam, width, height)
                frame, ids = view if view is not None else (None, None)
            else:
                frame, ids = (view_fn(cam, width, height) if view_fn else None), None
            if frame is not None and getattr(frame, "size", 0):
                views.append((f"{CAMERA_NAMES.get(cam, cam)} camera:", frame, ids))
        recorded = len(tools.pictures)
        views.extend((caption, rgb, None) for caption, rgb in tools.pictures)
        tools.pictures.clear()
        views = views[-MAX_IMAGES:]
        marked_ages = {}
        for _, _, ages in views:
            for track_id, age in (ages or {}).items():
                marked_ages[track_id] = min(age, marked_ages.get(track_id, age))
        parts: list[dict[str, Any]] = []
        text = ""
        if include_scene or observe:
            text = self._scene_text(robot, marked_ages if observe else None, include_scene)
            if include_scene:
                text = f"{header}\n{text}"
                if tools.todo.steps:
                    text += f"\nTo-do list:\n{tools.todo.text()}"
                if not views and not cameras:
                    text += "\n(No pictures with this message: call look to see.)"
        if cameras and len(views) <= recorded:
            text = "\n".join(filter(None, (text, "No recent camera pictures attached.")))
        if text:
            parts.append({"type": "text", "text": text})
        for caption, rgb, ages in views:
            if ages:
                caption = f"{caption[:-1]} (badges {', '.join(map(str, sorted(ages)))}):"
            parts += [{"type": "text", "text": caption}, _image_part(data_url(rgb, max_width=width))]
        return {"role": "user", "content": parts}

    @staticmethod
    def _trim_images(messages: list[dict[str, Any]]) -> None:
        """Keep only the newest pictures so a long task fits the model's context; remove each dropped picture along
        with its caption instead of leaving '(older picture removed)' stubs that concatenate into the user prompt."""
        seen = 0
        for msg in reversed(messages):
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            i = len(content) - 1
            while i >= 0:
                if content[i].get("type") == "image_url":
                    seen += 1
                    if seen > MAX_IMAGES:
                        content.pop(i)
                        if i > 0 and content[i - 1].get("type") == "text" and content[i - 1].get("text", "").endswith(":"):
                            content.pop(i - 1)
                            i -= 1
                i -= 1

    # ---- the loop ----------------------------------------------------------------------------------------------
    def do(self, instruction: str, robot: Any, speak: bool = True) -> TaskRecord:
        record = TaskRecord(instruction)
        cmd = instant(instruction)
        if cmd is not None:  # "stop" and "wait" never wait for a model
            robot.stop()
            self._mood(robot, "listening" if cmd == "wait" else "neutral", "listen" if cmd == "wait" else "uh_oh")
            return self._finish(robot, record, "preempted" if cmd == "stop" else "done", "")

        self._robot = robot
        self._stopped.clear()
        self._drain()  # words meant for an earlier task
        if hasattr(robot, "begin"):
            robot.begin()
        tools = Tools(robot, self.memory)
        self._mood(robot, "listening", "listen")
        self._push(status="thinking", instruction=instruction, steps=[], todo="", problems=[])

        # One user turn per request, as the model is trained for tool use (its card advises against multi-turn chat):
        # earlier requests one line each, the last one step by step, then what the robot senses and the words.
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}]
        context = []
        older = self.earlier[:-1] if self.last else self.earlier
        if older:
            context.append("Earlier requests (oldest first):\n" + "\n".join(f"- {e}" for e in older))
        if self.last:
            context.append(f"The last request, step by step:\n{self.last}")
        recalled = recalled_facts(tools.memory, instruction)
        if recalled:
            heading = ("Remembered facts (oldest to newest; later facts supersede earlier ones):"
                       if len(recalled) > 1 else "Remembered facts:")
            context.append(heading + "\n" + "\n".join(f"- {f}" for f in recalled))
        first = self._observation(robot, tools, list(self.cameras), "What you sense now:")
        if context:
            first["content"].insert(0, {"type": "text", "text": "\n".join(context) + "\n"})
        first["content"].append({"type": "text", "text": f'\nThe user says: "{instruction}"'})
        messages.append(first)
        reviewed = False
        retried_empty = False
        last_said = ""
        spoke_since_action = False
        last_call, repeats = None, 0
        seen_calls: dict[tuple[str, ...], int] = {}
        looks_here = 0
        blocked_tools: set[str] = set()
        partial_motion: tuple[tuple[str, ...], ActionResult] | None = None
        schemas = tools.schemas()
        think = self.think  # the request is news
        sorry = "Sorry, I couldn't finish that. Could you tell me another way?"

        try:
            for turn in range(self.max_turns):
                if self._stopped.is_set():
                    return self._finish(robot, record, "preempted", last_said)
                self._trim_images(messages)
                self._push(thought="(thinking...)")
                t0, began = time.perf_counter(), round(float(getattr(robot, "t", 0.0)), 1)
                available = (
                    [s for s in schemas if s["function"]["name"] not in blocked_tools] if blocked_tools else schemas
                )
                reply = self.brain.chat(messages, available, think=think)
                step = Step("", model_s=round(time.perf_counter() - t0, 3), tokens=reply.write_tokens,
                            thought=reply.thought.strip(), t=began)
                record.steps.append(step)
                if self._stopped.is_set():
                    return self._finish(robot, record, "preempted", last_said)
                cut_off = reply.finish_reason == "length"
                if (cut_off or not reply.tool_calls) and (heard := self._drain()):
                    messages.append(self._correction(robot, tools, heard))
                    reviewed = spoke_since_action = False
                    think = self.think
                    continue
                if cut_off:
                    sorry = "The model reply was cut off; I didn't run anything from that turn."
                    record.problems.append(sorry)
                    break
                self._push(steps=record.lines(thoughts=True), thought=step.thought or "(no reasoning this turn)")
                assistant: dict[str, Any] = {"role": "assistant", "content": reply.text}
                if step.thought:  # kept with its tool calls, as LFM3's template expects within a task
                    assistant["reasoning_content"] = step.thought
                if reply.tool_calls:
                    assistant["tool_calls"] = [{"id": c.id, "type": "function", "function": {
                        "name": c.name, "arguments": json.dumps(c.args)}} for c in reply.tool_calls]
                messages.append(assistant)

                if not reply.tool_calls:
                    if not reply.text:
                        if retried_empty:
                            record.problems.append("the model returned another empty answer")
                            break
                        retried_empty = True
                        messages.append({"role": "user", "content":
                                         "Your last turn had no answer or tool call. Continue the current request: "
                                         "call a tool for your next step, or reply to the user."})
                        think = self.think
                        continue
                    needs_review = partial_motion is not None or (not reviewed and (
                        tools.todo.open_steps() or (self.review and any(s.calls for s in record.steps))
                    ))
                    if needs_review and turn + 1 < self.max_turns:
                        reviewed = True
                        messages.append({"role": "user", "content": self._review(
                            instruction, record, tools, partial_motion[1].text if partial_motion else "",
                        )})
                        spoke_since_action = spoke_since_action and partial_motion is None
                        partial_motion = None
                        think = self._budget(news=False)
                        continue
                    open_steps = tools.todo.open_steps()
                    if open_steps:
                        record.problems.append(f"open to-do steps left unfinished: {'; '.join(open_steps)}")
                    if reply.text and speak and not spoke_since_action:
                        robot.say(speakable(reply.text))
                    step.said = reply.text if not spoke_since_action else ""  # avoid duplicate speech after say()
                    last_said = (reply.text if not spoke_since_action else "") or last_said
                    return self._finish(robot, record, "failed" if open_steps else "done", last_said)

                barrier = ""
                plan_renumbered = False
                news = False
                visual_result = None
                visual_parts = []
                recorded = 0
                for call in reply.tool_calls:
                    if self._stopped.is_set():
                        return self._finish(robot, record, "preempted", last_said)
                    skipped = "Not run: the user spoke (their words follow)." if not self._heard.empty() else barrier
                    if not skipped and plan_renumbered and call.name in ("todo_done", "todo_blocked"):
                        skipped = "Not run: read the updated step numbers before updating a step."
                    if skipped:
                        step.calls.append((call.name, call.args, skipped, False))
                        messages.append({"role": "tool", "tool_call_id": call.id, "content": skipped})
                        continue
                    key = ("say",) if call.name == "say" else (call.name, json.dumps(call.args, sort_keys=True))
                    repeats = repeats + 1 if key == last_call else 1
                    last_call = key
                    if call.name == "look":
                        seen_calls[key] = seen_calls.get(key, 0) + 1
                    move_cams: tuple[str, ...] = ()
                    if call.name == "say" and repeats > 1:
                        result, ok = ("Not run: you already spoke. If the request is finished, end with plain text "
                                      "and no tool call."), False
                        blocked_tools.add("say")
                        record.problems.append("say: repeated")
                    elif call.name == "look" and call.args.get("what") and looks_here >= 3:
                        result, ok = ("Not run: you already looked around from here without moving. Call search(...) "
                                      "or move somewhere else to see new views."), False
                        blocked_tools.update(("look", "look_back"))
                        record.problems.append("look: already looked around here")
                    # search too: 24 cases in 20 2D passes sent the same search 3+ times in a row, 132 calls, every
                    # one refused by search's own check, which left the tool offered (S3 used all 16 turns).
                    elif (repeats > REPEATS and call.name != "look"
                          or call.name == "look" and seen_calls[key] > REPEATS + 1):
                        n_prev = (seen_calls[key] if call.name == "look" else repeats) - 1
                        result, ok = (f"Not run: you made this exact call {n_prev} times. Do something "
                                      "different, or if the request is done, reply to the user."), False
                        blocked_tools.add(call.name)
                        record.problems.append(f"{call.name}: repeated {n_prev} times")
                    else:
                        running = ", ".join(f"{k}={v!r}" for k, v in call.args.items())
                        self._push(status="acting", steps=[*record.lines(thoughts=True), f"{call.name}({running}) ..."])
                        prior_steps = tools.todo.steps if call.name == "todo" else []
                        try:
                            res = self._run(robot, tools, call.name, call.args, record)
                        except Preempted:
                            step.calls.append((call.name, call.args, "Interrupted before returning a result.", False))
                            raise
                        result, ok = res.text, res.ok
                        if prior_steps:
                            revised = tools.todo.steps
                            plan_renumbered |= len(revised) < len(prior_steps) or any(
                                old is not new and (old.done or new.done or old.title != new.title)
                                for old, new in zip(prior_steps, revised, strict=False)
                            )
                        if res.status == "partial":
                            # C1 reported success on the T4 even after stopping 1.08 m from the sofa.
                            partial_motion = key, res
                            barrier = ("Not run: an earlier call in this turn ended short; inspect the result before "
                                       "deciding the next action.")
                        elif partial_motion and ok and res.status in ("reached", "already_there"):
                            previous_key, previous_result = partial_motion
                            goal = previous_result.arrival_goal
                            if (res.arrival_goal == goal if goal is not None else key == previous_key):
                                partial_motion = None
                        if self.see_after_moves is True and tools.moved:
                            move_cams = self.cameras
                        elif (self.see_after_moves == "front" and self.cameras and call.name in FRONT_CHECK
                              and (ok or tools.moved) and not tools.looked and not tools.pictures):
                            move_cams = ("front_camera",)
                        if call.name == "say" and ok:
                            last_said = str(call.args.get("text", ""))
                            spoke_since_action = True
                        elif ok:
                            spoke_since_action = False
                            blocked_tools.discard("say")
                            if call.name == "look" and call.args.get("what"):
                                looks_here += 1
                                if looks_here >= 3:
                                    blocked_tools.update(("look", "look_back"))
                            elif call.name != "look":
                                blocked_tools.discard(call.name)
                    news = not ok or call.name == "ask"
                    if not ok:
                        barrier = barrier or "Skipped: an earlier call in this turn failed."
                    elif call.name == "ask":
                        barrier = "Not run: read the result of asking the user before deciding the next action."
                    cameras = list(dict.fromkeys([*move_cams, *tools.looked]))
                    if cameras or tools.pictures:
                        recorded = len(tools.pictures)
                        seen = self._observation(robot, tools, cameras, "What you sense now:",
                                                 include_scene=self.see_after_moves is True and tools.moved)
                        visual_parts = seen["content"]
                        if any(part["type"] == "image_url" for part in visual_parts):
                            barrier = barrier or (
                                "Not run: inspect the returned pictures before deciding the next action."
                            )
                        else:
                            news = True
                            barrier = barrier or (
                                "Not run: no recent camera pictures were available after the previous action."
                            )
                    if ok and call.name == "search":
                        barrier = barrier or "Not run: read the search result before deciding the next action."
                    step.calls.append((call.name, call.args, result, ok))
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                    if visual_parts:
                        visual_result = messages[-1]
                self._push(status="acting", steps=record.lines(thoughts=True), todo=tools.todo.text() if tools.todo.steps else "",
                           problems=list(record.problems))
                if tools.todo.steps and not {c.name for c in reply.tool_calls} & {"todo", "todo_done"}:
                    messages[-1]["content"] += f"\nTo-do list now:\n{tools.todo.text()}"
                if turn + 1 >= self.max_turns // 2:
                    # 47 of 150 cases in three 2D passes used every turn without a word to the user; C3 reached the
                    # right box first, then drove box to box for 14 more turns, never told how many it had.
                    messages[-1]["content"] += f"\n{self.max_turns - turn - 1} of {self.max_turns} turns are left."
                if tools.moved:
                    seen_calls.clear()
                    looks_here = 0
                    blocked_tools.clear()
                if tools.moved and self.see_after_moves is not True:  # where places are from here now ("come back here")
                    snap = robot.snapshot()
                    places = self._places_text(robot, snap.get("pose"))
                    if places:
                        messages[-1]["content"] += f"\nNow in the {snap.get('room') or 'room'}. {places}"

                heard = self._drain()
                think = self._budget(news=news or bool(heard))
                if visual_result is not None:  # pictures come back inside the requesting tool result, as trained: a
                    # new user message would end its tool-use turn and drop the reasoning so far.
                    visual_result["content"] = [{"type": "text", "text": visual_result["content"]}, *visual_parts]
                if heard:  # the user spoke mid-task: a new user turn with their words and what the robot senses now
                    messages.append(self._correction(robot, tools, heard, recorded=recorded))
                    reviewed = spoke_since_action = False
                tools.moved, tools.looked = False, []
            else:
                record.problems.append(f"no answer after {self.max_turns} turns")
        except SafetyStop as exc:
            record.problems.append(f"safety stop: {exc}")
        except Preempted:
            return self._finish(robot, record, "preempted", last_said)
        except Exception as exc:  # the model server failed: say so plainly rather than freeze
            record.problems.append(f"{type(exc).__name__}: {exc}")
        if speak:
            try:
                robot.say(sorry)
                last_said = sorry
            except Preempted:
                return self._finish(robot, record, "preempted", last_said)
        return self._finish(robot, record, "failed", last_said)

    def _budget(self, news: bool) -> int:
        """Tokens the model may reason for before its next call."""
        return self.think if news or self.think_when == "always" else 0

    def _drain(self) -> list[str]:
        heard = []
        while not self._heard.empty():
            heard.append(self._heard.get_nowait())
        return heard

    def _correction(self, robot: Any, tools: Tools, heard: list[str], *, recorded: int = 0) -> dict[str, Any]:
        words = "".join(f'The user says: "{h}"\n' for h in heard)
        # Leave room for recorded evidence, but never let it crowd out what the user is asking about now.
        return self._observation(robot, tools, list(self.cameras), words + "What you sense now:",
                                 camera_limit=max(1, MAX_IMAGES - recorded))

    @staticmethod
    def _review(instruction: str, record: TaskRecord, tools: Tools, partial_motion: str = "") -> str:
        """Before a task ends: the request, what was done, and what remains."""
        calls = [line for line in record.lines() if not line.startswith("reply:")]
        done = ("\n".join(f"- {line}" for line in calls[-12:]) if calls else
                "- no tool call at all: you have not moved, turned, looked or done anything for this request; only "
                "tools do things, so saying you did something does not do it")
        open_steps = [step.title for step in tools.todo.steps if not step.done and not step.blocker]
        blocked_steps = [step.status() for step in tools.todo.steps if not step.done and step.blocker]
        done_steps = tools.todo.done_steps()
        todo = f"\nCompleted to-do steps: {'; '.join(done_steps)}." if done_steps else ""
        if blocked_steps:
            todo += f"\nBlocked to-do steps: {'; '.join(blocked_steps)}."
        if open_steps:
            todo += f"\nOpen to-do steps: {'; '.join(open_steps)}."
        if partial_motion:
            todo += f"\nUnfinished motion: {partial_motion} Do not describe it as complete unless you finish it."
        return (f'Before you finish, check every part of the request: "{instruction}"\nWhat you did:\n{done}{todo}\n'
                "Use the user's later corrections if they changed the request. "
                "If you can finish a remaining step, use tools now. Otherwise reply briefly with what you finished "
                "and what remains, with no tool call.")

    def _run(self, robot: Any, tools: Tools, name: str, args: dict[str, Any],
             record: TaskRecord) -> ActionResult:
        """One call. A user who speaks during it interrupts it; a plain stop ends the task (Preempted)."""
        self._acting.set()
        try:
            if not self._heard.empty():
                if not self._stopped.is_set():
                    robot.resume()
                return ActionResult("Not run: the user spoke (their words follow).", False, "interrupted")
            res = tools.execute(name, args)
        except Preempted:
            if self._stopped.is_set() or self._heard.empty():
                raise
            robot.resume()
            return ActionResult("Interrupted: the user spoke (their words follow).", False, "interrupted")
        finally:
            self._acting.clear()
        if not res.ok:
            record.problems.append(f"{name}: {res.text}")
            self._mood(robot, "uh_oh", "uh_oh")
        elif name == "search" and tools.found is not None:
            self._mood(robot, "happy", "happy")
        return res

    def _finish(self, robot: Any, record: TaskRecord, outcome: str, said: str) -> TaskRecord:
        record.outcome = outcome
        self._robot = None
        if outcome == "done":
            self._mood(robot, "happy" if not record.problems else "neutral", "happy" if not record.problems else "")
        elif outcome == "failed":
            self._mood(robot, "uh_oh", "uh_oh")
        actions = "; ".join(line.split(" -> ")[0] + " -> " + line.split(" -> ")[1][:60] if " -> " in line else line
                            for line in record.lines() if not line.startswith("reply:"))[:300]
        summary = (f'"{record.instruction}" -> {outcome}' + (f"; you did: {actions}" if actions else "")
                   + (f'; you said "{said[:100]}"' if said else ""))
        began = getattr(robot, "places", {}).get("start")
        moved_note = ""
        if began is not None and hasattr(robot, "x") and hasattr(robot, "y"):
            dm = math.dist((began.x, began.y), (robot.x, robot.y))
            if dm > 0.1:
                moved_note = f"; ended {dm:.1f} m from where that request began"
        self.earlier = [*self.earlier, summary][-EARLIER:]
        self.last = "\n".join([f'  The user said "{record.instruction}"', *(f"  {line}" for line in record.lines()),
                               f"  ({outcome}{moved_note})"])
        self._push(status="stopped" if outcome == "preempted" else outcome, steps=record.lines(thoughts=True),
                   problems=list(record.problems))
        return record

    def forget(self) -> None:
        """Start a fresh conversation (a new room, a new user)."""
        self.earlier, self.last = [], ""

    @staticmethod
    def _mood(robot: Any, posture: str, chime: str) -> None:
        """The body shows how things are going (a posture and a chime), so the model never has to."""
        express = getattr(robot, "express", None)
        if express is not None:
            try:
                express(posture, chime_kind=chime, seconds=0.0)
            except (Exception, Preempted):  # noqa: BLE001 - only a gesture: a stop is seen by the next real step
                pass

    def _push(self, **fields: Any) -> None:
        if self.live is not None:
            self.live.update(**fields)
