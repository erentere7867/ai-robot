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
from robot.agent.tools import CAMERA_NAMES, Tools, recalled_facts, speakable
from robot.perception.camera import data_url

MAX_TURNS = 16  # model replies per task
MAX_OBJECTS = 8  # nearest objects listed in full; keep context lean and let look/search/recall fetch specifics
MAX_IMAGES = 3  # pictures kept in the conversation (older ones replaced by a note; saves ~1100 tokens/image)
IMAGE_WIDTH = 768  # pixels across each picture; at 512 the 2B model misjudged colours of small things (a red bike)
THINK = {"off": 0, "low": 128, "medium": 512, "high": 2048}  # most tokens of reasoning before each turn
EARLIER = 4  # earlier tasks summarised at the start of a new one
REPEATS = 2  # the same call (same arguments) this many times in a row runs; the next is refused with a note

SYSTEM = """You are Sprocket, a small wheeled robot who lives with people, like a friendly pet that can also help.
You have three cameras: front, rear-left (behind you on the left) and rear-right (behind you on the right). You act
only by calling tools. Your motion tools reach open floor outside objects. You cannot park underneath or inside an
object, climb steps or pick things up. When the requested destination is unsupported, stay put and tell the user.
Never use go_to, drive or go_toward to try to squeeze underneath/inside furniture or climb stairs — refuse right
away and ask permission before substituting a nearby spot.

How you work:
- The pictures are what you see right now. Use look(direction, what) for a fresh picture with candidate IDs
  without moving. A search can drive away; don't use it for a condition about what's visible from here.
- The object list holds people and the things tools have looked for; the numbers on the pictures are its ids. Use
  ids as targets when you have them, else name the thing ("red ball"): tools look for it with the cameras.
- Detector labels are guesses. Check the search picture against the requested features before driving or saying
  you found it. If the view doesn't distinguish the objects, say so rather than guessing an ID.
- To approach any named thing, surface, area or saved place (such as "where you were before") use go_to by name or ID
  (even if not in the object list yet). Use its measured ranges to compare objects before moving; their sizes in the
  wide camera pictures can mislead. Use go_toward only for an unnamed spot in robot directions.
- Call tools to act. You get each result back, then decide what to do next. If one fails, read why and try another
  way, or ask the user.
- If something is not in view, reason about where it likely belongs and give search a room, a visible landmark/area in
  near (e.g. near="kitchen" or near="stairs"), or a promising camera direction in toward before saying it is not there.
- For a request with several parts, first write a to-do list with todo, then tick each step off with todo_done.
- When the request is done, reply with one or two short, warm spoken sentences and no tool call. That ends the task.
- If a request is impossible or unsafe for you, say so kindly instead."""


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
    def _scene_text(self, robot: Any) -> str:
        snap = robot.snapshot()
        user_id = snap.get("user")
        rooms = [r for r in snap.get("rooms", []) if r]
        lines = [f"You are in the {snap.get('room') or 'room'}"
                 + (f" (rooms you know: {', '.join(rooms)})" if len(rooms) > 1 else "")
                 + f". Battery {snap.get('battery', 1.0):.0%}."]
        everything = sorted(snap.get("objects", []), key=lambda o: o.distance)
        objects = everything[:MAX_OBJECTS]
        if user_id is not None and not any(o.id == user_id for o in objects):
            user_obj = next((o for o in everything if o.id == user_id), None)
            if user_obj is not None:
                objects = [*objects[:MAX_OBJECTS - 1], user_obj]
        rest = len(everything) - len(objects)
        if objects:
            lines.append("Objects (id, what, colour, how far, which way):")
            for o in objects:
                who = "person (the user)" if o.id == user_id else o.label
                tint = f", {o.color}" if o.color and o.id != user_id else ""
                seen = "" if o.last_seen < 2 else f", seen {o.last_seen:.0f} s ago"
                lines.append(f"  {o.id} {who}{tint}, {o.distance:.1f} m {o.side}{seen}")
            if rest > 0:
                lines.append(f"  (+{rest} farther; use look, search, or recall to inspect others)")
        else:
            lines.append("Objects: none seen yet.")
        places = self._places_text(robot, snap.get("pose"))
        if places:
            lines.append(places)
        return "\n".join(lines)

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
                     include_scene: bool = True) -> dict[str, Any]:
        parts: list[dict[str, Any]] = []
        if include_scene:
            text = f"{header}\n{self._scene_text(robot)}"
            if tools.todo.steps:
                text += f"\nTo-do list:\n{tools.todo.text()}"
            parts.append({"type": "text", "text": text})
        view_fn = getattr(robot, "camera_view", None)
        shown = []
        width, height = self.image_width, round(self.image_width * 9 / 16)
        for cam in cameras:
            frame = view_fn(cam, width, height) if view_fn else None
            if frame is None or getattr(frame, "size", 0) == 0:
                continue
            url = data_url(frame, max_width=width)
            name = CAMERA_NAMES.get(cam, cam)
            parts += [{"type": "text", "text": f"{name} camera:"}, _image_part(url)]
            shown.append(name)
        for caption, rgb in tools.pictures:
            parts += [{"type": "text", "text": caption}, _image_part(data_url(rgb, max_width=width))]
            shown.append(caption)
        tools.pictures.clear()
        if include_scene:
            if not shown and cameras:
                parts[0]["text"] += "\n(No camera pictures right now.)"
            elif not shown:
                parts[0]["text"] += "\n(No pictures with this message: call look to see.)"
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
        last_said = ""
        spoke_since_action = False
        last_call, repeats = None, 0
        seen_calls: dict[tuple[str, ...], int] = {}
        looks_here = 0
        blocked_tools: set[str] = set()
        think = self.think  # the request is news
        sorry = "Sorry, I couldn't finish that. Could you tell me another way?"

        try:
            for _ in range(self.max_turns):
                if self._stopped.is_set():
                    return self._finish(robot, record, "preempted", last_said)
                self._trim_images(messages)
                self._push(thought="(thinking...)")
                t0, began = time.perf_counter(), round(float(getattr(robot, "t", 0.0)), 1)
                schemas = [s for s in tools.schemas() if s["function"]["name"] not in blocked_tools]
                reply = self.brain.chat(messages, schemas, think=think)
                step = Step("", model_s=round(time.perf_counter() - t0, 3), tokens=reply.write_tokens,
                            thought=reply.thought.strip(), t=began)
                record.steps.append(step)
                if self._stopped.is_set():
                    return self._finish(robot, record, "preempted", last_said)
                if reply.finish_reason == "length":
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
                    heard = self._drain()
                    if heard:  # the user spoke while the model was answering: carry on with their words
                        messages.append({"role": "user", "content": "".join(f'The user says: "{h}"\n'
                                                                            for h in heard).strip()})
                        think = self.think
                        continue
                    if not reviewed and (tools.todo.open_steps() or (self.review and any(s.calls for s in record.steps))):
                        reviewed = True
                        messages.append({"role": "user", "content": self._review(instruction, record, tools)})
                        think = self._budget(news=False)
                        continue
                    open_steps = tools.todo.open_steps()
                    if open_steps:
                        record.problems.append(f"open to-do steps left unfinished: {'; '.join(open_steps)}")
                        return self._finish(robot, record, "failed", last_said)
                    if reply.text and speak and not spoke_since_action:
                        robot.say(speakable(reply.text))
                    step.said = reply.text if not spoke_since_action else ""  # avoid duplicate speech after say()
                    last_said = (reply.text if not spoke_since_action else "") or last_said
                    return self._finish(robot, record, "done", last_said)

                failed = False
                partial = False
                visual_result = None
                for call in reply.tool_calls:
                    if self._stopped.is_set():
                        return self._finish(robot, record, "preempted", last_said)
                    if not self._heard.empty():
                        result, ok = "Not run: the user spoke (their words follow).", False
                        failed = True
                        step.calls.append((call.name, call.args, result, ok))
                        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                        continue
                    key = ("say",) if call.name == "say" else (call.name, json.dumps(call.args, sort_keys=True))
                    repeats = repeats + 1 if key == last_call else 1
                    last_call = key
                    seen_calls[key] = seen_calls.get(key, 0) + 1
                    if failed:
                        result, ok = (
                            "Not run: an earlier call in this turn ended short; inspect the result before deciding "
                            "the next action." if partial else "Skipped: an earlier call in this turn failed."
                        ), False
                    elif visual_result is not None:
                        result, ok = "Not run: inspect the returned pictures before deciding the next action.", False
                    elif call.name == "say" and repeats > 1:
                        result, ok = ("Not run: you already spoke. If the request is finished, end with plain text "
                                      "and no tool call."), False
                        blocked_tools.add("say")
                        record.problems.append("say: repeated")
                    elif call.name == "look" and call.args.get("what") and looks_here >= 3:
                        result, ok = ("Not run: you already looked around from here without moving. Call search(...) "
                                      "or move somewhere else to see new views."), False
                        blocked_tools.update(("look", "look_back"))
                        record.problems.append("look: already looked around here")
                    elif (call.name == "search" and seen_calls[key] > 1) or (
                        repeats > REPEATS and call.name != "look"
                    ) or (call.name == "look" and seen_calls[key] > REPEATS + 1):
                        n_prev = (seen_calls[key] if call.name in ("look", "search") else repeats) - 1
                        result, ok = (f"Not run: you made this exact call {n_prev} times. Do something "
                                      "different, or if the request is done, reply to the user."), False
                        if call.name != "search":  # other search(near=...) areas remain valid
                            blocked_tools.add(call.name)
                        record.problems.append(f"{call.name}: repeated {n_prev} times")
                    else:
                        running = ", ".join(f"{k}={v!r}" for k, v in call.args.items())
                        self._push(status="acting", steps=[*record.lines(thoughts=True), f"{call.name}({running}) ..."])
                        try:
                            result, ok = self._run(robot, tools, call.name, call.args, record)
                        except Preempted:
                            step.calls.append((call.name, call.args, "Interrupted before returning a result.", False))
                            raise
                        failed = not ok
                        if tools.last_result is not None and tools.last_result.status == "partial":
                            partial = True
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
                    step.calls.append((call.name, call.args, result, ok))
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                    if visual_result is None and (tools.looked or tools.pictures):
                        visual_result = messages[-1]
                self._push(status="acting", steps=record.lines(thoughts=True), todo=tools.todo.text() if tools.todo.steps else "",
                           problems=list(record.problems))
                if tools.todo.steps and not {c.name for c in reply.tool_calls} & {"todo", "todo_done"}:
                    messages[-1]["content"] += f"\nTo-do list now:\n{tools.todo.text()}"
                if tools.moved:
                    seen_calls = {k: v for k, v in seen_calls.items() if k[0] == "search"}
                    looks_here = 0
                    blocked_tools.clear()
                if tools.moved and self.see_after_moves is not True:  # where places are from here now ("come back here")
                    snap = robot.snapshot()
                    places = self._places_text(robot, snap.get("pose"))
                    if places:
                        messages[-1]["content"] += f"\nNow in the {snap.get('room') or 'room'}. {places}"

                heard = self._drain()
                think = self._budget(news=failed or bool(heard))
                full_move = self.see_after_moves is True and tools.moved
                open_loop = any((ok or tools.moved) and name in ("go_toward", "drive", "move", "turn")
                                for name, _, _, ok in step.calls)
                front_move = (self.see_after_moves == "front" and open_loop and bool(self.cameras)
                              and not tools.looked and not tools.pictures)
                move_cams = self.cameras if full_move else ("front_camera",) if front_move else ()
                cameras = list(dict.fromkeys([*move_cams, *tools.looked]))
                if heard:  # the user spoke mid-task: a new user turn with their words and what the robot senses now
                    header = "".join(f'The user says: "{h}"\n' for h in heard) + "What you sense now:"
                    messages.append(self._observation(robot, tools, cameras, header))
                elif cameras or tools.pictures:  # pictures come back inside the requesting tool result, as trained: a
                    # new user message would end its tool-use turn and drop the reasoning so far. Skip re-sending the
                    # full scene list on stationary looks/crops and single-frame post-move snapshots.
                    seen = self._observation(robot, tools, cameras, "What you sense now:", include_scene=full_move)
                    result_msg = visual_result if visual_result is not None else messages[-1]
                    result_msg["content"] = [{"type": "text", "text": result_msg["content"]}, *seen["content"]]
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

    @staticmethod
    def _review(instruction: str, record: TaskRecord, tools: Tools) -> str:
        """Before a task ends: the request, what was done, and any open to-do steps, for one last check."""
        calls = [line for line in record.lines() if not line.startswith("reply:")]
        done = ("\n".join(f"- {line}" for line in calls[-12:]) if calls else
                "- no tool call at all: you have not moved, turned, looked or done anything for this request; only "
                "tools do things, so saying you did something does not do it")
        open_steps = tools.todo.open_steps()
        todo = f"\nOpen to-do steps: {'; '.join(open_steps)}." if open_steps else ""
        return (f'Before you finish, check every part of the request: "{instruction}"\nWhat you did:\n{done}{todo}\n'
                "If something is still missing, do it now with tools. If everything is done, give your final short "
                "reply to the user, with no tool call.")

    def _run(self, robot: Any, tools: Tools, name: str, args: dict[str, Any],
             record: TaskRecord) -> tuple[str, bool]:
        """One call. A user who speaks during it interrupts it; a plain stop ends the task (Preempted)."""
        self._acting.set()
        try:
            if not self._heard.empty():
                if not self._stopped.is_set():
                    robot.resume()
                return "Not run: the user spoke (their words follow).", False
            result, ok = tools.run(name, args)
        except Preempted:
            if self._stopped.is_set() or self._heard.empty():
                raise
            robot.resume()
            return "Interrupted: the user spoke (their words follow).", False
        finally:
            self._acting.clear()
        if not ok:
            record.problems.append(f"{name}: {result}")
            self._mood(robot, "uh_oh", "uh_oh")
        elif name == "search" and tools.found is not None:
            self._mood(robot, "happy", "happy")
        return result, ok

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
