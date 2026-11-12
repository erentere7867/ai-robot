"""The planner: the model writes a program for each instruction, and repairs it when the gate or the run says no.

Output is three fields, say_first, plan and program (architecture section 7). They are written as labelled lines with
the program in a code block, and a grammar holds the model to that form. **Amended:** the spec asks for JSON; with JSON
a small model has to escape every quote and newline of the program inside a string, which is where it fails, while the
fields and their meaning are unchanged.

The worked examples teach the API's style only. None of them solves a task in the suite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from robot.agent.api import Object, api_card, side
from robot.agent.context import scene_table
from robot.mind.brain import Brain, Reply

RULES = """You control a small wheeled robot. For each instruction, write a short Python program that does it, using only \
the API below.

Rules:
- Use only robot, geom, math, the four exceptions and plain Python. No imports, no print.
- Known objects are in the table, by id: use robot.obj(id). Use robot.where or robot.search for anything else.
- Every turn and move needs its direction in words: robot.turn(90, "left"), robot.move(1.0, "backward"). "Back up"
  is robot.move(metres, "backward"); "spin" is robot.spin(turns), on the spot.
- go_to(thing, distance=0.3): distance is how far short of it to stop, never the thing's own distance.
- Behind, in front of, left or right of a thing: robot.go_to(robot.spot(thing, side="behind")).
- Around a thing: robot.orbit(thing, laps=2, direction="right"); "right" is clockwise, "left" anticlockwise.
- To look for something not in the table: robot.search(description, area=...): "everywhere" unless a room is named.
- Whether a thing is open, on, empty or the like: robot.check("Is the door open?", thing). Free space in front:
  robot.clearance("ahead").
- What the user told you to remember: robot.recall(words). Where you last saw a thing: robot.where(description)[0],
  with .position and .last_seen.
- "Me" and "the user" are robot.user(). "Come here": robot.go_to(robot.user(), distance=0.5).
- "start" is saved by itself when a task begins: robot.go_to("start") comes back to it. Never save_place("start").
- Do only what was asked: come back to "start" only when told to.
- "Earlier" lists what you did before. Use it for "the other one", "that one", "again" or a correction ("no, ...").
- Ask with robot.ask only when two things fit equally well and nothing earlier decides.
- If the robot cannot do it (it has wheels but no arms, and cannot fly or climb), say so with robot.say and stop.
- Keep the program short. When it is done, say what you did with robot.say.

Answer in exactly this form:
say: <one short sentence to speak now>
plan: <one sentence>
```python
<the program>
```"""

GRAMMAR = r"""root ::= "say: " text "\nplan: " text "\n```python\n" line ("\n" line)* "\n```"
text ::= [^\n`]{1,200}
line ::= [^\n`]*
"""


def _context(room: str, objects: list[tuple[int, str, str, float, float, float]], user: int | None = None,
             others: str = "none", earlier: tuple[str, ...] = ()) -> str:
    """A worked example's situation, built by the same code as real ones: (id, label, colour, metres, bearing, seen)."""
    table = scene_table([Object(i, label, (0.0, 0.0), d, b, side(b), 0.3, color, seen)
                         for i, label, color, d, b, seen in objects], user)
    return f"Robot: in room {room}, battery 85%. Other rooms: {others}.\nKnown objects:\n{table}" + _earlier(earlier)


def _earlier(earlier: tuple[str, ...] | list[str]) -> str:
    return ("\nEarlier:\n" + "\n".join(f"- {line}" for line in earlier)) if earlier else ""


# Each example teaches one pattern the untrained model got wrong (directions as words, spot for behind or in front,
# orbit, coming back to "start", following, searching, the conversation so far). None solves a task in the suite.
SEEDS: list[tuple[str, str, str]] = [
    (_context("lounge", [(2, "sofa", "grey", 2.1, 60, 0), (5, "lamp", "white", 3.0, -40, 0)]),
     "Go and wait behind the sofa.",
     """say: Going behind the sofa.
plan: Find the free spot behind the sofa and drive there.
```python
robot.go_to(robot.spot(robot.obj(2), side="behind"))
robot.say("I'm behind the sofa.")
```"""),
    (_context("hall", []),
     "Turn right 45 degrees, then drive forward a metre and a half.",
     """say: Turning, then driving forward.
plan: Turn right 45 degrees, then move forward 1.5 metres.
```python
robot.turn(45, "right")
robot.move(1.5, "forward")
robot.say("Done: I turned right and drove a metre and a half.")
```"""),
    (_context("conservatory", [(7, "plant", "green", 1.8, -25, 0)]),
     "Do a lap around the plant, clockwise, then come back.",
     """say: Going round the plant.
plan: Orbit the plant once clockwise, then return to where the task began.
```python
robot.orbit(robot.obj(7), laps=1, direction="right")
robot.go_to("start")
robot.say("I went round the plant and came back.")
```"""),
    (_context("kitchen", [(3, "cup", "red", 0.4, 10, 0), (8, "cup", "green", 2.2, -70, 0)],
              earlier=('"Go to the cup": you drove to the red cup 3 (1.6 m away); said "Here." (done)',)),
     "No, the other one.",
     """say: Going to the other cup.
plan: Earlier I went to cup 3, so go to the other cup, 8.
```python
robot.go_to(robot.obj(8), distance=0.3)
robot.say("Here is the other cup, the green one.")
```"""),
    (_context("study", [(4, "desk", "brown", 1.5, 0, 0)], others="hall, kitchen"),
     "Is there an umbrella anywhere?",
     """say: Let me look for an umbrella.
plan: It is not in the table, so search everywhere and report.
```python
umbrella = robot.search("umbrella", area="everywhere")
if umbrella:
    robot.say(f"Yes, there is a {umbrella.color} umbrella in the {robot.room()}.")
else:
    robot.say("I couldn't find an umbrella.")
```"""),
    (_context("hall", [(6, "person", "", 1.4, 20, 0)], user=6),
     "Walk with me for a minute.",
     """say: I'll walk with you.
plan: Follow the user at a metre for sixty seconds.
```python
me = robot.user()
if me:
    robot.follow(me, distance=1.0, seconds=60)
    robot.say("That was a minute.")
else:
    robot.say("I can't see you.")
```"""),
    (_context("playroom", [(3, "toy", "green", 0.9, -20, 0), (8, "toy", "yellow", 2.6, 75, 0)]),
     "Bring me to a toy.",
     """say: Which toy?
plan: Two toys fit equally, so ask which one, then go to it.
```python
answer = (robot.ask("The green toy or the yellow one?") or "green").lower()
toy = robot.obj(8) if "yellow" in answer else robot.obj(3)
robot.go_to(toy, distance=0.3)
robot.say(f"Here is the {toy.color} toy.")
```"""),
    (_context("hall", [(2, "lamp", "white", 2.3, 40, 0)]),
     "How far away is the lamp?",
     """say: Let me check.
plan: Read the lamp's distance and side from the table.
```python
lamp = robot.obj(2)
robot.say(f"The lamp is {lamp.distance:.1f} metres away, {lamp.side} of me.")
```"""),
]

JUDGE_THRESHOLD = 0.5  # set from labelled programs; see robot/eval/judge.py

_FORM = re.compile(r"say: (?P<say>[^\n]*)\nplan: (?P<plan>[^\n]*)\n```python\n(?P<program>.*?)\n```", re.S)


class FormatError(ValueError):
    """The model's answer did not have the three fields."""


@dataclass
class PlanResult:
    say_first: str
    plan: str
    program: str
    reply: Reply


class LocalPlanner:
    """The on-board model. Holds one task's conversation so a repair sees what went before."""

    def __init__(self, brain: Brain | None = None, max_tokens: int = 600, think: int = 0) -> None:
        self.brain = brain or Brain()
        self.max_tokens, self.think = max_tokens, think
        self.prefix = [{"role": "system", "content": f"{RULES}\n\nAPI:\n{api_card()}"}]
        for context, instruction, answer in SEEDS:
            self.prefix += [{"role": "user", "content": f"{context}\nInstruction: {instruction}"},
                            {"role": "assistant", "content": answer}]
        self.messages: list[dict[str, Any]] = []

    def warm(self) -> None:
        """Have the model read the rules, API card and examples now, so the first instruction finds them cached
        (thousands of tokens: on a CPU that is minutes)."""
        self.brain.complete([*self.prefix, {"role": "user", "content": "Instruction: wait"}], 'root ::= "ok"', 1)

    def _write(self) -> PlanResult:
        reply = self.brain.complete(self.messages, GRAMMAR, self.max_tokens, self.think)
        self.messages.append({"role": "assistant", "content": reply.text})
        match = _FORM.search(reply.text)
        if not match:
            raise FormatError(reply.text[:200])
        return PlanResult(match["say"].strip(), match["plan"].strip(), match["program"].rstrip() + "\n", reply)

    def plan(self, context: str, instruction: str, earlier: list[str] | tuple[str, ...] = ()) -> PlanResult:
        """`earlier`: one line for each of the last few instructions and what the robot did about them."""
        self.messages = [*self.prefix, {"role": "user",
                                        "content": f"{context}{_earlier(earlier)}\nInstruction: {instruction}"}]
        return self._write()

    def judge(self, instruction: str, readback: str, threshold: float | None = None) -> tuple[bool, float]:
        """Does what the program will do (its dry run, in words) match the instruction? A fresh question, outside the
        planning conversation: judging a plain description is easier than writing the program. The answer is the
        model's probability of "yes", compared with a threshold measured on labelled programs (JUDGE_THRESHOLD)."""
        messages = [{"role": "system", "content": "You check a small robot's plan before it moves. Compare what the "
                     "robot was asked with what its program will do. Directions, distances, counts and targets must "
                     "match. Answer yes or no."},
                    {"role": "user", "content": f"Asked: {instruction}\nThe program will: {readback}\n"
                                                "Does the program do what was asked?"}]
        p = self.brain.yes_probability(messages)
        return p >= (JUDGE_THRESHOLD if threshold is None else threshold), p

    def repair(self, problem: str, context: str | None = None) -> PlanResult:
        now = f"\nThe robot now:\n{context}" if context else ""
        self.messages.append({"role": "user", "content": f"{problem}{now}\nWrite the whole program again, fixed, in "
                                                         "the same form."})
        return self._write()
