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
- Prefer skills: search to find something, orbit to circle something, follow to follow a person, go_to to drive \
somewhere. Use move, turn and spin for plain motion, with directions as words.
- The table of known objects gives each object's id, colour, distance and side as seen from the robot. Use \
robot.obj(id) for an object in the table.
- robot.where looks through everything seen so far; robot.detect only what is in view now; robot.search goes looking.
- "Me" and "the user" are robot.user().
- Before leaving a place you must come back to, call robot.save_place("start"); robot.go_to("start") returns there.
- If two things match the instruction equally well, ask with robot.ask.
- If the robot cannot do it (it has wheels but no arms, and cannot fly or climb), say so with robot.say and stop.
- Answer the user with robot.say. When the task is done, say what you did.

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
             others: str = "none") -> str:
    """A worked example's situation, built by the same code as real ones: (id, label, colour, metres, bearing, seen)."""
    table = scene_table([Object(i, label, (0.0, 0.0), d, b, side(b), 0.3, color, seen)
                         for i, label, color, d, b, seen in objects], user)
    return f"Robot: in room {room}, battery 85%. Other rooms: {others}.\nKnown objects:\n{table}"


SEEDS: list[tuple[str, str, str]] = [
    (_context("hall", [(2, "lamp", "white", 2.3, 40, 0)]),
     "How far away is the lamp?",
     """say: Let me check.
plan: Look the lamp up in memory and report its distance and side.
```python
lamps = robot.where("lamp")
if lamps:
    robot.say(f"The lamp is {lamps[0].distance:.1f} metres away, {lamps[0].side} of me.")
else:
    robot.say("I haven't seen a lamp.")
```"""),
    (_context("study", [(4, "person", "", 1.2, -100, 0), (7, "window", "", 3.1, 10, 0)], user=4),
     "Drive to the window, then turn to face me.",
     """say: Heading to the window.
plan: Go to the window from the table, then face the user.
```python
try:
    robot.go_to(robot.obj(7))
    robot.face(robot.user())
    robot.say("I'm at the window.")
except PathBlocked:
    robot.say("Something is blocking the way to the window.")
```"""),
    (_context("garden room", []),
     "Stay here until you see a dog, then bark.",
     """say: I'll keep watch for a dog.
plan: Wait until a dog is in view, face it and bark.
```python
dog = robot.wait_for("dog")
robot.face(dog)
robot.say("Woof!")
```"""),
    (_context("bedroom", [(5, "shoe", "red", 1.4, 130, 12)]),
     "Turn to look at the shoe and tell me what it's made of.",
     """say: Let me look at the shoe.
plan: Face the shoe from the table, ask the vision model, and report.
```python
shoe = robot.obj(5)
robot.face(shoe)
material = robot.describe(shoe, question="What is this shoe made of?")
robot.say(f"The shoe looks like {material}.")
```"""),
    (_context("playroom", [(3, "toy", "green", 0.9, -20, 0), (8, "toy", "yellow", 2.6, 75, 0)]),
     "Pick a toy for me to play with.",
     """say: Let me find you a toy.
plan: Ask whether the user wants the closest or the biggest toy, then go to it.
```python
toys = robot.where("toy")
if not toys:
    robot.say("I don't see any toys.")
else:
    answer = (robot.ask("Do you want the closest toy or the biggest one?") or "closest").lower()
    toy = max(toys, key=lambda t: t.radius) if "big" in answer else min(toys, key=lambda t: t.distance)
    robot.go_to(toy, distance=0.3)
    robot.say(f"Here is the {toy.color} {toy.label}.")
```"""),
    (_context("hall", [(6, "person", "", 1.0, 0, 0)], user=6, others="garage, kitchen"),
     "Go and see whether my bike is in the garage, then come back.",
     """say: I'll go and look for your bike.
plan: Save this spot, search the garage for the bike, and always come back to report.
```python
robot.save_place("start")
try:
    bike = robot.search("bike", area="garage")
finally:
    robot.go_to("start")
robot.say("Your bike is in the garage." if bike else "I couldn't find a bike in the garage.")
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

    def plan(self, context: str, instruction: str) -> PlanResult:
        self.messages = [*self.prefix, {"role": "user", "content": f"{context}\nInstruction: {instruction}"}]
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
