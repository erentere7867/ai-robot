from dataclasses import dataclass
from enum import Enum, auto


class Skill(Enum):
    IDLE = auto()
    SEARCHING = auto()
    NAVIGATING = auto()
    FOLLOWING = auto()
    WAITING = auto()
    FAILED = auto()


@dataclass(frozen=True)
class Goal:
    skill: Skill
    target: str = ""
    region: str = ""
    timeout: float | None = None


class SkillMachine:
    def __init__(self) -> None:
        self.state = Skill.IDLE
        self.goal: Goal | None = None

    @property
    def busy(self) -> bool:
        return self.state not in (Skill.IDLE, Skill.FAILED)

    def start(self, goal: Goal) -> None:
        if self.busy:
            raise RuntimeError("A skill is already running")
        if goal.skill in (Skill.IDLE, Skill.FAILED):
            raise ValueError("A goal must name a running skill")
        self.goal = goal
        self.state = goal.skill

    def finish(self, ok: bool) -> None:
        self.state = Skill.IDLE if ok else Skill.FAILED
        self.goal = None
