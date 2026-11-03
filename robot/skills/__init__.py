"""Skills the language model can invoke as tools."""

from robot.skills import navigation, perception, social
from robot.skills.base import Skill, SkillContext, SkillError, SkillRegistry, SkillResult


def build_default_registry(eye_colors: list[str], *, ask_camera: bool = False) -> SkillRegistry:
    """`ask_camera`: add a tool for questions about the view (when the decision model cannot see images)."""
    registry = SkillRegistry()
    # Order matters to a small model: the first tools listed get picked more often.
    for skill in [*navigation.skills(), *social.skills(eye_colors), *perception.skills(ask=ask_camera)]:
        registry.add(skill)
    return registry


__all__ = ["Skill", "SkillContext", "SkillError", "SkillRegistry", "SkillResult", "build_default_registry"]
