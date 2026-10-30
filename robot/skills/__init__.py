"""Skills the language model can invoke as tools."""

from robot.skills import navigation, perception, social
from robot.skills.base import Skill, SkillContext, SkillError, SkillRegistry, SkillResult


def build_default_registry(eye_colors: list[str]) -> SkillRegistry:
    registry = SkillRegistry()
    for skill in [*perception.skills(), *navigation.skills(), *social.skills(eye_colors)]:
        registry.add(skill)
    return registry


__all__ = ["Skill", "SkillContext", "SkillError", "SkillRegistry", "SkillResult", "build_default_registry"]
