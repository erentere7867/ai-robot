"""What things are called: matching labels, and the words people use to refer to things.
"""

from __future__ import annotations

import re

from robot.core.angles import wrap_deg

COLORS = {
    "red", "blue", "green", "yellow", "orange", "brown", "purple", "violet", "pink", "white", "black",
    "gray", "grey", "teal", "cyan", "tan", "beige", "gold", "silver", "magenta", "lavender",
}
# The VLM is not consistent about colour names (the same ball is "red" one time, "pink" the next).
COLOR_GROUPS = [{"brown", "tan", "beige", "orange", "cardboard", "gold"}, {"purple", "violet", "lavender", "magenta"},
                {"gray", "grey", "silver"}, {"teal", "cyan", "green"}, {"teal", "cyan", "blue"},
                {"red", "pink", "magenta", "crimson", "rose"}, {"yellow", "gold", "orange"}]
SYNONYMS = [
    {"box", "carton", "crate", "package", "cube", "block"},
    {"ball", "sphere", "orb"},
    {"mug", "cup", "cylinder", "can", "glass", "jar", "bottle"},
    {"duck", "duckling", "bird", "toy"},
    {"dock", "charger", "station", "pad", "base", "platform"},
    {"person", "human", "man", "woman", "people", "child", "kid"},
]
FILLER = {"a", "an", "the", "small", "large", "big", "little", "tiny", "toy", "object", "shaped", "colored",
          "coloured", "plastic", "wooden", "rubber", "my", "your", "that", "this", "of", "on", "with", "and"}
# Words for the human who is talking. The model is told the human is "me"; people say "come here".
SPEAKER_WORDS = {"me", "you", "user", "human", "the human", "the user", "person talking", "speaker", "here",
                 "over here", "to me", "the person talking to you", "whoever is talking"}
PERSON_WORDS = {"person", "human", "man", "woman", "people", "child", "kid"}
START_WORDS = {"start", "start point", "starting point", "home", "where i started", "where you started", "origin"}


# Referring to things by where they are or how they compare: "the thing behind you", "the closest box",
# "whichever one is nearest". Egocentric directions are (centre bearing, half width) in degrees, + = left.
_SECTORS = {"behind": (180.0, 65.0), "back": (180.0, 65.0), "left": (90.0, 60.0), "right": (-90.0, 60.0),
            "front": (0.0, 45.0), "ahead": (0.0, 45.0)}
_SUPERLATIVES = {"closest": "near", "nearest": "near", "farthest": "far", "furthest": "far", "biggest": "big",
                 "largest": "big", "smallest": "small", "tiniest": "small"}
_REFERRING = {"you", "me", "i", "of", "to", "in", "at", "side", "thing", "things", "one", "ones", "which",
              "whichever", "is", "are", "over", "there", "here", "it", "stuff", "item", "whatever", "something",
              "next", "now", "currently", "just", "right", "most"} | set(_SECTORS) | set(_SUPERLATIVES)


def _tokens(label: str) -> list[str]:
    return re.findall(r"[a-z]+", label.lower())


def _colors(tokens: list[str]) -> set[str]:
    found = {t for t in tokens if t in COLORS}
    if "cardboard" in tokens:
        found.add("brown")
    return found


def _nouns(tokens: list[str]) -> set[str]:
    return {t for t in tokens if t not in COLORS and t not in FILLER}


def _colors_compatible(a: set[str], b: set[str]) -> bool:
    if not a or not b or a & b:
        return True
    return any(a & g and b & g for g in COLOR_GROUPS)


def _nouns_compatible(a: set[str], b: set[str]) -> bool:
    if a & b:
        return True
    return any(a & g and b & g for g in SYNONYMS)


def labels_match(a: str, b: str) -> bool:
    return label_score(a, b) > 0


def _near_miss(a: str, b: str) -> bool:
    """One typo apart ("boz" for "box"): same first letter, one letter changed, added or dropped."""
    if a[0] != b[0] or min(len(a), len(b)) < 3 or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b, strict=True)) == 1
    short, long = sorted((a, b), key=len)
    return any(long[:i] + long[i + 1:] == short for i in range(len(long)))


def label_score(a: str, b: str) -> int:
    """2 = same kind of thing (shared noun), 1 = synonym (box~cube) or a typo of one, 0 = different;
    colours must agree."""
    ta, tb = _tokens(a), _tokens(b)
    if not _colors_compatible(_colors(ta), _colors(tb)):
        return 0
    na, nb = _nouns(ta), _nouns(tb)
    if na & nb:
        return 2
    if _nouns_compatible(na, nb):
        return 1
    return 1 if any(_near_miss(x, y) for x in na for y in nb) else 0


def parse_reference(text: str) -> tuple[str, tuple[float, float] | None, str | None]:
    """("box", sector, "near") for "the closest box behind you": what kind, where, which of several."""
    words = _tokens(text)
    sectors = [_SECTORS[w] for w in words if w in _SECTORS]
    superlative = next((_SUPERLATIVES[w] for w in words if w in _SUPERLATIVES), None)
    kind = " ".join(w for w in words if w not in _REFERRING and w not in FILLER)
    return kind, sectors[0] if len(sectors) == 1 else None, superlative


def _associates(track_label: str, det_label: str) -> bool:
    """Can this detection be this track? A catch-all "object" box can be anything (by position), including a
    thing the VLM has since named ("keys")."""
    return det_label == "object" or labels_match(track_label, det_label)


def head_noun(label: str) -> str:
    nouns = [t for t in _tokens(label) if t not in COLORS and t not in FILLER]
    return (nouns[-1] if nouns else "object")[:16]


def direction_words(rel_bearing_deg: float) -> str:
    a = wrap_deg(rel_bearing_deg)
    side = "left" if a > 0 else "right"
    b = abs(a)
    if b < 10:
        return "straight ahead"
    if b < 35:
        return f"ahead, slightly {side}"
    if b < 70:
        return f"front-{side}"
    if b < 110:
        return f"to my {side}"
    if b < 160:
        return f"behind-{side}"
    return "behind me"


def age_words(age_s: float) -> str:
    if age_s < 20:
        return "seen just now"
    if age_s < 90:
        return f"seen {age_s:.0f}s ago"
    if age_s < 5400:
        return f"seen {age_s / 60:.0f} min ago"
    return f"seen {age_s / 3600:.0f} h ago"
