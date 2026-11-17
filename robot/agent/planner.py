"""The closed-loop multimodal planner: receives scene summary + annotated images and emits structured Intent updates."""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass, replace
from typing import Any

from robot.agent.api import Intent
from robot.mind.brain import Brain, Reply, _with_image

SYSTEM_PROMPT = """You are Sprocket, a warm, friendly home companion robot in the room.
Speak like a friend in brief, casual, natural sentences (e.g., "Hey! What's up?", "Yo! What's on your mind?", "Hi there — how can I help?", "On it, heading to the table."). NEVER use stiff robotic boilerplate ("Hello! I'm Sprocket v2...") or invent unseen objects. When Earlier: already has a reply, vary your phrasing so you do not repeat the exact same sentence.

Grounding & Reasoning:
- Rely strictly on Surroundings:, Known objects:, Route notes / facts:, Completed Steps in Current Task:, and camera observations; never invent objects, counts, or IDs.
- In Surroundings: and Known objects:, desk(table) = table, sofa(couch) = couch, person(user) = user. Always read the exact numeric id from Known objects:; never copy example IDs.

Action & Output Format (exact 7 plain lines — say, chime, posture, action, target, params, subgoal; no markdown or XML):
- Postures: neutral, listening, curious, happy, uh_oh, yielding | Chimes: wake, listen, curious, happy, uh_oh, yield, none
- Greetings, chat, or visual/spatial questions ("what do you see?", "what's around/behind you?", "where is X?"): answer directly and naturally in say: using Surroundings: and Known objects: (mention colors, objects, and where they are — ahead, beside, or behind you) with action: done, target: none, params: {}. NEVER move, turn, or use search_frontier on a greeting or question.
- go_to (known object, user, or room): target: <numeric id|user|room>, params: {"distance": 0.5}. NEVER use move or search_frontier for a known target.
- Relative motion (target: none): move ({"metres": 1.0, "direction": "forward"|"backward"}), turn ({"degrees": 90, "direction": "left"|"right"}), spin ({"turns": 1|2, "direction": "left"|"right"}).
- orbit: target: <id|name>, params: {"laps": 2, "direction": "right"|"left", "distance": 0.5}; pursue: target: user, params: {"distance": 1.0}.
- search_frontier: ONLY when the target is NOT in Known objects:. Also available: face, peek, step_closer, look_under, ask, remember, motion_sequence.
- refuse: impossible or unsafe requests (flying, stairs, heavy lifting, closed doors) -> explain in say:, action: refuse, target: none, params: {}.

Multi-Step Completion:
- Read Completed Steps in Current Task: first. NEVER repeat any listed step. Do only the next uncompleted step; once Completed Steps in Current Task: satisfies Instruction:, immediately emit action: done, target: none, params: {}.

Examples:
Instruction: how's it going?
Completed Steps in Current Task:
(none)
say: Doing great! What are we up to?
chime: wake
posture: happy
action: done
target: none
params: {}
subgoal: chat with user

Instruction: what do you see?
Surroundings: ahead: grey desk(table) (1.2 m); behind: you (person) (2.0 m)
Completed Steps in Current Task:
(none)
say: There is a grey desk right ahead, and you are behind me about two metres back.
chime: curious
posture: curious
action: done
target: none
params: {}
subgoal: describe surroundings

Instruction: go to the mug
Known objects: id=9 mug (2.4 m front)
Completed Steps in Current Task:
(none)
say: On it, heading over to the mug.
chime: listen
posture: curious
action: go_to
target: mug
params: {"distance": 0.5}
subgoal: approach target

Instruction: back up one metre, then spin twice
Completed Steps in Current Task:
- Step 1: move target=none params={"direction": "backward", "metres": 1.0} (completed)
- Step 2: spin target=none params={"direction": "left", "turns": 2} (completed)
say: All done — backed up and spun around!
chime: happy
posture: happy
action: done
target: none
params: {}
subgoal: completed multi-step motion"""

GRAMMAR = r"""root ::= "say: " say-text "\nchime: " chime "\nposture: " posture "\naction: " action "\ntarget: " target "\nparams: " json-obj "\nsubgoal: " subgoal-text
say-first ::= [a-zA-Z0-9]
say-char ::= [^\r\n"<>`:;{}\[\]\\]
say-text ::= (say-first say-char{0,109})?
subgoal-char ::= [^\r\n"<>`:;{}\[\]\\]
subgoal-text ::= [a-zA-Z0-9] subgoal-char{0,48}
chime ::= "wake" | "listen" | "curious" | "happy" | "uh_oh" | "yield" | "none" | ""
posture ::= "neutral" | "listening" | "curious" | "happy" | "uh_oh" | "yielding"
action ::= "go_to" | "pursue" | "orbit" | "search_frontier" | "peek" | "step_closer" | "look_under" | "face" | "move" | "turn" | "spin" | "motion_sequence" | "ask" | "remember" | "done" | "refuse"
target ::= "none" | [a-zA-Z0-9_()][a-zA-Z0-9_() -]*
json-ws ::= [ \t]*
json-str ::= "\"" ([^"\\\r\n] | "\\" ["\\/bfnrt])* "\""
json-num ::= "-"? [0-9]+ ("." [0-9]+)?
json-bool ::= "true" | "false" | "null"
json-val ::= json-str | json-num | json-bool | json-arr | json-obj
json-arr ::= "[" json-ws (json-val ("," json-ws json-val)*)? json-ws "]"
json-kv ::= json-str json-ws ":" json-ws json-val
json-obj ::= "{" json-ws (json-kv ("," json-ws json-kv)*)? json-ws "}"
"""


class FormatError(ValueError):
    """The model's answer did not have the required Intent fields."""


@dataclass
class PlanResult:
    intent: Intent
    reply: Reply
    say_first: str = ""
    plan: str = ""

    def __post_init__(self) -> None:
        if not self.say_first:
            self.say_first = self.intent.say
        if not self.plan:
            self.plan = self.intent.subgoal or self.intent.action


_VALID_CHIMES = frozenset({"wake", "listen", "curious", "happy", "uh_oh", "yield"})
_VALID_POSTURES = frozenset({"neutral", "listening", "curious", "happy", "uh_oh", "yielding"})
_POSTURE_ALIASES = {
    "alert": "curious",
    "focused": "curious",
    "sad": "uh_oh",
    "error": "uh_oh",
    "relaxed": "neutral",
}
_DONE_ACTIONS = frozenset({"", "none", "say", "answer", "greet", "chat", "respond", "reply", "speak"})
_NULL_TARGETS = frozenset({"", "none", "null", "n/a", "na", "nothing", "self"})
_USER_TARGETS = frozenset({"user", "the user", "person(user)", "me"})


def _clean_text(val: Any) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    # Remove leaked XML/special token tags and markdown code fence tails.
    s = re.sub(r"<\|[^|>]*\|>", "", s)
    s = re.sub(r"</?[a-zA-Z_][^>\n]{0,30}>", "", s)
    s = re.sub(r"```.*", "", s).strip()
    s = re.split(
        r"(?:;\s*|\b)(?:chime|posture|action|target|params|subgoal)\s*:",
        s,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    return s


def _extract_earlier_replies(earlier: tuple[str, ...] | list[str]) -> list[str]:
    """Extract Sprocket's previously spoken replies from `Earlier:` history entries."""
    replies: list[str] = []
    for line in earlier:
        text = str(line)
        found = re.findall(r'\bsaid\s+"([^"]+)"', text)
        if not found:
            stripped = re.sub(r'^"[^"]+"\s*(?::|->)\s*', "", text.strip())
            found = re.findall(r'"([^"]+)"', stripped)
        replies.extend(found)
    return replies


def _supports_sampling_kwargs(fn: Any) -> bool:
    """Return True if `fn` accepts both `temperature` and `presence_penalty` keyword arguments."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    return has_var_kw or ("temperature" in params and "presence_penalty" in params)


_GREETING_INSTRUCTIONS = frozenset(
    {
        "hey",
        "hi",
        "hello",
        "yo",
        "sup",
        "good morning",
        "good evening",
        "howdy",
        "hey there",
        "hi there",
        "how's it going",
        "how are you",
        "how're you",
        "what's up",
        "whats up",
        "who are you",
    }
)
_FALLBACK_GREETINGS = (
    "Hey! What's up?",
    "Hi there — how can I help?",
    "Yo! What's on your mind?",
)
_MOTION_PHRASE_PREFIXES = (
    "on it",
    "heading ",
    "driving ",
    "going to ",
    "moving ",
    "turning ",
    "spinning ",
    "searching ",
)
_NON_MOVING_ACTIONS = frozenset({"done", "ask", "remember", "refuse"})
_ACTION_COMMAND_RE = re.compile(
    r"\b(?:go\s+to|go|move|drive|head|walk|come|turn|spin|rotate|orbit|circle|follow|pursue|"
    r"search|look\s+for|look\s+under|peek|step|approach|bring|fetch|take|back\s+up|face)\b",
    re.I,
)
_VISUAL_QUERY_RE = re.compile(
    r"\b(?:"
    r"what(?:\s+\w+){0,2}\s+(?:do|can)\s+(?:you|u)\s+see|"
    r"what(?:'s|\s+is|\s+are)\s+(?:around|near(?:by)?|in\s+front|ahead|behind|beside|next\s+to|to\s+(?:the|your)\s+(?:left|right)|here|in\s+(?:the|your)\s+room|over\s+there)|"
    r"(?:see|spot)\s+anything|anything\s+(?:ahead|behind|around|nearby|there)|"
    r"(?:do|can)\s+(?:you|u)\s+see\b|"
    r"(?:is|are)\s+there\s+(?:a|an|any)\b|"
    r"(?:where|which\s+way)(?:'s|\s+is|\s+are)\b|"
    r"(?:describe|tell\s+me\s+about)\s+(?:what\s+(?:you|u)\s+see|(?:your\s+)?surroundings|(?:the\s+)?room|around)|"
    r"who(?:'s|\s+is)\s+(?:here|there|around|behind|with|in\s+front)"
    r")\b",
    re.I,
)
_QUERY_STOPWORDS = frozenset(
    {
        "a", "an", "and", "any", "are", "around", "can", "do", "here", "in", "is", "me",
        "my", "near", "nearby", "of", "on", "right", "room", "see", "spot", "tell",
        "the", "there", "way", "what", "whats", "what's", "where", "wheres", "where's",
        "which", "who", "you", "your",
    }
)
_QUERY_SYNONYMS = {"table": "desk", "desk": "table", "couch": "sofa", "sofa": "couch"}
_SIDE_REGION_MAP = {
    "ahead": "ahead",
    "front": "ahead",
    "front-left": "ahead",
    "front-right": "ahead",
    "left": "sides",
    "right": "sides",
    "behind": "behind",
    "behind-left": "behind",
    "behind-right": "behind",
    "rear": "behind",
    "rear-left": "behind",
    "rear-right": "behind",
    "back": "behind",
}


def _is_visual_query(instruction: str) -> bool:
    low = instruction.strip().lower().rstrip("!.?, ")
    if not low or _ACTION_COMMAND_RE.search(low):
        return False
    return bool(_VISUAL_QUERY_RE.search(low))


def _is_conversational_or_visual_query(instruction: str) -> bool:
    low = instruction.strip().lower().rstrip("!.?, ")
    if not low:
        return False
    if low in _GREETING_INSTRUCTIONS:
        return True
    return _is_visual_query(instruction)


def _format_visual_item(item: str, inst_low: str) -> str:
    """Turn a synthesized item like 'grey desk(table) (0.9 m)' into natural speech text."""
    s = item.strip()
    if s.startswith("you (person)"):
        return re.sub(r"^you \(person\)", "you", s)
    return re.sub(
        r"([a-zA-Z0-9_]+)\(([^)]+)\)",
        lambda m: (
            m.group(2)
            if m.group(2).lower() in inst_low and m.group(2).lower() != "person"
            else m.group(1)
        ),
        s,
    )


def _parse_context_surroundings(context: str, inst_low: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {"ahead": [], "sides": [], "behind": []}
    m = re.search(r"(?m)^Surroundings:\s*(.+)$", context)
    if m:
        body = m.group(1).strip()
        if body and body != "(nothing detected yet)":
            for chunk in body.split(";"):
                if ":" not in chunk:
                    continue
                region, items_str = chunk.split(":", 1)
                rkey = region.strip().lower()
                if rkey in groups:
                    for raw in items_str.split(","):
                        cleaned = _format_visual_item(raw, inst_low)
                        if cleaned:
                            groups[rkey].append(cleaned)
        if any(groups.values()):
            return groups

    # Fallback: parse Known objects: lines directly when no Surroundings: summary is present.
    in_known = False
    for line in context.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("known objects:"):
            in_known = True
            continue
        if not in_known:
            continue
        if not stripped:
            break
        if "(none" in stripped.lower():
            continue
        dir_m = re.search(
            r"\b(ahead|front-left|front-right|front|left|right|behind-left|behind-right|behind|rear-left|rear-right|rear|back)\b",
            stripped,
            re.I,
        )
        rkey = _SIDE_REGION_MAP.get(dir_m.group(1).lower(), "ahead") if dir_m else "ahead"
        dist_m = re.search(r"(\d+(?:\.\d+)?)\s*m\b", stripped)
        dist_txt = f" ({dist_m.group(1)} m)" if dist_m else ""
        lbl_m = re.search(r"(?:id\s*[=:]\s*-?\d+|-?\d+)\s+([a-zA-Z0-9_()-]+)", stripped)
        lbl = lbl_m.group(1) if lbl_m else "object"
        lbl = _format_visual_item(lbl.replace("_", " "), inst_low)
        color_m = re.search(r"\b(?:color=|\[)([a-zA-Z]+)", stripped)
        color = color_m.group(1) if color_m and color_m.group(1).lower() != "unknown" else ""
        if color and "you" not in lbl and not lbl.lower().startswith(color.lower()):
            lbl = f"{color} {lbl}"
        groups[rkey].append(f"{lbl}{dist_txt}")
    return groups


def _join_natural(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _synthesize_visual_answer(instruction: str, context: str) -> str:
    """Build a grounded spoken answer for visual/spatial questions strictly from `context`."""
    inst_low = instruction.strip().lower().rstrip("!.?, ")
    groups = _parse_context_surroundings(context, inst_low)
    all_items = [(region, it) for region, items in groups.items() for it in items]

    if not all_items:
        return "I don't see anything detected around me yet."

    # Specific target lookup ("where is the desk?", "do you see a mug?", "who is here?")
    if re.search(r"\b(?:where|do\s+you\s+see|can\s+you\s+see|is\s+there|are\s+there|who)\b", inst_low):
        terms = [
            w for w in re.findall(r"[a-z]+", inst_low) if w not in _QUERY_STOPWORDS and len(w) > 1
        ]
        if "who" in inst_low and "you" not in terms:
            terms.extend(["you", "person"])
        expanded = list(terms)
        for t in terms:
            if t in _QUERY_SYNONYMS:
                expanded.append(_QUERY_SYNONYMS[t])
        for region, item in all_items:
            ilow = item.lower()
            if any(re.search(rf"\b{re.escape(t)}\b", ilow) for t in expanded):
                place = {"ahead": "ahead of me", "sides": "off to the side", "behind": "behind me"}[region]
                return f"I see {item} {place}."
        if terms and "where" in inst_low:
            summary = "; ".join(
                f"{r}: {_join_natural(items[:2])}" for r, items in groups.items() if items
            )
            return f"I don't see that right now — around me I see {summary}."

    if "behind" in inst_low:
        return (
            f"Behind me I see {_join_natural(groups['behind'])}."
            if groups["behind"]
            else "I don't see anything behind me right now."
        )
    if "ahead" in inst_low or "in front" in inst_low:
        return (
            f"Ahead of me I see {_join_natural(groups['ahead'])}."
            if groups["ahead"]
            else "I don't see anything directly ahead of me right now."
        )
    if any(w in inst_low for w in ("left", "right", "beside", "side")):
        return (
            f"Off to the side I see {_join_natural(groups['sides'])}."
            if groups["sides"]
            else "I don't see anything off to my sides right now."
        )

    clauses: list[str] = []
    if groups["ahead"]:
        clauses.append(f"ahead of me there's {_join_natural(groups['ahead'][:3])}")
    if groups["sides"]:
        clauses.append(f"off to the side there's {_join_natural(groups['sides'][:3])}")
    if groups["behind"]:
        clauses.append(f"behind me there's {_join_natural(groups['behind'][:3])}")
    text = "; ".join(clauses)
    return text[0].upper() + text[1:] + "."


_GENERIC_SUBGOALS = frozenset({"", "approach target", "go to target", "drive to target"})
_TARGET_SUBGOAL_ACTIONS = frozenset({"go_to", "orbit", "peek", "look_under", "face", "step_closer"})
_LABEL_SYNONYM_HINTS = {"desk": "table", "sofa": "couch"}


def _pick_object_label(raw_label: str, inst_low: str) -> str:
    m = re.match(r"^([^(]+?)(?:\(([^)]+)\))?$", raw_label.strip())
    if not m:
        return raw_label.strip()
    base = m.group(1).strip()
    alias = (m.group(2) or "").strip()
    if alias and re.search(rf"\b{re.escape(alias.lower())}\b", inst_low):
        return alias.lower()
    if base.lower() == "person" and alias.lower() == "user":
        return "user"
    syn = _LABEL_SYNONYM_HINTS.get(base.lower())
    if syn and re.search(rf"\b{re.escape(syn)}\b", inst_low):
        return syn
    return base


def _ground_subgoal(intent: Intent, instruction: str, context: str) -> Intent:
    """Replace generic or example-echoed subgoals (e.g. 'approach target', 'drive to mug') with actual target."""
    if intent.action not in _TARGET_SUBGOAL_ACTIONS or intent.target is None:
        return intent

    inst_low = instruction.lower()
    sg_low = intent.subgoal.strip().lower()
    tgt_str = str(intent.target).strip()
    tgt_low = tgt_str.lower()
    has_mug_echo = "mug" in sg_low and "mug" not in inst_low and "mug" not in tgt_low
    if sg_low not in _GENERIC_SUBGOALS and not has_mug_echo:
        return intent

    label = ""
    if tgt_low == "user":
        label = "user"
    elif isinstance(intent.target, int) or re.fullmatch(r"-?\d+", tgt_str):
        tid = re.escape(tgt_str)
        ctx_match = re.search(rf"\bid\s*[=:]\s*{tid}\s+([a-zA-Z0-9_()]+)", context)
        if not ctx_match:
            ctx_match = re.search(rf"(?m)^\s*{tid}\s+([a-zA-Z0-9_()]+)", context)
        if ctx_match:
            label = _pick_object_label(ctx_match.group(1), inst_low)
        else:
            phrase_match = re.search(
                r"\b(?:to|around|at|toward|near|under|face)\s+"
                r"(?:(?:the|a|an|my|that|this|red|blue|brown|yellow|green|white|black|gray|grey|big|small|nearest)\s+)*"
                r"([a-zA-Z]+)\b",
                inst_low,
            )
            label = phrase_match.group(1) if phrase_match else f"object {tgt_str}"
    else:
        label = _pick_object_label(tgt_str, inst_low)

    if not label:
        return intent
    verb = intent.action.replace("_", " ")
    return replace(intent, subgoal=f"{verb} {label}")


def _normalize_chime(val: Any) -> str:
    low = _clean_text(val).lower()
    return low if low in _VALID_CHIMES else ""


def _normalize_posture(val: Any) -> str:
    low = _clean_text(val).lower()
    if low in _VALID_POSTURES:
        return low
    return _POSTURE_ALIASES.get(low, "neutral")


def _normalize_action(val: Any) -> str:
    low = _clean_text(val).lower().replace("-", "_").replace(" ", "_")
    return "done" if low in _DONE_ACTIONS else low


def _parse_target(val: Any) -> int | str | None:
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val
    s = _clean_text(val)
    low = s.lower()
    if low not in _USER_TARGETS:
        s = re.sub(r"\s*\([^)]*\)", "", s).strip()
        low = s.lower()
    if not s or low in _NULL_TARGETS:
        return None
    if low in _USER_TARGETS:
        return "user"
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    m = re.match(r"^(?:id\s*[=:]?\s*|#\s*)(-?\d+)(?:\s+.*)?$", s, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.match(r"^(-?\d+)\s+[a-zA-Z_].*$", s)
    if m:
        return int(m.group(1))
    m = re.match(r"^.*\s+#(-?\d+)$", s)
    if m:
        return int(m.group(1))
    return s


def parse_intent(text: str) -> Intent:
    """Parses line-structured or JSON output into a normalized Intent dataclass."""
    data: dict[str, Any] = {
        "say": "",
        "chime": "",
        "posture": "neutral",
        "action": "done",
        "target": None,
        "params": {},
        "subgoal": "",
    }

    # First attempt: JSON parse if wrapped in JSON
    trimmed = text.strip()
    if trimmed.startswith("{") and trimmed.endswith("}"):
        try:
            parsed = json.loads(trimmed)
            if isinstance(parsed, dict):
                data.update(parsed)
                return Intent(
                    say=_clean_text(data.get("say", "")),
                    chime=_normalize_chime(data.get("chime", "")),
                    posture=_normalize_posture(data.get("posture", "neutral")),
                    action=_normalize_action(data.get("action", "done")),
                    target=_parse_target(data.get("target")),
                    params=data.get("params", {}) if isinstance(data.get("params"), dict) else {},
                    subgoal=_clean_text(data.get("subgoal", "")),
                )
        except Exception:
            pass

    # Line-by-line field parsing
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, val = line.split(":", 1)
        k = key.strip().lower()
        v = val.strip()
        if k == "say":
            data["say"] = _clean_text(v)
        elif k == "chime":
            data["chime"] = _normalize_chime(v)
        elif k == "posture":
            data["posture"] = _normalize_posture(v)
        elif k == "action":
            data["action"] = _normalize_action(v)
        elif k == "target":
            data["target"] = _parse_target(v)
        elif k == "params":
            v_clean = re.sub(r"```.*", "", v).strip()
            if v_clean and v_clean.startswith("{") and v_clean.endswith("}"):
                try:
                    loaded = json.loads(v_clean)
                    data["params"] = loaded if isinstance(loaded, dict) else {}
                except Exception:
                    data["params"] = {}
            else:
                data["params"] = {}
        elif k == "subgoal":
            data["subgoal"] = _clean_text(v)

    return Intent(
        say=_clean_text(data["say"]),
        chime=_normalize_chime(data["chime"]),
        posture=_normalize_posture(data["posture"]),
        action=_normalize_action(data["action"]),
        target=_parse_target(data["target"]),
        params=data["params"] if isinstance(data["params"], dict) else {},
        subgoal=_clean_text(data["subgoal"]),
    )


class LocalPlanner:
    """The on-board closed-loop planner constraining LFM3-VL-2B to compact Intent outputs."""

    def __init__(
        self,
        brain: Brain | None = None,
        max_tokens: int = 160,
        think: int = 0,
        multimodal: bool = False,
    ) -> None:
        self.brain = brain or Brain()
        self.max_tokens = max_tokens
        self.think = think
        self.multimodal = multimodal
        self.messages: list[dict[str, Any]] = []

    def plan_step(
        self,
        context: str,
        instruction: str,
        image_url: str | None = None,
        scratchpad: str = "",
        earlier: tuple[str, ...] | list[str] = (),
        completed_steps: tuple[str, ...] | list[str] = (),
    ) -> Intent:
        """Plans the next immediate Intent step given multimodal observation and memory."""
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Scene State:\n{context}\n"
        )
        if scratchpad and "Blockers: [none]" not in scratchpad:
            prompt += f"Working Memory Scratchpad:\n{scratchpad}\n"
        if earlier:
            prompt += "Earlier:\n" + "\n".join(f"- {line}" for line in earlier) + "\n"
        completed_block = "\n".join(f"- {step}" for step in completed_steps) if completed_steps else "(none)"
        prompt += (
            f"\nInstruction: {instruction}\n"
            f"Completed Steps in Current Task:\n{completed_block}\n"
            f"Next Intent:"
        )

        if image_url:
            messages = _with_image(image_url, prompt)
        else:
            messages = [{"role": "user", "content": prompt}]

        self.messages = messages
        temp, penalty = (0.35, 0.5) if earlier else (0.0, 0.0)
        reply, supports_sampling = self._complete_with_sampling(
            messages, temperature=temp, presence_penalty=penalty
        )
        intent = parse_intent(reply.text)

        prev_said = {s.strip().lower() for s in _extract_earlier_replies(earlier) if s.strip()}
        if earlier and supports_sampling and prev_said and intent.say.strip().lower() in prev_said:
            retry_reply, _ = self._complete_with_sampling(
                messages, temperature=0.65, presence_penalty=0.8
            )
            intent = parse_intent(retry_reply.text)

        if _is_conversational_or_visual_query(instruction):
            say_low = intent.say.strip().lower()
            motion_say = any(say_low.startswith(p) for p in _MOTION_PHRASE_PREFIXES) or not say_low
            tried_to_move = (
                intent.action not in _NON_MOVING_ACTIONS or intent.target is not None or motion_say
            )
            if _is_visual_query(instruction):
                if tried_to_move:
                    answer = _synthesize_visual_answer(instruction, context)
                    intent = replace(
                        intent,
                        say=answer,
                        chime="curious",
                        posture="curious",
                        action="done",
                        target=None,
                        params={},
                        subgoal="describe surroundings",
                    )
                elif intent.action == "done":
                    intent = replace(intent, target=None, params={}, subgoal="describe surroundings")
            elif tried_to_move:
                say = intent.say
                if motion_say:
                    say = next((g for g in _FALLBACK_GREETINGS if g.lower() not in prev_said), _FALLBACK_GREETINGS[0])
                intent = replace(intent, say=say, action="done", target=None, params={}, subgoal="greet user")

        return _ground_subgoal(intent, instruction, context)

    def _complete_with_sampling(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> tuple[Reply, bool]:
        supports_sampling = _supports_sampling_kwargs(self.brain.complete)
        if supports_sampling and (temperature != 0.0 or presence_penalty != 0.0):
            reply = self.brain.complete(
                messages,
                GRAMMAR,
                self.max_tokens,
                self.think,
                temperature=temperature,
                presence_penalty=presence_penalty,
            )
            return reply, True
        return self.brain.complete(messages, GRAMMAR, self.max_tokens, self.think), supports_sampling

    def decide_step(
        self,
        context: str,
        instruction: str,
        image_url: str | None = None,
        scratchpad: str = "",
        earlier: tuple[str, ...] | list[str] = (),
        completed_steps: tuple[str, ...] | list[str] = (),
    ) -> Intent:
        """Alias for plan_step."""
        return self.plan_step(
            context,
            instruction,
            image_url=image_url,
            scratchpad=scratchpad,
            earlier=earlier,
            completed_steps=completed_steps,
        )

    def plan(
        self,
        context: str,
        instruction: str,
        earlier: tuple[str, ...] | list[str] = (),
    ) -> PlanResult:
        """Backwards-compatible plan interface returning PlanResult."""
        intent = self.plan_step(context, instruction, earlier=earlier)
        reply = Reply(text=f"action: {intent.action}")
        return PlanResult(intent=intent, reply=reply)

    def repair(self, problem: str, context: str | None = None) -> PlanResult:
        """Attempts an intent repair when an action fails."""
        prompt = f"The previous action encountered a problem: {problem}.\n"
        if context:
            prompt += f"Current State:\n{context}\n"
        prompt += "Provide an alternative Intent to recover or resolve:"
        self.messages.append({"role": "user", "content": prompt})
        reply = self.brain.complete(self.messages, GRAMMAR, self.max_tokens, self.think)
        intent = parse_intent(reply.text)
        return PlanResult(intent=intent, reply=reply)
