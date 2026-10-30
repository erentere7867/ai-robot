"""Memory curation policy: fact normalization, subject inference, validation, and extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass

from robot.memory.models import MemoryProposal, MemoryProvenance

NAME_SUBJECT = "name"
MAX_FACT_CHARS = 300
# Snake-case tool identifiers are never durable facts; plain-word tools are rejected only on their own.
DEFAULT_RESERVED_TERMS = frozenset(
    {
        "perform_trick",
        "tilt_head",
        "set_gaze",
        "set_eye_color",
        "raw_gpio_write",
        "schedule_wake",
        "robot_response",
        "drive",
        "spin",
        "stop",
        "observe",
    }
)

_STOPWORDS = {
    "a", "an", "the", "that", "this", "my", "our", "your", "is", "are", "was", "were", "to", "be", "about",
    "please", "hey", "sprocket", "robot", "i", "me", "user", "user's", "of", "in", "on", "for", "with", "and",
    "or", "it", "really", "just", "also", "very", "am", "i'm", "im", "has", "have",
}  # fmt: skip

_NAME_SYNONYMS = {
    "name", "user name", "username", "first name", "full name", "preferred name", "nickname", "name of user",
    "name of the user",
}  # fmt: skip

_REMEMBER_PATTERNS = [
    re.compile(r"^\s*(?:please\s+)?remember\s+(?:that\s+)?(.+?)\s*[.!?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:please\s+)?don'?t\s+forget\s+(?:that\s+)?(.+?)\s*[.!?]*\s*$", re.IGNORECASE),
]

_FORGET_PATTERNS = [
    re.compile(r"^\s*(?:please\s+)?forget\s+(?:that\s+|about\s+)?(.+?)\s*[.!?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:please\s+)?delete\s+memory\s+(?:of\s+|about\s+)?(.+?)\s*[.!?]*\s*$", re.IGNORECASE),
]

# Leading greetings/fillers; `(?![\w'])` keeps possessives such as "Robot's" intact.
_LEAD_FILLER = re.compile(
    r"^(?:(?:hi|hello|hey|well|so|ok|okay|oh|also|and|but|btw|by\s+the\s+way|actually|fyi|that|robot|sprocket|"
    r"just\s+so\s+you\s+know)(?![\w'])[\s,!.:;-]*)+",
    re.IGNORECASE,
)
# Words that end (or rule out) a name: "call me at 5", "I go by bus", "my name is very common".
_NAME_STOP = {
    "a", "an", "the", "and", "but", "or", "so", "because", "please", "thanks", "thank", "nice", "what", "whats",
    "what's", "how", "who", "why", "where", "when", "i", "im", "i'm", "you", "your", "yours", "not", "from",
    "by", "to", "too", "also", "btw", "okay", "ok", "robot", "sprocket", "later", "back", "if", "now", "tomorrow",
    "today", "anytime", "sometime", "maybe", "soon", "once", "again", "crazy", "is", "it", "that", "this",
    "since", "here", "there", "anything", "something", "whatever", "tonight", "after", "before",
    # prepositions, pronouns, and determiners
    "at", "on", "in", "up", "down", "out", "off", "over", "under", "into", "onto", "for", "with", "without",
    "about", "around", "via", "per", "as", "like", "than", "every", "each", "all", "any", "some", "no", "my",
    "his", "her", "their", "our", "its", "me", "him", "them", "us", "we", "they", "he", "she", "whenever",
    "ever", "right", "only", "more", "less", "first", "next", "last", "asap", "immediately", "sometimes",
    # times of day and weekdays
    "morning", "evening", "afternoon", "night", "weekend", "weekends", "noon", "midnight", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday",
    # "I go by bus" and "I go by my gut"
    "bus", "car", "train", "foot", "bike", "bicycle", "plane", "taxi", "cab", "subway", "metro", "boat", "ferry",
    "tram", "motorcycle", "scooter", "walking", "myself", "ear", "gut", "instinct", "feel",
    # descriptions rather than names: "my name is very common", "call me old-fashioned"
    "very", "quite", "pretty", "kind", "sort", "hard", "long", "short", "common", "unusual", "difficult", "easy",
    "funny", "weird", "strange", "lazy", "silly", "stupid", "paranoid", "old-fashioned", "names", "spelled",
    "spelt", "pronounced", "written",
}  # fmt: skip
# Fillers skipped before a name: "my name is actually Deniz".
_NAME_FILLER = {"actually", "just", "really", "simply", "officially", "now", "still"}
_NAME_WORD = re.compile(r"[^\W\d_][\w'.-]*")
_VALUE = (
    r"(?P<v>[^,.;!?]+?)\s*(?=$|[,.;!?]|\s+(?:and|but|with|since|because|for|so|though|now|after|until|while|"
    r"when|where)\b)"
)
_NAME_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:my|user'?s)\s+name\s*(?:is|=)\s+(?P<v>.+)", re.IGNORECASE), "User's name is {}"),
    (re.compile(r"\bmy\s+name's\s+(?P<v>.+)", re.IGNORECASE), "User's name is {}"),
    (re.compile(r"\b(?:i\s+am|i'm|user\s+is)\s+called\s+(?P<v>.+)", re.IGNORECASE), "User's name is {}"),
    (
        re.compile(
            r"(?:^|[,.;:!?]\s*|\b(?:hi|hello|hey|and|so|ok|okay|also|just|please|to|you\s+can|you\s+may|"
            r"(?:can|could|would|will)\s+you(?:\s+please)?)\s+)call\s+me\s+(?P<v>.+)",
            re.IGNORECASE,
        ),
        "User wants to be called {}",
    ),
    (re.compile(r"\b(?:i|user)\s+(?:go|goes)\s+by\s+(?P<v>.+)", re.IGNORECASE), "User wants to be called {}"),
    (re.compile(r"\buser\s+wants\s+to\s+be\s+called\s+(?P<v>.+)", re.IGNORECASE), "User wants to be called {}"),
]
# (pattern, subject template, fact template); `{n}` subjects carry a proper-name value.
_PERSONAL_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(rf"\b(?:i|user)\s+(?:currently\s+|now\s+|still\s+)?lives?\s+in\s+{_VALUE}", re.IGNORECASE),
        "residence",
        "User lives in {v}",
    ),
    (
        re.compile(rf"\b(?:i\s+am|i'm|user\s+is)\s+(?:originally\s+)?from\s+{_VALUE}", re.IGNORECASE),
        "hometown",
        "User is from {v}",
    ),
    (re.compile(rf"\b(?:i|user)\s+works?\s+as\s+{_VALUE}", re.IGNORECASE), "job", "User works as {v}"),
    (re.compile(rf"\b(?:i|user)\s+works?\s+(?:at|for)\s+{_VALUE}", re.IGNORECASE), "workplace", "User works at {v}"),
    (
        re.compile(
            r"\b(?:i|user)\s+(?:have|has|got|owns?)\s+(?:a|an|one)\s+(?:[a-z-]+\s+){0,2}?(?P<n>[a-z]+)\s+"
            r"(?:named|called)\s+(?P<v>.+)",
            re.IGNORECASE,
        ),
        "{n}'s name",
        "User has a {n} named {v}",
    ),
    (
        re.compile(r"\b(?:my|user'?s)\s+(?P<n>[a-z]+)\s+is\s+(?:named|called)\s+(?P<v>.+)", re.IGNORECASE),
        "{n}'s name",
        "User's {n} is named {v}",
    ),
    (
        re.compile(r"\b(?:i\s+am|i'm|user\s+is)\s+(?P<v>\d{1,3})\s+years?\s+old\b", re.IGNORECASE),
        "age",
        "User is {v} years old",
    ),
]
_NEGATION_BEFORE = re.compile(
    r"(?:\b(?:not|never|dont|if|suppose|pretend|imagine|assume|whether|stop)\b|n't)[^,.;!?]*$", re.IGNORECASE
)
_PREFERENCE = re.compile(
    r"^(?:i|user)\s+(?:really\s+|also\s+|absolutely\s+|totally\s+)?"
    r"(?:prefer|prefers|love|loves|like|likes|enjoy|enjoys|hate|hates|dislike|dislikes|"
    r"(?:don't|do\s+not|doesn't|does\s+not)\s+like)\s+(.+)$",
    re.IGNORECASE,
)
_PRESENT_VERBS = {
    "live", "work", "like", "love", "prefer", "enjoy", "hate", "dislike", "want", "need", "play", "study", "speak",
    "drink", "eat", "own", "use", "drive", "wake", "sleep", "get", "take", "watch", "listen", "think", "know",
    "believe", "feel", "cook", "run", "walk", "swim", "ride", "teach", "collect", "support", "visit", "wear",
    "keep", "hope", "plan", "practice", "train", "come", "make", "read", "write", "sing", "paint", "draw",
    "build", "code", "program", "travel", "commute", "share", "spend", "go", "do", "have", "fix", "grow", "miss",
    "rent", "bake", "hike", "climb", "dance", "attend", "follow", "learn", "sell", "buy", "pay", "earn", "save",
    "fish", "garden", "knit", "sew", "jog", "cycle", "ski", "surf", "dive", "camp", "shop", "clean", "wash",
    "feed", "care", "call", "text", "host", "lead", "coach", "design", "develop", "repair", "translate", "act",
    "volunteer", "babysit", "manage", "meditate", "pray", "stream", "tutor", "bike", "skate", "sail", "row",
}  # fmt: skip
_IRREGULAR_VERBS = {"am": "is", "have": "has", "do": "does", "go": "goes", "was": "was", "were": "was"}
_ADVERB = r"(?:really|also|usually|always|never|often|sometimes|still|actually|absolutely|totally|mostly)"

_INSTRUCTION_PATTERNS = [
    re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass)\b.{0,40}?\b(?:instructions?|prompts?|rules?|directives?|"
        r"guidelines?|guardrails?|safety|constraints?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:system|developer)\s+(?:prompt|message|instructions?)\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\b|\bdeveloper\s+mode\b|\bjailbreak", re.IGNORECASE),
    re.compile(r"\bnew\s+(?:instructions?|rules?|persona)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:system|assistant|developer|tool)\s*:", re.IGNORECASE),
    re.compile(r"<\|[^>]*\|>|\[/?INST\]|</?(?:system|tool_call|tool|assistant)>", re.IGNORECASE),
    re.compile(r"[{}`]"),
    re.compile(r"\b[A-Za-z_]\w*\([^)]*\)"),
    re.compile(r"\b(?:call|use|run|execute|invoke|trigger)\s+(?:the\s+)?(?:tool|function|command)\b", re.IGNORECASE),
]
# Snake-case identifiers look like tool calls; the lookahead spares e-mail addresses such as john_doe@x.com.
_SNAKE_IDENTIFIER = re.compile(r"\b[a-z]{2,}(?:_[a-z]{2,})+\b(?!@)")
_QUESTION_START = re.compile(
    r"^(?:what|who|whom|whose|where|when|why|how|which)(?:'s|\s+(?:is|are|was|were|do|does|did|can|could|"
    r"should|would|will|has|have)\b)",
    re.IGNORECASE,
)
_YES_NO_START = re.compile(
    r"^(?:is|are|do|does|did|can|could|would|will|should|have|has|am|was|were)\s+(?:i|you|my|we|it|there)\b",
    re.IGNORECASE,
)
_COMMAND_VERBS = {
    "go", "come", "turn", "look", "drive", "spin", "stop", "move", "find", "follow", "tell", "show", "say", "play",
    "dance", "sing", "make", "do", "bring", "get", "take", "put", "open", "close", "wait", "sit", "help", "check",
    "remember", "forget", "delete", "call", "give", "try", "start", "keep", "roll", "rotate", "face", "search",
    "explore", "return", "charge", "wake", "sleep", "be", "set", "use", "let", "lets", "let's", "watch", "listen",
    "hold", "pick", "push", "fetch", "avoid", "guard", "patrol", "can", "could", "would", "will", "don't", "dont",
    "never", "please", "back", "head", "tilt", "nod", "shake",
}  # fmt: skip
_TRANSIENT_WORDS = {
    "tired", "hungry", "thirsty", "sleepy", "bored", "sad", "angry", "upset", "sick", "ill", "sore", "busy", "late",
    "early", "ready", "done", "finished", "gone", "here", "there", "back", "home", "away", "out", "cold", "hot",
    "warm", "wet", "dirty", "broken", "dead", "low", "empty", "full", "missing", "lost", "stuck", "open", "closed",
    "on", "off", "charging", "running", "killing", "hurting", "aching", "fine", "okay", "ok", "great", "bad",
    "good", "better", "worse", "coming", "going", "leaving", "waiting", "sleeping", "working", "eating", "cooking",
    "visiting", "not", "almost", "about",
}  # fmt: skip
_TIME_MARKERS = {
    "today", "tonight", "now", "currently", "yesterday", "tomorrow", "this", "still", "again", "anymore",
    "already", "soon", "later", "right",
}  # fmt: skip
# Attribute subjects are durable whatever the value; relation subjects still reject transient states.
_ATTRIBUTE_KEYWORDS = {
    "name", "favorite", "favourite", "birthday", "job", "occupation", "hometown", "age", "email", "number",
    "address", "allergy", "allergies", "pronouns", "language", "team", "school", "university", "major", "hobby",
    "hobbies",
}  # fmt: skip
_RELATION_KEYWORDS = {
    "wife", "husband", "partner", "son", "daughter", "mom", "mother", "dad", "father", "brother", "sister", "kid",
    "kids", "child", "children", "family", "friend", "boss", "pet", "cat", "dog",
}  # fmt: skip
_HYPOTHETICAL_START = re.compile(
    r"^(?:if|suppose|imagine|pretend|assume|what\s+if|when|whenever|unless)\b", re.IGNORECASE
)
_PRONOUN_OBJECTS = {
    "it", "this", "that", "these", "those", "you", "him", "her", "them", "here", "there", "everything",
    "nothing", "something", "when", "how", "what", "us",
}  # fmt: skip
_IMPLICIT_MY_FACT = re.compile(r"\b((?:my|our)\s+[a-z0-9 '_-]{2,35}?\s+(?:is|are)\s+[^.!?,;]{2,50})", re.IGNORECASE)
_IMPLICIT_PREFERENCE = re.compile(
    r"\b(i\s+(?:really\s+|also\s+|absolutely\s+|totally\s+)?(?:prefer|love|like|enjoy|hate|dislike|"
    r"(?:don't|do\s+not)\s+like)\s+[^.!?,;]{2,50})",
    re.IGNORECASE,
)
_IMPLICIT_HOME = re.compile(
    r"\b(the\s+(?:charging\s+dock|home\s+base|kitchen|living\s+room|office|bedroom|bathroom|stairs)\s+"
    r"(?:is|are)\s+[^.!?,;]{2,50})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FactCandidate:
    """Durable fact extracted from a user statement, ready to store."""

    subject: str
    text: str
    provenance: MemoryProvenance = MemoryProvenance.MODEL_PROPOSED
    importance: float = 0.5


def canonical_subject(subject: str) -> str:
    """Normalize a subject key so equivalent facts supersede each other."""
    subj = re.sub(r"\s+", " ", subject.strip().lower()).strip(" .,:;!?\"")
    subj = re.sub(r"^(?:(?:the|my|our|user's|users|user)\s+)+", "", subj)
    subj = re.sub(r"\bfavourite\b", "favorite", subj)
    subj = re.sub(r"\bcolour\b", "color", subj)
    if subj in _NAME_SYNONYMS:
        return NAME_SUBJECT
    return subj[:160]


def default_importance(subject: str, provenance: MemoryProvenance | str) -> float:
    """Importance floor: names 1.0, user-explicit facts 0.8, everything else 0.5."""
    if canonical_subject(subject) == NAME_SUBJECT:
        return 1.0
    if str(provenance) == MemoryProvenance.USER_EXPLICIT:
        return 0.8
    return 0.5


def _strip_lead(text: str) -> str:
    return _LEAD_FILLER.sub("", text.strip()).strip()


def _clean_name(raw: str) -> tuple[str | None, str]:
    """Return `(capitalized name of up to 3 words, unconsumed remainder)`."""
    tokens = raw.split()
    while tokens and tokens[0].strip(".,;:!?\"()").lower() in _NAME_FILLER:
        tokens = tokens[1:]
    words: list[str] = []
    for token in tokens:
        word = token.strip(".,;:!?\"()")
        if not word or word.lower() in _NAME_STOP or not _NAME_WORD.fullmatch(word):
            break
        words.append(word if any(ch.isupper() for ch in word) else word[:1].upper() + word[1:])
        if len(words) == 3 or token[-1:] in ".,;:!?":
            break
    return (" ".join(words) or None), " ".join(tokens[len(words) :])


def _clean_value(raw: str) -> str | None:
    value = re.sub(r"\s+", " ", raw).strip(" '\"")
    if not 2 <= len(value) <= 60 or value.split()[0].lower() in _PRONOUN_OBJECTS | {"not"}:
        return None
    return value


def _negated(text: str, start: int) -> bool:
    return bool(_NEGATION_BEFORE.search(text[max(0, start - 40) : start]))


def match_personal_fact(text: str, *, anchored: bool = False) -> tuple[str, str, bool] | None:
    """Return `(subject, third-person fact, covers_rest)` for durable personal statements (name, home, pets...).

    `anchored` requires the statement to start the text; `covers_rest` is True when nothing follows the value.
    """
    for pattern, template in _NAME_PATTERNS:
        matches = [pattern.match(text)] if anchored else list(pattern.finditer(text))
        for m in matches:
            if m is None or _negated(text, m.start()):
                continue
            name, rest = _clean_name(m.group("v"))
            if name:
                return NAME_SUBJECT, template.format(name), not rest.strip(" .!?")
    for pattern, subject_tpl, fact_tpl in _PERSONAL_PATTERNS:
        m = pattern.match(text) if anchored else pattern.search(text)
        if not m or _negated(text, m.start()):
            continue
        noun = (m.groupdict().get("n") or "").lower()
        if "{n}" in subject_tpl:
            if noun in _PRONOUN_OBJECTS or noun == "name":
                continue
            value, rest = _clean_name(m.group("v"))
        else:
            value, rest = _clean_value(m.group("v")), text[m.end() :]
        if value:
            return subject_tpl.format(n=noun), fact_tpl.format(n=noun, v=value), not rest.strip(" .!?")
    return None


def match_name(text: str) -> tuple[str, str] | None:
    """Detect 'my name is X' / 'call me X' anywhere in `text` and return `(subject, third-person fact)`."""
    for pattern, template in _NAME_PATTERNS:
        for m in pattern.finditer(text):
            if _negated(text, m.start()):
                continue
            name, _rest = _clean_name(m.group("v"))
            if name:
                return NAME_SUBJECT, template.format(name)
    return None


def infer_subject(text: str) -> str:
    """Extract a canonical subject key from a fact statement for contradiction/update resolution."""
    cleaned = _strip_lead(text).rstrip(".!?")
    personal = match_personal_fact(cleaned, anchored=True)
    if personal:
        return personal[0]
    m_robot = re.match(r"^(?:your|robot'?s)\s+([a-z0-9 '_-]{2,40}?)\s+(?:is|are|was)\s+.+$", cleaned, re.IGNORECASE)
    if m_robot:
        return "robot's " + canonical_subject(m_robot.group(1))
    # Match patterns like "my favorite color is green" or "My cat's name is Mochi"
    m = re.match(
        r"^(?:my|our|the|user'?s?)\s+([a-z0-9 '_-]{2,40}?)\s+(?:is|are|was|equals)\s+.+$",
        cleaned,
        re.IGNORECASE,
    )
    if m:
        return canonical_subject(m.group(1))

    # Match preference statements like "I prefer green eye color" or "User doesn't like spiders"
    m_pref = _PREFERENCE.match(cleaned)
    if m_pref:
        rest = m_pref.group(1).strip().lower()
        if "eye color" in rest or "eyes" in rest:
            return "eye color"
        tokens = [t for t in re.findall(r"[a-z0-9']+", rest) if t not in _STOPWORDS]
        return " ".join(tokens[-2:]) if tokens else "preference"

    # Match "<subject> is <value>"; "User is <predicate>" keys on the predicate so facts do not collide.
    m2 = re.match(r"^([a-z0-9 '_-]{2,35}?)\s+(?:is|are)\s+(.+)$", cleaned, re.IGNORECASE)
    if m2:
        words = [w for w in m2.group(1).lower().split() if w not in _STOPWORDS]
        if words:
            return canonical_subject(" ".join(words[:4]))
        predicate = [t for t in re.findall(r"[a-z0-9']+", m2.group(2).lower()) if t not in _STOPWORDS]
        if predicate:
            return " ".join(predicate[:3])

    tokens = re.findall(r"[a-z0-9']+", cleaned.lower())
    meaningful = [t for t in tokens if t not in _STOPWORDS]
    return canonical_subject(" ".join(meaningful[:4])) if meaningful else "general"


def _third_person(match: re.Match[str]) -> str:
    adverb, verb = match.group("adv") or "", match.group("verb")
    lower = verb.lower()
    if lower in _IRREGULAR_VERBS:
        verb = _IRREGULAR_VERBS[lower]
    elif lower in _PRESENT_VERBS:
        if re.search(r"(?:s|sh|ch|x|z|o)$", lower):
            verb += "es"
        elif re.search(r"[^aeiou]y$", lower):
            verb = verb[:-1] + "ies"
        else:
            verb += "s"
    return f"User {adverb}{verb}"


def _sub_pronoun(pattern: str, replacement: str, text: str) -> str:
    """Rewrite a pronoun unless it is capitalized mid-sentence, as in titles ("My Chemical Romance")."""

    def repl(m: re.Match[str]) -> str:
        sentence_start = not text[: m.start()].strip() or bool(re.search(r"[.!?:;]\s*$", text[: m.start()]))
        return replacement if sentence_start or m.group(0)[:1].islower() else m.group(0)

    return re.sub(pattern, repl, text, flags=re.IGNORECASE)


def normalize_fact_text(text: str) -> str:
    """Convert first-person user statements into clear third-person durable facts."""
    cleaned = _strip_lead(text).rstrip(".!? ")
    personal = match_personal_fact(cleaned, anchored=True)
    if personal and personal[2]:
        return personal[1]
    # "my favorite color is green" -> "User's favorite color is green"
    cleaned = re.sub(r"\bmy\s+name's\b", "my name is", cleaned, flags=re.IGNORECASE)
    cleaned = _sub_pronoun(r"\b(?:my|mine)\b", "User's", cleaned)
    cleaned = _sub_pronoun(r"\bcall\s+me\b", "call User", cleaned)
    cleaned = re.sub(r"\b(?:i\s+am|i'm)(?![\w'])", "User is", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bi've\b", "User has", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bi'd\b", "User would", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bi'll\b", "User will", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bi\s+(?:don't|do\s+not)\b", "User doesn't", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(rf"\bi\s+(?P<adv>{_ADVERB}\s+)?(?P<verb>[a-z]+)\b", _third_person, cleaned, flags=re.IGNORECASE)
    cleaned = _sub_pronoun(r"\b(?:me|myself)\b", "User", cleaned)
    cleaned = _sub_pronoun(r"\b(?:you\s+are|you're)(?![\w'])", "Robot is", cleaned)
    cleaned = _sub_pronoun(r"\byour\b", "Robot's", cleaned)
    return cleaned


def canonical_fact(text: str, subject: str | None = None) -> tuple[str, str]:
    """Return `(subject, third-person content)`; recognized personal patterns override the subject hint.

    Without a hint the subject is inferred from the third-person content, so "I'm vegetarian" and the
    model's "User is vegetarian" share one key and supersede each other.
    """
    cleaned = _strip_lead(text).rstrip(".!? ")
    content = normalize_fact_text(cleaned)
    personal = match_personal_fact(cleaned, anchored=True) or match_personal_fact(content, anchored=True)
    if personal:
        return personal[0], content
    if subject and subject.strip():
        return canonical_subject(subject), content
    return canonical_subject(infer_subject(content)), content


def is_question(text: str) -> bool:
    stripped = _strip_lead(text)
    return stripped.endswith("?") or bool(_QUESTION_START.match(stripped) or _YES_NO_START.match(stripped))


def is_instruction_like(text: str, reserved_terms: frozenset[str] | set[str] = DEFAULT_RESERVED_TERMS) -> bool:
    """True for prompt-injection phrasing, role markers, code/JSON, or tool identifiers."""
    if any(p.search(text) for p in _INSTRUCTION_PATTERNS) or _SNAKE_IDENTIFIER.search(text):
        return True
    lowered = text.lower().strip(" .!")
    for term in reserved_terms:
        term = term.lower()
        if lowered == term or ("_" in term and re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE)):
            return True
    return False


def fact_rejection_reason(
    text: str,
    *,
    max_chars: int = MAX_FACT_CHARS,
    reserved_terms: frozenset[str] | set[str] = DEFAULT_RESERVED_TERMS,
) -> str | None:
    """Return why `text` must not be stored as a durable fact, or None when acceptable."""
    stripped = text.strip()
    if not stripped:
        return "empty"
    if len(stripped) > max_chars:
        return "too_long"
    if is_instruction_like(stripped, reserved_terms):
        return "instruction_like"
    if is_question(stripped):
        return "question"
    if len(re.findall(r"[^\W_]+", stripped)) < 2:
        return "not_a_fact"
    return None


def _is_command(clause: str) -> bool:
    words = clause.split(maxsplit=1)
    first = words[0].lower().strip(",.!") if words else ""
    return first in _COMMAND_VERBS or bool(re.match(r"^(?:you|robot)\b(?!')", clause, re.IGNORECASE))


def _is_transient(value: str) -> bool:
    words = re.findall(r"[a-z']+", value.lower())
    if not words or words[0] in _TRANSIENT_WORDS:
        return True
    return any(w in _TIME_MARKERS for w in words)


def _clause_candidates(clause: str) -> list[FactCandidate]:
    if _is_command(clause) or _HYPOTHETICAL_START.match(clause):
        return []
    personal = match_personal_fact(clause)
    if personal:
        importance = 1.0 if personal[0] == NAME_SUBJECT else 0.8
        return [FactCandidate(personal[0], personal[1], MemoryProvenance.USER_EXPLICIT, importance)]
    found: list[FactCandidate] = []
    for m in _IMPLICIT_MY_FACT.finditer(clause):
        snippet = m.group(1).strip().rstrip(".!?,;")
        value = re.split(r"\s+(?:is|are)\s+", snippet, maxsplit=1, flags=re.IGNORECASE)[-1]
        subj = infer_subject(snippet)
        subject_words = set(re.findall(r"[a-z]+", subj))
        attribute = bool(subject_words & _ATTRIBUTE_KEYWORDS)
        durable = attribute or bool(subject_words & _RELATION_KEYWORDS)
        if subj == NAME_SUBJECT or _negated(clause, m.start()) or not value.split():
            continue  # Names come only from the stricter name patterns.
        if value.split()[0].lower() == "not" or (not attribute and _is_transient(value)):
            continue
        provenance = MemoryProvenance.USER_EXPLICIT if durable else MemoryProvenance.MODEL_PROPOSED
        found.append(FactCandidate(subj, normalize_fact_text(snippet), provenance, 0.8 if durable else 0.5))
    for m in _IMPLICIT_PREFERENCE.finditer(clause):
        snippet = m.group(1).strip().rstrip(".!?,;")
        obj = _PREFERENCE.match(snippet)
        words = re.findall(r"[a-z']+", obj.group(1).lower()) if obj else []
        if not words or words[0] in _PRONOUN_OBJECTS or any(w in _TIME_MARKERS for w in words):
            continue
        found.append(
            FactCandidate(infer_subject(snippet), normalize_fact_text(snippet), MemoryProvenance.USER_EXPLICIT, 0.6)
        )
    for m in _IMPLICIT_HOME.finditer(clause):
        snippet = m.group(1).strip().rstrip(".!?,;")
        found.append(FactCandidate(infer_subject(snippet), normalize_fact_text(snippet), importance=0.6))
    return found


def extract_fact_candidates(user_text: str) -> list[FactCandidate]:
    """Extract durable personal facts from a statement, skipping questions, commands, and transient states."""
    candidates: list[FactCandidate] = []
    seen: set[str] = set()
    for sentence in re.findall(r"[^.!?;]+[.!?;]*", user_text):
        question = is_question(sentence.strip())
        for clause in re.split(r",|\s+(?:and|but)\s+", sentence.strip().rstrip(".!?;"), flags=re.IGNORECASE):
            clause = _strip_lead(clause)
            if not clause:
                continue
            named = match_name(clause)
            if named:
                found = [FactCandidate(named[0], named[1], MemoryProvenance.USER_EXPLICIT, 1.0)]
            elif question or is_question(clause):
                continue
            else:
                found = _clause_candidates(clause)
            for cand in found:
                subj = cand.subject.strip().lower()
                if len(subj) < 2 or subj in seen or subj in {"general", "preference"}:
                    continue
                if fact_rejection_reason(cand.text):
                    continue
                seen.add(subj)
                candidates.append(cand)
    return candidates


def extract_explicit_memory_proposals(user_message: str) -> list[MemoryProposal]:
    """Detect explicit 'remember ...', 'forget ...', 'my name is ...' or 'call me ...' in a user message."""
    stripped = user_message.strip()
    # Strip optional leading wake word like "Sprocket, remember..."
    stripped = re.sub(
        r"^\s*(?:hey\s+)?(?:sprocket|robot)\s*[,:]?\s*",
        "",
        stripped,
        flags=re.IGNORECASE,
    )

    for pat in _FORGET_PATTERNS:
        m = pat.match(stripped)
        if m:
            topic = m.group(1).strip()
            subj = infer_subject(topic)
            return [
                MemoryProposal(
                    op="forget",
                    text=topic,
                    subject=subj,
                    provenance=MemoryProvenance.USER_EXPLICIT,
                    confidence=1.0,
                )
            ]

    # "Remember when we ...?" is a question, not a memory write.
    if not stripped.endswith("?"):
        for pat in _REMEMBER_PATTERNS:
            m = pat.match(stripped)
            if m:
                subj, fact = canonical_fact(m.group(1).strip())
                return [
                    MemoryProposal(
                        op="remember",
                        text=fact,
                        subject=subj,
                        provenance=MemoryProvenance.USER_EXPLICIT,
                        confidence=1.0,
                    )
                ]

    named = match_name(stripped)
    if named:
        return [
            MemoryProposal(
                op="remember",
                text=named[1],
                subject=NAME_SUBJECT,
                provenance=MemoryProvenance.USER_EXPLICIT,
                confidence=1.0,
            )
        ]
    return []
