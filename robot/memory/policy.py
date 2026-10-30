"""Memory curation policy, subject normalization, and explicit command extraction."""

from __future__ import annotations

import re

from robot.memory.models import MemoryProposal, MemoryProvenance

_STOPWORDS = {
    "a",
    "an",
    "the",
    "that",
    "this",
    "my",
    "our",
    "your",
    "is",
    "are",
    "was",
    "were",
    "to",
    "be",
    "about",
    "please",
    "hey",
    "sprocket",
    "robot",
}

_REMEMBER_PATTERNS = [
    re.compile(r"^\s*(?:please\s+)?remember\s+(?:that\s+)?(.+?)\s*[.!?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:please\s+)?don'?t\s+forget\s+(?:that\s+)?(.+?)\s*[.!?]*\s*$", re.IGNORECASE),
]

_FORGET_PATTERNS = [
    re.compile(r"^\s*(?:please\s+)?forget\s+(?:that\s+|about\s+)?(.+?)\s*[.!?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:please\s+)?delete\s+memory\s+(?:of\s+|about\s+)?(.+?)\s*[.!?]*\s*$", re.IGNORECASE),
]


def infer_subject(text: str) -> str:
    """Extract a canonical subject key from a fact statement for contradiction/update resolution."""
    cleaned = text.strip().rstrip(".!?")
    # Match patterns like "my favorite color is green" or "My cat's name is Mochi". The model writes
    # "The user's favorite color is green": that must share the subject (and tombstones) of "my ...".
    m = re.match(
        r"^(?:that\s+)?(?:the\s+user'?s|my|our|the|user'?s?)\s+([a-z0-9 '_-]{2,40}?)\s+(?:is|are|was|equals)\s+.+$",
        cleaned,
        re.IGNORECASE,
    )
    if m:
        subj = re.sub(r"\s+", " ", m.group(1).strip().lower())
        return subj

    # Match preference statements like "I prefer green eye color" or "User prefers green eye color"
    m_pref = re.match(
        r"^(?:i|user)\s+(?:really\s+)?(?:prefer|prefers|love|loves|like|likes)\s+(.+)$",
        cleaned,
        re.IGNORECASE,
    )
    if m_pref:
        rest = m_pref.group(1).strip().lower()
        if "eye color" in rest or "eyes" in rest:
            return "eye color"
        tokens = [t for t in re.findall(r"[a-z0-9']+", rest) if t not in _STOPWORDS]
        return " ".join(tokens[-2:]) if tokens else "preference"

    # Match "<subject> is <value>"
    m2 = re.match(r"^([a-z0-9 '_-]{2,35}?)\s+(?:is|are)\s+.+$", cleaned, re.IGNORECASE)
    if m2:
        words = [w for w in m2.group(1).lower().split() if w not in _STOPWORDS]
        if words:
            return " ".join(words[:4])

    tokens = re.findall(r"[a-z0-9']+", cleaned.lower())
    meaningful = [t for t in tokens if t not in _STOPWORDS]
    return " ".join(meaningful[:4]) if meaningful else "general"


def normalize_fact_text(text: str) -> str:
    """Convert first-person user statements into clear third-person durable facts."""
    cleaned = text.strip().rstrip(".!?")
    cleaned = re.sub(r"^\s*that\s+", "", cleaned, flags=re.IGNORECASE)
    # "my favorite color is green" -> "User's favorite color is green"
    cleaned = re.sub(r"\bmy\b", "User's", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bi\s+like\b", "User likes", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bi\s+love\b", "User loves", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bi\s+prefer\b", "User prefers", cleaned, flags=re.IGNORECASE)
    return cleaned


def extract_explicit_memory_proposals(user_message: str) -> list[MemoryProposal]:
    """Detect explicit 'remember ...' or 'forget ...' instructions in a user message."""
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

    for pat in _REMEMBER_PATTERNS:
        m = pat.match(stripped)
        if m:
            raw_fact = m.group(1).strip()
            fact = normalize_fact_text(raw_fact)
            subj = infer_subject(raw_fact)
            return [
                MemoryProposal(
                    op="remember",
                    text=fact,
                    subject=subj,
                    provenance=MemoryProvenance.USER_EXPLICIT,
                    confidence=1.0,
                )
            ]

    return []
