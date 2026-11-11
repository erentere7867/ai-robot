"""One tracker in the map frame for all cameras: each object keeps its id as it passes from one camera to the next.

Detections arrive as (label, colour, map position, radius, score) from any camera. A detection joins the nearest
track inside a gate that has its label, or another label and a similar size: the detector names one thing
differently from different sides (a low coffee table is sometimes a "tv"), so a track's label is a vote of its
sightings weighted by score. People are kept apart from things. The track's position and size follow its sightings
smoothly; a detection near no track starts one, which counts as an object once it has been seen twice (a person seen
clearly counts at once). The vision model then looks at each new object's crop and names it (`verify`): the fast
detector proposes, the slower model decides, and an object it sees nothing in is dropped. It looks again when the
object is seen at least twice as large (`to_name`), so names from close views win. A track whose place a camera views clearly without
seeing it collects misses, and after three it is marked absent: negative evidence (Fable), so the robot stops
believing in things that have gone. People are tracked the same way and are free to move.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

SAME = {"couch": "sofa", "refrigerator": "fridge", "trash can": "bin", "garbage can": "bin", "potted plant": "plant",
        "houseplant": "plant", "cardboard box": "box", "mug": "cup", "television": "tv", "monitor": "tv",
        "rucksack": "backpack", "bag": "backpack", "sports ball": "ball", "dining table": "table",
        "coffee table": "table", "office chair": "chair", "armchair": "chair", "man": "person", "woman": "person"}


def canonical(label: str) -> str:
    """The planner's word for a detector's label."""
    label = label.lower().strip()
    return SAME.get(label, label)


@dataclass
class Detection:
    label: str
    x: float
    y: float
    radius: float
    color: str = ""
    score: float = 1.0
    camera: str = ""
    cut: bool = False  # cut by the picture's edge: its place is roughly right, its size says nothing


@dataclass
class Track:
    id: int
    label: str
    x: float
    y: float
    radius: float
    first_seen: float
    last_seen: float
    hits: int = 1
    misses: int = 0
    present: bool = True
    colors: Counter = field(default_factory=Counter)
    votes: Counter = field(default_factory=Counter)  # label -> summed detector score
    score: float = 0.0  # best detector score so far
    verified: bool = False  # the vision model has named it
    rejected: bool = False  # the vision model saw no object there
    named_from: int = 0  # pixels in the crop the vision model last named it from
    sized: bool = True  # False while it has only been seen cut by a picture's edge

    @property
    def color(self) -> str:
        return self.colors.most_common(1)[0][0] if self.colors else ""

    @property
    def confirmed(self) -> bool:
        return self.hits >= 2 or (self.person and self.score >= 0.5)

    @property
    def person(self) -> bool:
        return self.label == "person"


class Tracker:
    def __init__(self, gate: float = 0.6, smoothing: float = 0.35) -> None:
        self.gate, self.smoothing = gate, smoothing
        self.tracks: dict[int, Track] = {}
        self._next = 1

    def update(self, detections: list[Detection], now: float) -> list[tuple[Detection, Track]]:
        """Fold one camera frame's detections in; returns each detection with the track it matched or created."""
        touched: list[Track] = []
        pairs: list[tuple[Detection, Track]] = []
        for d in sorted(detections, key=lambda d: -d.score):
            label = canonical(d.label)
            best, best_gap = None, math.inf
            for t in self.tracks.values():
                same = t.label == label
                alike = d.cut or not t.sized or _alike(t.radius, d.radius)
                if t in touched or (not same and (t.person or label == "person" or not alike)):
                    continue
                gap = math.dist((t.x, t.y), (d.x, d.y))
                if same:
                    limit = max(self.gate, 1.5 * t.radius) * (2.5 if t.person else 1.0)  # people move between frames
                elif d.cut:  # the visible strip's middle can be a whole radius from the thing's own
                    limit = max(self.gate, t.radius + d.radius)
                else:  # another name for the same footprint: a backpack beside a chair stays a backpack
                    limit = max(t.radius, d.radius)
                if gap < limit and gap + (0.0 if same else 0.25) < best_gap:
                    best, best_gap = t, gap + (0.0 if same else 0.25)
            if best is None:
                best = Track(self._next, label, d.x, d.y, d.radius, now, now, sized=not d.cut)
                self.tracks[best.id] = best
                self._next += 1
            else:
                a = 0.7 if best.person else self.smoothing
                best.x += a * (d.x - best.x)
                best.y += a * (d.y - best.y)
                if not d.cut:  # a cut view says nothing of size; the first whole one sets it
                    best.radius += (self.smoothing if best.sized else 1.0) * (d.radius - best.radius)
                    best.sized = True
                best.hits += 1
                best.last_seen = now
            best.misses, best.present = 0, True
            best.votes[label] += d.score
            best.score = max(best.score, d.score)
            if not best.person and not best.verified:
                best.label = best.votes.most_common(1)[0][0]
            if d.color:
                best.colors[d.color] += 1
            touched.append(best)
            pairs.append((d, best))
        return pairs

    def missed(self, track: Track) -> None:
        """A camera looked clearly at where `track` should be and did not see it."""
        track.misses += 1
        if track.misses >= 3:
            track.present = False

    def verify(self, track: Track, label: str | None, pixels: int = 0) -> None:
        """The vision model's name for the track's crop of `pixels` pixels; None means it saw no object. People stay
        people: the detector is reliable on them, and they are tracked by other rules."""
        track.verified, track.named_from = True, pixels
        if label is None:
            track.rejected = True
        elif not track.person and canonical(label) != "person":
            track.label = canonical(label)

    def to_name(self, crops: dict[int, object]) -> Track | None:
        """The next object for the vision model to name: one never named, oldest first, else one now seen at least
        twice as large as when it was last named (a far backpack can look like a book; close up it does not)."""
        tracks = sorted((t for t in self.objects() if t.id in crops), key=lambda t: t.first_seen)
        fresh = next((t for t in tracks if not t.verified), None)
        return fresh or next((t for t in tracks if not t.person and pixels(crops[t.id]) >= 2 * t.named_from), None)

    def objects(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.confirmed and not t.rejected]


def pixels(image: object) -> int:
    """Pixels in an image array (height x width)."""
    return int(image.shape[0] * image.shape[1])


def _alike(a: float, b: float) -> bool:
    """Sizes close enough to be one thing named two ways (a cup on a table is not the table; the part of a table that
    one camera sees past the edge of its picture is)."""
    return max(a, b) <= 3.0 * max(min(a, b), 0.05)
