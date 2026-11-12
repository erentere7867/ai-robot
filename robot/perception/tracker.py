"""One tracker in the map frame for all cameras: each object keeps its id as it passes from one camera to the next.

Detections arrive as (label, colour, map position, radius, score) from any camera. A detection joins the nearest
track inside a gate that has its label, or another label and a similar size: the detector names one thing
differently from different sides (a low coffee table is sometimes a "tv"), so a track's label is a vote of its
sightings weighted by score. People are kept apart from things. A thing's place is the precision-weighted mean of its
sightings (a scalar Kalman filter: a depth reading counts more than a far floor estimate, and the place settles as
sightings add up, instead of being pulled a fixed share toward each new one, which made things twitch on the map).
A sighting's uncertainty includes the robot's own (localisation's, and timing while it turns): matching and merging
allow for it, so a sighting made while driving, a few tenths of a metre off, joins its thing instead of starting a
twin, and moves it little;
a person's place may move faster (WALKS), so it follows a walking person. Twins of one name left by renaming are
merged. A detection near no track starts one, which counts as an object once it has been seen twice (a person seen
clearly counts at once). The vision model then looks at each new object's crop and names it (`verify`): the fast
detector proposes, the slower model decides, and an object it sees nothing in is dropped. It looks again when the
object is seen at least twice as large (`to_name`), so names from close views win. A track whose place a camera views clearly without
seeing it collects misses, and after three it is marked absent: negative evidence (Fable), so the robot stops
believing in things that have gone. People are tracked the same way and are free to move; each carries the colours
of their clothes (`look`), so a person lost behind the furniture is known again when they reappear within walking
distance, and two people passing close, dressed differently, do not swap: "follow me" keeps its person.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SAME = {"couch": "sofa", "refrigerator": "fridge", "trash can": "bin", "garbage can": "bin", "potted plant": "plant",
        "houseplant": "plant", "cardboard box": "box", "mug": "cup", "television": "tv", "monitor": "tv",
        "rucksack": "backpack", "bag": "backpack", "sports ball": "ball", "dining table": "table",
        "coffee table": "table", "office chair": "chair", "armchair": "chair", "man": "person", "woman": "person"}
LIKE = 0.7  # how alike two looks must be (histogram intersection, 0 to 1) to know a lost person again
UNLIKE = 0.15  # below this, two looks are two people however close they stand (0.4 split one person across cameras)
MOVED = 0.001  # square metres per second a thing's place may drift unseen (someone may move it; misses handle more)
WALKS = 0.25  # the same for a person, who may walk at about half a metre a second


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
    sigma: float = 0.1  # one standard deviation of its place, metres (from how its range was found)
    look: Any = None  # a person's appearance: a normalised hue-saturation histogram of the clothes (`look`)


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
    var: float = 0.01  # variance of its place, square metres: shrinks as sightings agree
    look: Any = None  # a person's appearance, averaged over sightings, for knowing them again
    noise: float = 0.1  # how far its sightings usually scatter, metres (their sigma, averaged)
    sure: bool = False  # its name was confirmed by the vision model (a close-up rename turned a ball into a pillow)

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
                if t.person and t.look is not None and d.look is not None and \
                        float(np.minimum(t.look, d.look).sum()) < UNLIKE:
                    continue  # someone else, dressed differently: two people passing close do not swap
                gap = math.dist((t.x, t.y), (d.x, d.y))
                # how far apart one thing's places may honestly be; never over a metre, or neighbours run together
                unsure = min(1.0, 2.5 * math.sqrt(t.var + d.sigma ** 2))
                if same:
                    limit = max(self.gate, 1.5 * t.radius, unsure) * (2.5 if t.person else 1.0)  # people move
                elif d.cut:  # the visible strip's middle can be a whole radius from the thing's own
                    limit = max(self.gate, t.radius + d.radius, unsure)
                else:  # another name for the same footprint: a backpack beside a chair stays a backpack
                    limit = max(t.radius, d.radius, unsure)
                if gap < limit and gap + (0.0 if same else 0.25) < best_gap:
                    best, best_gap = t, gap + (0.0 if same else 0.25)
            if best is None and label == "person":
                best = self._known_again(d, now, touched)
            if best is None:
                best = Track(self._next, label, d.x, d.y, d.radius, now, now, sized=not d.cut, var=d.sigma ** 2,
                             noise=d.sigma)
                self.tracks[best.id] = best
                self._next += 1
            else:  # weigh each sighting by its precision; a thing settles as sightings add up, a person may walk on
                spread = best.var + (WALKS if best.person else MOVED) * max(0.0, now - best.last_seen)
                a = spread / (spread + d.sigma ** 2)
                best.var = (1 - a) * spread
                best.x += a * (d.x - best.x)
                best.y += a * (d.y - best.y)
                if not d.cut:  # a cut view says nothing of size; the first whole one sets it
                    best.radius += (self.smoothing if best.sized else 1.0) * (d.radius - best.radius)
                    best.sized = True
                best.hits += 1
                best.last_seen = now
            best.misses, best.present = 0, True
            best.noise = 0.8 * best.noise + 0.2 * d.sigma
            best.votes[label] += d.score
            best.score = max(best.score, d.score)
            if not best.person and not best.verified:
                best.label = best.votes.most_common(1)[0][0]
            if d.color:
                best.colors[d.color] += 1
            if d.look is not None:
                best.look = d.look if best.look is None else 0.8 * best.look + 0.2 * d.look
            touched.append(best)
            pairs.append((d, best))
        self._merge(pairs)
        return pairs

    def _known_again(self, d: Detection, now: float, touched: list[Track]) -> Track | None:
        """A person near no track who looks like someone lost from view within the last half-minute, and could have
        walked to where they are now: the same person, so "follow me" and "the user" keep them."""
        if d.look is None:
            return None
        best, best_likeness = None, LIKE
        for t in self.tracks.values():
            away = now - t.last_seen
            if not t.person or t in touched or t.look is None or not 0.3 < away < 30.0:
                continue
            if math.dist((t.x, t.y), (d.x, d.y)) > 1.0 + 1.5 * away:  # walking pace
                continue
            likeness = float(np.minimum(t.look, d.look).sum())  # histogram intersection, 1 for the same colours
            if likeness > best_likeness:
                best, best_likeness = t, likeness
        if best is not None:
            best.var = max(best.var, d.sigma ** 2)  # where they are now is new: let the sighting place them
        return best

    def _merge(self, pairs: list[tuple[Detection, Track]]) -> None:
        """One thing, one track: two tracks of the same name whose centres lie inside each other's footprint are
        merged into the older (a track renamed for a while can leave a twin behind); so are two things of about the
        same size, whatever their names, closer than their sightings' scatter allows (a sighting a few tenths of a
        metre off while driving, under another name, started a twin that later sightings fed). Two person tracks merge
        when they stand within half a metre and look alike: one person seen from two cameras in different light."""
        for t in sorted(self.tracks.values(), key=lambda t: t.id):
            if t.id not in self.tracks:
                continue
            for o in [o for o in self.tracks.values() if o.id > t.id and o.person == t.person]:
                gap = math.dist((t.x, t.y), (o.x, o.y))
                scatter = min(0.6, 2.5 * math.sqrt(t.var + o.var + max(t.noise, o.noise) ** 2))  # merging is surer to hurt
                if t.person:  # closer than 0.3 m is one person, however the cameras saw their clothes
                    alike = t.look is None or o.look is None or float(np.minimum(t.look, o.look).sum()) >= LIKE
                    if not (gap < 0.3 or (gap < 0.5 and alike)):
                        continue
                elif o.label == t.label:
                    if gap >= max(t.radius, o.radius, 0.15, scatter):
                        continue
                elif not (_alike(t.radius, o.radius) and gap < scatter):
                    continue
                w = o.var / (t.var + o.var)  # the more certain place counts more
                t.x, t.y = t.x + (1 - w) * (o.x - t.x), t.y + (1 - w) * (o.y - t.y)
                t.var = t.var * o.var / (t.var + o.var)
                t.hits += o.hits
                t.votes.update(o.votes)
                t.colors.update(o.colors)
                t.last_seen = max(t.last_seen, o.last_seen)
                t.score = max(t.score, o.score)
                del self.tracks[o.id]
                for k, (d, track) in enumerate(pairs):
                    if track is o:
                        pairs[k] = (d, t)

    def missed(self, track: Track) -> None:
        """A camera looked clearly at where `track` should be and did not see it."""
        track.misses += 1
        if track.misses >= 3:
            track.present = False

    def verify(self, track: Track, label: str | None, pixels: int = 0, sure: bool = False) -> None:
        """The vision model's name for the track's crop of `pixels` pixels; None means it saw no object. Its word
        wins either way: the detector called a backpack a person, which "follow me" could have followed. `sure`:
        the model confirmed the detector's own name (a yes or no), which is not asked again."""
        track.verified, track.named_from, track.sure = True, pixels, sure
        if label is None:
            track.rejected = True
        else:
            track.label = canonical(label)

    def to_name(self, crops: dict[int, object]) -> Track | None:
        """The next object for the vision model to name: one never named, oldest first, else one now seen at least
        twice as large as when it was last named (a far backpack can look like a book; close up it does not)."""
        tracks = sorted((t for t in self.objects() if t.id in crops), key=lambda t: t.first_seen)
        fresh = next((t for t in tracks if not t.verified), None)
        return fresh or next((t for t in tracks if not t.person and not t.sure
                              and pixels(crops[t.id]) >= 2 * t.named_from), None)

    def objects(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.confirmed and not t.rejected]


def pixels(image: object) -> int:
    """Pixels in an image array (height x width)."""
    return int(image.shape[0] * image.shape[1])


def _alike(a: float, b: float) -> bool:
    """Sizes close enough to be one thing named two ways: a cup on a table is not the table, and a ball (0.13 m)
    beside a chair (0.39 m) is not the chair (at three times, it was merged into it). A part cut by a picture's edge
    is matched by its own rule, its size saying nothing."""
    return max(a, b) <= 2.0 * max(min(a, b), 0.05)


def look(pixels: Any, bins: int = 8) -> Any:
    """A person's appearance from the middle of their box (clothes, not the background): a hue-saturation histogram
    of `bins` x `bins`, normalised to sum to 1. Brightness is left out, so a shadow does not change who someone is."""
    rgb = np.asarray(pixels, dtype=np.float32).reshape(-1, 3) / 255.0
    if rgb.shape[0] < 20:
        return None
    high, low = rgb.max(axis=1), rgb.min(axis=1)
    saturation = np.where(high > 0, (high - low) / np.maximum(high, 1e-6), 0.0)
    r, g, b = rgb.T
    span = np.maximum(high - low, 1e-6)
    hue = np.where(high == r, (g - b) / span % 6, np.where(high == g, (b - r) / span + 2, (r - g) / span + 4)) / 6
    hist, _, _ = np.histogram2d(hue, saturation, bins=bins, range=((0, 1), (0, 1)))
    return hist / hist.sum()

