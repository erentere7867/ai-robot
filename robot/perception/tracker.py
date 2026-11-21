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
from itertools import combinations
from typing import Any

import numpy as np

SAME = {"couch": "sofa", "refrigerator": "fridge", "trash can": "bin", "garbage can": "bin", "potted plant": "plant",
        "houseplant": "plant", "cardboard box": "box", "mug": "cup", "television": "tv", "monitor": "tv",
        "rucksack": "backpack", "bag": "backpack", "sports ball": "ball", "dining table": "table",
        "coffee table": "table", "office chair": "chair", "armchair": "chair", "man": "person", "woman": "person",
        "bicycle": "bike", "bean bag": "beanbag"}
LIKE = 0.7  # how alike two looks must be (histogram intersection, 0 to 1) to know a lost person again
UNLIKE = 0.15  # below this, two looks are two people however close they stand (0.4 split one person across cameras)
MOVED = 0.001  # square metres per second a thing's place may drift unseen (someone may move it; misses handle more)
WALKS = 0.25  # the same for a person, who may walk at about half a metre a second
_FURNITURE_ALIKE = frozenset({"desk", "table", "box", "chair", "sofa", "tv"})


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
    width: float = 0.0
    depth_extent: float = 0.0
    orientation_yaw: float = 0.0
    pose: Any = None  # HumanPose
    visual_vec: np.ndarray | None = None
    voice_id: str | None = None
    box: tuple[int, int, int, int] | None = None
    normal: tuple[float, float] | None = None  # unit normal to the camera's sightline in the map


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
    vx: float = 0.0
    vy: float = 0.0
    width: float = 0.0
    depth_extent: float = 0.0
    orientation_yaw: float = 0.0
    pose: Any = None  # HumanPose
    visual_vec: np.ndarray | None = None
    voice_id: str | None = None
    cut: bool = False
    apart: set[int] = field(default_factory=set, repr=False, compare=False)
    camera: str = ""
    normal: tuple[float, float] | None = None
    visual_color: str = ""  # named from the object crop when a request actually needs its colour
    color_from: int = 0
    visual: bool = False  # placed by the vision model, not the detector: detector misses are not evidence it left
    z: float = 0.0  # median elevation of a visual target, metres (>0.45 is above camera eye level, e.g. on a desk)
    spotted_from: tuple[float, float] = (0.0, 0.0)  # robot map (x, y) when a visual target was placed
    def predict(self, dt: float) -> tuple[float, float]:
        """Predicted 2D position (x, y) at dt seconds into the future."""
        return self.x + self.vx * dt, self.y + self.vy * dt

    @property
    def color(self) -> str:
        return self.visual_color or (self.colors.most_common(1)[0][0] if self.colors else "")

    @property
    def confirmed(self) -> bool:
        return self.hits >= 2 or (self.person and self.score >= 0.5)

    @property
    def person(self) -> bool:
        return self.label == "person"


GONE_AFTER = 6  # clear views in a row without it before a thing counts as gone: YOLOE-26l finds a TV in about 73% of
# frames, so three misses in a row came often with the TV right there ("tv 1 was not there when the cameras last
# looked"); six in a row at that rate is about 1 in 2,600
GONE_S = 3.0  # and seconds since it was last seen


class Tracker:
    def __init__(self, gate: float = 0.6, smoothing: float = 0.35) -> None:
        self.gate, self.smoothing = gate, smoothing
        self.tracks: dict[int, Track] = {}
        self._next = 1
        self.became: dict[int, int] = {}  # merged id -> the id it was merged into: ids already given out keep working

    def current(self, track_id: int) -> int:
        """The id a thing has now, given any id it has had."""
        while track_id in self.became:
            track_id = self.became[track_id]
        return track_id

    def update(self, detections: list[Detection], now: float) -> list[tuple[Detection, Track]]:
        """Fold one camera frame's detections in; returns each detection with the track it matched or created."""
        touched: list[Track] = []
        pairs: list[tuple[Detection, Track]] = []
        parts = []
        for d in sorted(detections, key=lambda d: -d.score):
            label = canonical(d.label)
            reserved = None
            if d.box is not None and label != "person":
                # A leg can have a better detector score than the whole seat. Don't let it consume the
                # one whole-object match, move its centre or shrink the footprint used on the next frame.
                nearby = [t for t in self.tracks.values() if t.label == label and t.confirmed and t.present
                          and not t.rejected and now - t.last_seen <= GONE_S
                          and math.hypot(t.x - d.x, t.y - d.y) <= t.radius + min(t.radius, d.sigma)
                          and not _across_seam(d, t)]
                if len(nearby) == 1 and nearby[0].sized and 2 * d.radius < nearby[0].radius:
                    t = nearby[0]
                    # A whole, separate outline of the parent in this frame is evidence for two objects,
                    # even when their map footprints nearly touch (e.g. a small bag beside a suitcase).
                    separate = any(
                        e is not d and e.box is not None and not e.cut and canonical(e.label) == label
                        and _alike(e.radius, t.radius) and math.hypot(e.x - t.x, e.y - t.y) <= t.radius
                        and _disjoint(d.box, e.box) for e in detections
                    )
                    if not separate:
                        parts.append((d, t))
                        continue
                    reserved = t
            best, best_gap = None, math.inf
            for t in self.tracks.values():
                same = t.label == label
                if t.visual and not same:
                    continue  # a detector guess under another word must not hijack or move a vision-placed track
                alike = d.cut or not t.sized or _alike(t.radius, d.radius)
                if t is reserved or t in touched or (not same and (t.person or label == "person" or not alike)):
                    continue
                if not t.person and _across_seam(d, t):
                    continue
                if t.person and t.look is not None and d.look is not None and \
                        float(np.minimum(t.look, d.look).sum()) < UNLIKE:
                    continue  # someone else, dressed differently: two people passing close do not swap
                if t.person and t.visual_vec is not None and d.visual_vec is not None:
                    v_dot = float(np.dot(t.visual_vec, d.visual_vec))
                    v_norm = float(np.linalg.norm(t.visual_vec) * np.linalg.norm(d.visual_vec))
                    if v_norm > 1e-6 and v_dot / v_norm < 0.2:
                        continue
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
                             noise=d.sigma, width=d.width, depth_extent=d.depth_extent,
                             orientation_yaw=d.orientation_yaw, pose=d.pose,
                             visual_vec=d.visual_vec.copy() if d.visual_vec is not None else None,
                             voice_id=d.voice_id, cut=d.cut)
                self.tracks[best.id] = best
                self._next += 1
            else:  # weigh each sighting by its precision; a thing settles as sightings add up, a person may walk on
                dt = max(0.0, now - best.last_seen)
                if dt > 0.02:
                    meas_vx = (d.x - best.x) / dt
                    meas_vy = (d.y - best.y) / dt
                    if math.hypot(meas_vx, meas_vy) < 4.0:
                        beta = 0.25
                        best.vx = (1.0 - beta) * best.vx + beta * meas_vx
                        best.vy = (1.0 - beta) * best.vy + beta * meas_vy
                spread = best.var + (WALKS if best.person else MOVED) * dt
                a = spread / (spread + d.sigma ** 2)
                if d.cut and best.sized and not best.cut and best.hits >= 2:
                    a = min(a, 0.15)
                best.var = (1 - a) * spread
                best.x += a * (d.x - best.x)
                best.y += a * (d.y - best.y)
                if not d.cut:  # a cut view says nothing of size; the first whole one sets it
                    best.radius += (self.smoothing if best.sized else 1.0) * (d.radius - best.radius)
                    best.sized = True
                best.hits += 1
                best.last_seen = now
                best.cut = d.cut
                if d.width > 0:
                    best.width = (1 - 0.3) * best.width + 0.3 * d.width if best.width > 0 else d.width
                if d.depth_extent > 0:
                    best.depth_extent = (1 - 0.3) * best.depth_extent + 0.3 * d.depth_extent if best.depth_extent > 0 else d.depth_extent
                best.orientation_yaw = d.orientation_yaw
                if d.pose is not None:
                    best.pose = d.pose
                if d.voice_id is not None:
                    best.voice_id = d.voice_id
            best.misses, best.present = 0, True
            best.camera, best.normal = d.camera, d.normal
            best.noise = 0.8 * best.noise + 0.2 * d.sigma
            best.votes[label] += d.score
            best.score = max(best.score, d.score)
            if not best.person and not best.verified:
                best.label = best.votes.most_common(1)[0][0]
            if d.color:
                best.colors[d.color] += 1
            if d.look is not None:
                best.look = d.look if best.look is None else 0.8 * best.look + 0.2 * d.look
            if d.visual_vec is not None:
                if best.visual_vec is None:
                    best.visual_vec = d.visual_vec.copy()
                else:
                    best.visual_vec = 0.8 * best.visual_vec + 0.2 * d.visual_vec
                    v_norm = float(np.linalg.norm(best.visual_vec))
                    if v_norm > 1e-6:
                        best.visual_vec = best.visual_vec / v_norm
            touched.append(best)
            pairs.append((d, best))
        self._merge(pairs)
        for d, prior in parts:
            track = self.tracks[self.current(prior.id)]
            track.last_seen, track.misses = now, 0
            pairs.append((d, track))
        return pairs

    def _known_again(self, d: Detection, now: float, touched: list[Track]) -> Track | None:
        """A person near no track who looks like someone lost from view within the last half-minute, and could have
        walked to where they are now: the same person, so 'follow me' and 'the user' keep them."""
        if d.look is None and d.visual_vec is None:
            return None
        best, best_likeness = None, LIKE
        for t in self.tracks.values():
            away = now - t.last_seen
            if not t.person or t in touched or not 0.3 < away < 30.0:
                continue
            if math.dist((t.x, t.y), (d.x, d.y)) > 1.0 + 1.5 * away:  # walking pace
                continue
            likeness = 0.0
            if t.look is not None and d.look is not None:
                likeness = float(np.minimum(t.look, d.look).sum())  # histogram intersection, 1 for the same colours
            if t.visual_vec is not None and d.visual_vec is not None:
                v_dot = float(np.dot(t.visual_vec, d.visual_vec))
                v_norm = float(np.linalg.norm(t.visual_vec) * np.linalg.norm(d.visual_vec))
                if v_norm > 1e-6:
                    likeness = max(likeness, v_dot / v_norm)
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
        # Two chairs 0.65 m apart acquired one id even while both were boxed in the same picture.
        # A noisy map range must not undo that image evidence on the next frame from another camera.
        for (d, t), (e, o) in combinations(pairs, 2):
            a, b = d.box, e.box
            if a is not None and b is not None and d.camera == e.camera and t.person == o.person and _disjoint(a, b):
                t.apart.add(o.id)
                o.apart.add(t.id)
        for t in sorted(self.tracks.values(), key=lambda t: t.id):
            if t.id not in self.tracks or t.rejected or not t.present:
                continue
            for o in [o for o in self.tracks.values()
                      if o.id > t.id and o.person == t.person and not o.rejected and o.present]:
                if o.id in t.apart or ((t.visual or o.visual) and t.label != o.label):
                    continue
                gap = math.dist((t.x, t.y), (o.x, o.y))
                if not t.person and _across_seam(t, o):
                    # As with two boxes in one picture, keep the distinction after turning away.
                    # Do not make a permanent decision from a lone return or an old mapped position.
                    if (t.label == o.label and gap < max(self.gate, t.radius + o.radius)
                            and t.confirmed and o.confirmed and t.present and o.present
                            and abs(t.last_seen - o.last_seen) < 0.5):
                        t.apart.add(o.id)
                        o.apart.add(t.id)
                    continue
                scatter = min(0.6, 2.5 * math.sqrt(t.var + o.var + max(t.noise, o.noise) ** 2))  # merging is surer to hurt
                if t.person:  # closer than 0.35 m is one person, however the cameras saw their clothes
                    alike = t.look is None or o.look is None or float(np.minimum(t.look, o.look).sum()) >= LIKE
                    edge_cut = getattr(t, "cut", False) or getattr(o, "cut", False)
                    if not (gap < 0.35 or (gap < 0.75 and (alike or edge_cut))):
                        continue
                elif o.label == t.label:
                    if gap >= max(t.radius, o.radius, 0.15, scatter):
                        continue
                elif (t.label in _FURNITURE_ALIKE and o.label in _FURNITURE_ALIKE
                      and _alike(t.radius, o.radius) and not (t.sure and o.sure and t.label != o.label)
                      and gap < max(t.radius + o.radius, scatter)):
                    pass
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
                t.verified = t.verified or o.verified
                t.sure = t.sure or o.sure
                if o.visual and not t.visual:
                    t.visual, t.z, t.spotted_from = True, o.z, o.spotted_from
                for neighbor in o.apart:
                    self.tracks[neighbor].apart.remove(o.id)
                    self.tracks[neighbor].apart.add(t.id)
                t.apart.update(o.apart)
                del self.tracks[o.id]
                self.became[o.id] = t.id
                for k, (d, track) in enumerate(pairs):
                    if track is o:
                        pairs[k] = (d, t)

    def missed(self, track: Track, now: float | None = None) -> bool:
        """A camera looked clearly at where `track` should be and did not see it: True once the detector's misses say
        it is gone, GONE_AFTER in a row and, given `now`, GONE_S since anything last saw it (misses come from all three
        cameras, and two of them each missing a flickering plant for a few frames took it off the map a second after
        a search found it). Whether it is gone is the caller's to decide (perception asks the vision model first)."""
        track.misses += 1
        return track.misses >= GONE_AFTER and (now is None or now - track.last_seen >= GONE_S)

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


def _disjoint(a: tuple, b: tuple) -> bool:
    return a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def _across_seam(a: Detection | Track, b: Track) -> bool:
    """Range may slide along a clipped sightline, not sideways across it. In overlapping cameras a leg's
    ray missed the other object's centre by 0.47 m, well outside their combined 0.28 m radii."""
    if a.cut == b.cut or a.camera == b.camera:
        return False
    normal = a.normal if a.cut else b.normal
    if normal is None:
        return False
    lateral = abs(normal[0] * (a.x - b.x) + normal[1] * (a.y - b.y))
    return lateral > a.radius + b.radius + 0.10


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
