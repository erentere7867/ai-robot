"""M0: can the brain see? Yes/no questions and picking numbered objects on real photos. Graded against human labels from COCO val2017.

    python -m robot.eval.brain_check --fetch   # downloads annotations and about 50 indoor photos
    python -m robot.eval.brain_check           # needs llama-server running LFM3-VL-2B with its vision projector

Pass: GROUND at least 80%, CHECK at least 85%. Planning is tested by the instruction suite in robot/eval/suite.py.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np

from robot.mind.brain import Brain, Reply

DATA = Path("~/robot-runtime/datasets/m0").expanduser()
RESULTS = Path("~/robot-runtime/runs/m0").expanduser()
ANNOTATIONS = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ROOM = {"chair", "couch", "bed", "dining table", "tv", "toilet", "sink", "refrigerator", "oven", "microwave",
        "potted plant"}
INDOOR = ROOM | {"laptop", "mouse", "remote", "keyboard", "cell phone", "toaster", "book", "clock", "vase", "scissors",
                 "teddy bear", "toothbrush", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                 "backpack", "umbrella", "handbag", "suitcase", "person", "cat", "dog"}
MIN_AREA, TARGET_AREA = 0.01, 0.02


def download(url: str, path: Path, attempts: int = 20) -> None:
    """Fetch a file, resuming after dropped connections."""
    for attempt in range(attempts):
        have = path.stat().st_size if path.exists() else 0
        try:
            with httpx.stream("GET", url, timeout=120, follow_redirects=True, trust_env=False,
                              headers={"Range": f"bytes={have}-"} if have else {}) as r:
                if r.status_code == 416:  # already complete
                    return
                r.raise_for_status()
                with path.open("ab" if r.status_code == 206 else "wb") as out:
                    for chunk in r.iter_bytes(1 << 20):
                        out.write(chunk)
            return
        except httpx.TransportError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(30.0, 2.0 * (attempt + 1)))


def fetch(count: int = 50) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    json_path = DATA / "instances_val2017.json"
    if not json_path.exists():
        archive = DATA / "annotations.zip"
        download(ANNOTATIONS, archive)
        with zipfile.ZipFile(archive) as z:
            json_path.write_bytes(z.read("annotations/instances_val2017.json"))
        archive.unlink()
    coco = json.loads(json_path.read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    boxes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    crowd: set[int] = set()
    for a in coco["annotations"]:
        (crowd.add(a["image_id"]) if a["iscrowd"] else boxes[a["image_id"]].append(a))
    images = {i["id"]: i for i in coco["images"]}
    chosen: list[dict[str, Any]] = []
    candidates = sorted(boxes)
    random.Random(7).shuffle(candidates)
    for image_id in candidates:
        if image_id in crowd:
            continue
        info = images[image_id]
        size = info["width"] * info["height"]
        labels = {names[a["category_id"]] for a in boxes[image_id]}
        kept = [{"label": names[a["category_id"]], "box": a["bbox"], "area": a["bbox"][2] * a["bbox"][3] / size}
                for a in boxes[image_id]]
        if not labels <= INDOOR or not labels & ROOM or sum(k["area"] >= MIN_AREA for k in kept) < 3:
            continue
        if not (DATA / info["file_name"]).exists():
            download(info["coco_url"], DATA / info["file_name"])
        chosen.append({"file": info["file_name"], "width": info["width"], "height": info["height"], "objects": kept})
        if len(chosen) == count:
            break
    (DATA / "manifest.json").write_text(json.dumps(chosen), encoding="utf-8")
    print(f"{len(chosen)} photos in {DATA}")


def load(item: dict[str, Any]) -> np.ndarray:
    bgr = cv2.imread(str(DATA / item["file"]))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def scaled(rgb: np.ndarray, width: int = 512) -> tuple[np.ndarray, float]:
    scale = width / rgb.shape[1]
    return cv2.resize(rgb, (width, round(rgb.shape[0] * scale))), scale


def draw_marks(rgb: np.ndarray, boxes: list[list[float]]) -> np.ndarray:
    """Numbered boxes on the image: the marks GROUND picks from."""
    out = rgb.copy()
    for number, (x, y, w, h) in enumerate(boxes, 1):
        x1, y1, x2, y2 = round(x), round(y), round(x + w), round(y + h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 230, 0), 2)
        (tw, th), _ = cv2.getTextSize(str(number), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tag_y = max(th + 6, y1)
        cv2.rectangle(out, (x1, tag_y - th - 6), (x1 + tw + 6, tag_y), (255, 230, 0), -1)
        cv2.putText(out, str(number), (x1 + 3, tag_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def url(rgb: np.ndarray) -> str:
    import base64

    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


class Tally:
    def __init__(self) -> None:
        self.results: dict[str, list[bool]] = defaultdict(list)
        self.replies: dict[str, list[Reply]] = defaultdict(list)
        self.misses: list[dict[str, Any]] = []

    def add(self, group: str, ok: bool, reply: Reply, detail: dict[str, Any]) -> None:
        self.results[group].append(ok)
        self.replies[group.split("/")[0]].append(reply)
        if not ok:
            self.misses.append({"group": group, **detail})

    def score(self, prefix: str) -> tuple[int, int]:
        hits = [r for g, v in self.results.items() if g == prefix or g.startswith(prefix + "/") for r in v]
        return sum(hits), len(hits)


def check_photos(brain: Brain, tally: Tally, photos: list[dict[str, Any]]) -> None:
    rng = random.Random(11)
    for item in photos:
        rgb, scale = scaled(load(item))
        objects = sorted((o for o in item["objects"] if o["area"] >= MIN_AREA), key=lambda o: -o["area"])[:8]
        boxes = [[v * scale for v in o["box"]] for o in objects]
        marked = url(draw_marks(rgb, boxes))
        plain = url(rgb)
        everything = {o["label"] for o in item["objects"]}
        absent = sorted(INDOOR - everything)
        by_label: dict[str, list[int]] = defaultdict(list)
        for i, o in enumerate(objects):
            by_label[o["label"]].append(i)
        # GROUND: one object of its kind; a kind that is not there; the leftmost or rightmost of several.
        single = [lb for lb, idx in by_label.items() if len(idx) == 1 and objects[idx[0]]["area"] >= TARGET_AREA
                  and sum(o["label"] == lb for o in item["objects"]) == 1]
        for label in rng.sample(single, min(2, len(single))):
            got, reply = brain.ground(plain, marked, f"the {label}", len(boxes))
            tally.add("ground/single", got == by_label[label][0] + 1, reply, {"file": item["file"], "ask": label, "got": got})
        missing = rng.choice(absent)
        got, reply = brain.ground(plain, marked, f"the {missing}", len(boxes))
        tally.add("ground/absent", got is None, reply, {"file": item["file"], "ask": missing, "got": got})
        for label, idx in by_label.items():
            if len(idx) >= 2:
                order = sorted(idx, key=lambda i: objects[i]["box"][0] + objects[i]["box"][2] / 2)
                gap = (objects[order[1]]["box"][0] + objects[order[1]]["box"][2] / 2
                       - objects[order[0]]["box"][0] - objects[order[0]]["box"][2] / 2)
                if gap > 0.12 * item["width"] and sum(o["label"] == label for o in item["objects"]) == len(idx):
                    got, reply = brain.ground(plain, marked, f"the leftmost {label}", len(boxes))
                    tally.add("ground/relative", got == order[0] + 1, reply,
                              {"file": item["file"], "ask": f"leftmost {label}", "got": got})
                    break
        # CHECK: is something there, and where is it relative to something else?
        present = [o["label"] for o in objects if o["area"] >= TARGET_AREA]
        if present:
            label = rng.choice(present)
            yes, reply = brain.check(plain, f"Is there a {label} in this picture?")
            tally.add("check/present", yes, reply, {"file": item["file"], "ask": label})
        missing = rng.choice(absent)
        yes, reply = brain.check(plain, f"Is there a {missing} in this picture?")
        tally.add("check/absent", not yes, reply, {"file": item["file"], "ask": missing})
        apart = [(a, b) for a in single for b in single if a != b and (
            objects[by_label[a][0]]["box"][0] + objects[by_label[a][0]]["box"][2] < objects[by_label[b][0]]["box"][0])]
        if apart:
            a, b = rng.choice(apart)
            if rng.random() < 0.5:
                yes, reply = brain.check(plain, f"Is the {a} to the left of the {b}?")
                tally.add("check/spatial", yes, reply, {"file": item["file"], "ask": f"{a} left of {b}"})
            else:
                yes, reply = brain.check(plain, f"Is the {b} to the left of the {a}?")
                tally.add("check/spatial", not yes, reply, {"file": item["file"], "ask": f"{b} left of {a}"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--photos", type=int, default=50)
    args = parser.parse_args()
    if args.fetch:
        fetch(args.photos)
        return
    brain, tally = Brain(), Tally()
    if not brain.healthy():
        raise SystemExit("llama-server is not answering on http://127.0.0.1:8080")
    photos = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))[:args.photos]
    check_photos(brain, tally, photos)
    print("\ngroup               score")
    summary: dict[str, Any] = {}
    for group in sorted(tally.results):
        hits, n = sum(tally.results[group]), len(tally.results[group])
        summary[group] = [hits, n]
        print(f"{group:<20}{hits}/{n}  {100 * hits / n:.0f}%")
    for name, target in (("ground", 80), ("check", 85)):
        hits, n = tally.score(name)
        if n:
            verdict = "PASS" if 100 * hits / n >= target else "FAIL"
            replies = tally.replies[name]
            read, write = statistics.median(r.read_ms for r in replies), statistics.median(r.write_ms for r in replies)
            print(f"{name.upper():<8}{hits}/{n} = {100 * hits / n:.1f}%  {verdict}   median read {read:.0f} ms, write {write:.0f} ms (laptop GPU)")
            summary[name.upper()] = {"hits": hits, "n": n, "verdict": verdict, "read_ms": read, "write_ms": write}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "latest.json").write_text(json.dumps({"summary": summary, "misses": tally.misses}, indent=1,
                                                    default=str), encoding="utf-8")
    print(f"details: {RESULTS / 'latest.json'}")


if __name__ == "__main__":
    main()
