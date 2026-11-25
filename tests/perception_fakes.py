import base64
from collections.abc import Callable
from types import SimpleNamespace

import cv2
import numpy as np

from robot.perception.detector import Detector


def make_detector(server, words: tuple[str, ...]) -> Detector:
    detector = Detector(server=server)
    detector._features = {word: np.full(512, i, np.float32) for i, word in enumerate((*detector.always, *words))}
    return detector


def make_image_brain(probability: Callable[[np.ndarray], float]) -> SimpleNamespace:
    def yes_probability(messages: list[dict]) -> float:
        url = messages[0]["content"][0]["image_url"]["url"]
        raw = base64.b64decode(url.split(",", 1)[1])
        return probability(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR))

    return SimpleNamespace(yes_probability=yes_probability)
