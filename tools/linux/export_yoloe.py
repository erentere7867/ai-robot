"""Export YOLOE-26-seg to ONNX for the GPU vision server (tools/vision_server.py), with the words as an input.

In: picture 1x3x736x1280 (RGB 0..1, the 1280x720 frame on rows 8..727 of grey 114), words 1xSLOTSx512 (MobileCLIP
text features before the head's own text layer, as `get_text_pe(..., without_reprta=True)` gives them). Out: raw
predictions 1x(4+SLOTS+32)x19320 (box centre and size, one score per word, mask coefficients) and mask prototypes
1x32x184x320. The word count is fixed: TorchScript export (opset 17) is the one DirectML runs ("The parameter is
incorrect" for the dynamo export), and it bakes the count in; the server fills unused slots and ignores them.

Run from the models folder (elsewhere ultralytics downloads its 242 MB text encoder again), or let
tools/linux/setup_models.sh --gpu-models run it:
    cd models && ~/sprocket-venv/bin/python ../tools/linux/export_yoloe.py yoloe-26l-seg.pt vision/yoloe26l_8.onnx
"""
import sys

import numpy as np
import torch
from torch import nn
from ultralytics import YOLOE

SLOTS = 8

weights, out = sys.argv[1:3]
yolo = YOLOE(weights)
words = ["person", "chair", "ball", "cup", "dog", "phone", "keys", "bag"][:SLOTS]  # any words: only the shape stays
yolo.set_classes(words)
net = yolo.model.eval()
net.fuse(verbose=False)
raw = net.get_text_pe(words, without_reprta=True).float().clone()  # a normal tensor: inference-mode ones can't be traced
net.requires_grad_(False)
head = net.model[-1]
head.export, head.format, head.dynamic, head.max_det = True, "onnx", False, 300


class Wrapped(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, picture: torch.Tensor, words: torch.Tensor) -> torch.Tensor:
        self.net.model[-1].nc = words.shape[1]  # the head splits its output by the number of words
        return self.net.predict(picture, tpe=words)


wrapped = Wrapped(net).eval()
x = torch.rand(1, 3, 736, 1280)
with torch.no_grad():
    ref = wrapped(x, raw)
torch.onnx.export(wrapped, (x, raw), out, input_names=["picture", "words"], output_names=["predictions", "protos"],
                  opset_version=17, dynamo=False, do_constant_folding=True)

import onnxruntime as ort  # noqa: E402

pred, _ = ort.InferenceSession(out, providers=["CPUExecutionProvider"]).run(None, {"picture": x.numpy(),
                                                                                   "words": raw.numpy()})
print(f"saved {out}: predictions {pred.shape}, largest difference from torch "
      f"{float(np.abs(pred - ref[0].numpy()).max()):.2e}")
