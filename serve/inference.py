"""SageMaker inference handler (PyTorch container): model.tar.gz ships this file as code/inference.py
next to models.py and preprocess.py, copied from src/petcnn. `python -m petcnn predict` runs the same
four functions locally, so what you test is what the endpoint runs.
"""
import io
import json
import os

import numpy as np
import torch
from PIL import Image

import models
import preprocess

MAX_BYTES = 4_000_000          # serverless endpoints accept at most 4 MB per request


def model_fn(model_dir):
    from safetensors.torch import load_file
    meta = json.load(open(os.path.join(model_dir, "meta.json")))
    net = models.build(meta["arch"], len(meta["classes"]), pretrained=False)
    net.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")))
    return net.eval(), meta["classes"]


def input_fn(body, content_type):
    if not (content_type.startswith("image/") or content_type == "application/octet-stream"):
        raise ValueError(f"send image bytes (image/jpeg, image/png); got {content_type}")
    if len(body) > MAX_BYTES:
        raise ValueError(f"image larger than {MAX_BYTES} bytes")
    with Image.open(io.BytesIO(body)) as im:
        return torch.from_numpy(preprocess.to_input(im))


def predict_fn(x, model_and_classes):
    net, classes = model_and_classes
    with torch.no_grad():
        p = torch.softmax(net(x), 1)[0].numpy()
    return classes, p


def output_fn(prediction, accept):
    classes, p = prediction
    top = int(np.argmax(p))
    return json.dumps({"label": classes[top], "confidence": round(float(p[top]), 4),
                       "probabilities": {c: round(float(v), 4) for c, v in zip(classes, p)}})
