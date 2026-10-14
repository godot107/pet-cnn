"""The endpoint handler (serve/inference.py) and its model.tar.gz, without AWS."""
import importlib
import io
import json
import sys
import tarfile

import numpy as np
import pytest
from PIL import Image

from petcnn import config, models, preprocess, sagemaker, train


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (300, 200, 3), dtype=np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def toy_model(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS", tmp_path)
    train.save(models.SmallCNN(2), tmp_path / "toy", {"arch": "smallcnn", "classes": ["cat", "dog"]})
    return tmp_path / "toy"


def test_model_tar_layout(toy_model):
    with tarfile.open(fileobj=io.BytesIO(sagemaker.model_tar("toy"))) as t:
        names = set(t.getnames())
    assert names == {"model.safetensors", "meta.json", "code/inference.py", "code/requirements.txt",
                     "code/models.py", "code/preprocess.py"}


def test_inference_handler_roundtrip(toy_model, monkeypatch):
    monkeypatch.setitem(sys.modules, "models", models)
    monkeypatch.setitem(sys.modules, "preprocess", preprocess)
    monkeypatch.syspath_prepend(str(config.ROOT / "serve"))
    inf = importlib.import_module("inference")
    model = inf.model_fn(str(toy_model))
    out = json.loads(inf.output_fn(inf.predict_fn(inf.input_fn(_jpeg(), "image/jpeg"), model), "application/json"))
    assert out["label"] in ("cat", "dog") and abs(sum(out["probabilities"].values()) - 1) < 1e-3
    with pytest.raises(ValueError):
        inf.input_fn(b"{}", "application/json")
