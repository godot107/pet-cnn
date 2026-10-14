import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from petcnn import data, evaluate, features, models, preprocess, train


def test_preprocess_shapes_and_aspect():
    wide = Image.new("RGB", (400, 200), (255, 0, 0))
    sq = preprocess.square(wide)
    assert sq.shape == (preprocess.SIZE, preprocess.SIZE, 3) and sq.dtype == np.uint8
    x = preprocess.to_input(wide)
    assert x.shape == (1, 3, preprocess.CROP, preprocess.CROP) and x.dtype == np.float32
    # pure red, ImageNet-normalised
    assert np.allclose(x[0, 0], (1 - preprocess.MEAN[0]) / preprocess.STD[0], atol=1e-5)


def test_torch_and_numpy_normalisation_agree():
    imgs = np.random.default_rng(0).integers(0, 256, (3, preprocess.CROP, preprocess.CROP, 3), dtype=np.uint8)
    a = preprocess.normalize(imgs)
    b = train.to_tensor(torch.from_numpy(imgs)).numpy()
    assert np.allclose(a, b, atol=1e-5)


def test_hamming_pairs_and_groups():
    h = np.array([0b0, 0b1, 0b11, 0xFFFF_0000_0000_0000], dtype=np.uint64)
    pairs = data._hamming_pairs(h, 1)
    assert set(pairs) == {(0, 1), (1, 2)}
    g = data._groups(4, pairs)
    assert g[0] == g[1] == g[2] != g[3]


def _toy_index():
    rows = []
    breeds = {"Bengal": "cat", "Persian": "cat", "beagle": "dog", "pug": "dog", "boxer": "dog"}
    i = 0
    for b, sp in breeds.items():
        for k in range(20):
            rows.append({"image": f"{b}_{k}", "breed": b, "species": sp, "official": "test" if k < 6 else "trainval",
                         "dup_group": i})
            i += 1
    df = pd.DataFrame(rows)
    # a near-duplicate pair straddling trainval/test, and one inside trainval
    df.loc[df.image == "Persian_10", "dup_group"] = int(df.loc[df.image == "Persian_0", "dup_group"].iloc[0])
    df.loc[df.image == "pug_12", "dup_group"] = int(df.loc[df.image == "pug_11", "dup_group"].iloc[0])
    return df


CFG = {"seed": 0, "val_frac": 0.2, "unseen_breeds": {"cat": ["Bengal"], "dog": ["pug"]}}


def test_splits_have_no_leakage():
    df = data.make_splits(_toy_index(), CFG)
    for s in ("official", "unseen-breeds"):
        col = df[f"split_{s}"]
        # no duplicate group spans two of train/val/test
        spans = df[col != "drop"].groupby("dup_group")[f"split_{s}"].nunique()
        assert (spans == 1).all(), s
        assert {"train", "val", "test"} <= set(col)
    assert df.loc[df.image == "Persian_10", "split_official"].item() == "drop"
    unseen = df[df["split_unseen-breeds"] == "test"].breed.unique()
    assert set(unseen) == {"Bengal", "pug"}
    assert not df[df["split_unseen-breeds"].isin(["train", "val"])].breed.isin(["Bengal", "pug"]).any()
    assert (df["split_leaky-2023"] == "test").sum() == (df.official == "test").sum()


def test_unknown_unseen_breed_rejected():
    with pytest.raises(ValueError):
        data.make_splits(_toy_index(), {**CFG, "unseen_breeds": {"cat": ["Tabby"], "dog": []}})


def test_metrics_binary():
    y = np.array([0, 0, 1, 1])
    proba = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.3, 0.7]])
    m = evaluate.metrics(y, proba, ["cat", "dog"], n_boot=50)
    assert m["accuracy"] == 0.75 and m["confusion"] == [[1, 1], [0, 2]]
    assert m["per_class"]["cat"]["recall"] == 0.5 and m["roc_auc"] == 1.0


@pytest.mark.parametrize("arch", ["smallcnn", "resnet18"])
def test_models_forward(arch):
    m = models.build(arch, 2, pretrained=False).eval()
    with torch.no_grad():
        assert m(torch.randn(2, 3, preprocess.CROP, preprocess.CROP)).shape == (2, 2)


def test_augment_shape_and_determinism():
    x = torch.randint(0, 256, (4, preprocess.SIZE, preprocess.SIZE, 3), dtype=torch.uint8)
    a = train.augment(x, torch.Generator().manual_seed(1))
    b = train.augment(x, torch.Generator().manual_seed(1))
    assert a.shape == (4, preprocess.CROP, preprocess.CROP, 3) and torch.equal(a, b)


def test_features_dimension():
    imgs = np.random.default_rng(0).integers(0, 256, (2, preprocess.SIZE, preprocess.SIZE, 3), dtype=np.uint8)
    assert features.extract(imgs).shape == (2, 48 + 1764)


def test_device_cuda_never_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit):
        train.device("cuda")
    assert train.device("auto").type == "cpu"
