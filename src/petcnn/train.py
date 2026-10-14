"""Training with a real validation set: early stopping on val loss, best weights restored.

Images stay uint8 (N x 144 x 144 x 3, ~60 KB each) until a batch is cropped, flipped and normalised.
The 2023 notebook turned every image into float tensors up front and ran out of 14 GB of RAM; this
keeps the full dataset under 0.5 GB.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import config, data, models, preprocess


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device(want: str = "auto") -> torch.device:
    """auto: CUDA if it actually works (a driver/library mismatch can report available but fail), else CPU.
    cuda: GPU or stop; never silently falls back. cpu: CPU."""
    if want == "cpu":
        return torch.device("cpu")
    err = "torch.cuda.is_available() is False"
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except RuntimeError as e:
            err = str(e)
    if want == "cuda":
        raise SystemExit(f"--device cuda: no usable GPU ({err}). A driver/library mismatch needs a reboot; "
                         "check with nvidia-smi.")
    print(f"CUDA unusable ({err}); using CPU")
    return torch.device("cpu")


_MEAN = torch.tensor(preprocess.MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(preprocess.STD).view(1, 3, 1, 1)


def to_tensor(x: torch.Tensor) -> torch.Tensor:
    """uint8 NHWC -> normalised float NCHW (torch twin of preprocess.normalize)."""
    x = x.permute(0, 3, 1, 2).float() / 255.0
    return (x - _MEAN.to(x.device)) / _STD.to(x.device)


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random CROP crop (one offset per batch) + per-image horizontal flip, on uint8 NHWC."""
    span = preprocess.SIZE - preprocess.CROP
    t, l = (int(v) for v in torch.randint(0, span + 1, (2,), generator=gen))
    x = x[:, t:t + preprocess.CROP, l:l + preprocess.CROP, :]
    flip = torch.rand(len(x), generator=gen) < 0.5
    x = x.clone()
    x[flip] = x[flip].flip(2)
    return x


@torch.no_grad()
def predict_proba(model: nn.Module, imgs: np.ndarray, batch: int = 128) -> np.ndarray:
    """Class probabilities for uint8 SIZE squares, centre-cropped exactly as at serving time."""
    dev = next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(imgs), batch):
        x = torch.from_numpy(np.ascontiguousarray(preprocess.center_crop(imgs[i:i + batch]))).to(dev)
        out.append(torch.softmax(model(to_tensor(x)), 1).cpu().numpy())
    return np.concatenate(out)


def _epoch_loss(model, x_all, y_all, loss_fn, dev, batch) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(x_all), batch):
            x = torch.from_numpy(np.ascontiguousarray(preprocess.center_crop(x_all[i:i + batch]))).to(dev)
            y = torch.from_numpy(y_all[i:i + batch]).to(dev)
            total += loss_fn(model(to_tensor(x)), y).item() * len(y)
    return total / len(x_all)


def train(arch: str, scheme: str, name: str, task: str | None = None, epochs: int | None = None,
          limit: int | None = None, out_dir: Path | None = None, device_pref: str = "auto") -> Path:
    cfg = config.load()
    tcfg = cfg["train"]
    task = task or tcfg["task"]
    epochs = epochs or tcfg["epochs"]
    seed = cfg["data"]["seed"]
    seed_all(seed)
    if tcfg.get("num_threads"):
        torch.set_num_threads(tcfg["num_threads"])

    xtr, ytr, _ = data.load(scheme, "train", task, limit, seed)
    xva, yva, _ = data.load(scheme, "val", task, limit and max(limit // 4, 32), seed)
    classes = data.class_names(task)
    dev = device(device_pref)
    model = models.build(arch, len(classes)).to(dev)
    print(f"{name}: {arch} ({models.n_params(model) / 1e6:.2f}M params) on {dev}, "
          f"train {len(xtr)} / val {len(xva)}, {len(classes)} classes")

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"][arch], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    # Class weights: cats are a third of the data.
    w = torch.tensor(len(ytr) / (len(classes) * np.bincount(ytr, minlength=len(classes)).clip(1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=w.to(dev))
    gen = torch.Generator().manual_seed(seed)
    xtr_t, ytr_t = torch.from_numpy(xtr), torch.from_numpy(ytr)

    best, best_state, bad, history = float("inf"), None, 0, []
    bs = tcfg["batch_size"]
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(xtr_t), generator=gen)
        run = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            x = to_tensor(augment(xtr_t[idx], gen).to(dev))
            y = ytr_t[idx].to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            run += loss.item() * len(idx)
        sched.step()
        val = _epoch_loss(model, xva, yva, loss_fn, dev, 256)
        history.append({"epoch": ep, "train_loss": run / len(perm), "val_loss": val, "sec": round(time.time() - t0, 1)})
        print(f"  epoch {ep:2d}  train {run / len(perm):.4f}  val {val:.4f}  ({time.time() - t0:.0f}s)")
        if val < best - 1e-4:
            best, bad = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tcfg["patience"]:
                print(f"  early stop: no val improvement for {bad} epochs")
                break

    model.load_state_dict(best_state)
    out = out_dir or (config.MODELS / name)
    save(model.cpu(), out, {"device": dev.type, "arch": arch, "scheme": scheme, "task": task, "classes": classes,
                            "best_val_loss": best, "epochs_run": len(history), "limit": limit,
                            "size": preprocess.SIZE, "crop": preprocess.CROP})
    (out / "history.json").write_text(json.dumps(history, indent=2))
    return out


# --- persistence: safetensors + JSON, never pickle -------------------------------------------

def save(model: nn.Module, out: Path, meta: dict) -> None:
    from safetensors.torch import save_file
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, out / "model.safetensors")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out}")


def load_model(name: str, dev: torch.device | None = None) -> tuple[nn.Module, dict]:
    from safetensors.torch import load_file
    d = config.MODELS / name
    meta = json.loads((d / "meta.json").read_text())
    model = models.build(meta["arch"], len(meta["classes"]), pretrained=False)
    model.load_state_dict(load_file(d / "model.safetensors"))
    return model.to(dev or device()).eval(), meta

