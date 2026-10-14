"""Honest scoring: per-class precision/recall/F1, confusion matrix, bootstrap CI, one JSON per run."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import config, data, features


def metrics(y: np.ndarray, proba: np.ndarray, classes: list[str], n_boot: int = 1000, seed: int = 0) -> dict:
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
    pred = proba.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(len(classes)), zero_division=0)
    rng = np.random.default_rng(seed)
    correct = (pred == y).astype(float)
    boots = [correct[rng.integers(0, len(y), len(y))].mean() for _ in range(n_boot)]
    out = {
        "n": int(len(y)),
        "accuracy": float(correct.mean()),
        "accuracy_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f.mean()),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(classes)},
        "confusion": confusion_matrix(y, pred, labels=range(len(classes))).tolist(),
    }
    if len(classes) == 2:
        out["roc_auc"] = float(roc_auc_score(y, proba[:, 1]))
    return out


def save(name: str, scheme: str, task: str, m: dict, extra: dict | None = None) -> None:
    out = config.RESULTS / scheme
    out.mkdir(parents=True, exist_ok=True)
    rec = {"name": name, "scheme": scheme, "task": task, **(extra or {}), **m}
    (out / f"{name}.json").write_text(json.dumps(rec, indent=2))
    plot_confusion(rec, out / f"{name}_confusion.png")
    ci = m["accuracy_ci95"]
    print(f"{name} [{scheme}/{task}] accuracy {m['accuracy']:.3f} (95% CI {ci[0]:.3f}-{ci[1]:.3f}), "
          f"balanced {m['balanced_accuracy']:.3f}, macro F1 {m['macro_f1']:.3f}")


def plot_confusion(rec: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(rec["confusion"])
    classes = list(rec["per_class"])
    small = len(classes) <= 4
    fig, ax = plt.subplots(figsize=(4, 3.5) if small else (11, 10))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=0 if small else 90, fontsize=None if small else 7)
    ax.set_yticks(range(len(classes)), classes, fontsize=None if small else 7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    if small:
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"{rec['name']} ({rec['scheme']}): acc {rec['accuracy']:.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --- runners --------------------------------------------------------------------------------

def classic(scheme: str, task: str | None = None, limit: int | None = None) -> None:
    """Colour histogram + HOG -> standardised logistic regression; C chosen on val, scored once on test."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    task = task or config.load()["train"]["task"]
    classes = data.class_names(task)
    xs = {s: data.load(scheme, s, task, limit) for s in ("train", "val", "test")}
    t0 = time.time()
    feats = {s: features.extract(x) for s, (x, _, _) in xs.items()}
    print(f"features {feats['train'].shape[1]}-d in {time.time() - t0:.0f}s")
    best = None
    for c in (0.001, 0.01, 0.1, 1.0):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, class_weight="balanced"))
        clf.fit(feats["train"], xs["train"][1])
        acc = clf.score(feats["val"], xs["val"][1])
        print(f"  C={c:<6} val acc {acc:.3f}")
        if best is None or acc > best[0]:
            best = (acc, c, clf)
    m = metrics(xs["test"][1], best[2].predict_proba(feats["test"]), classes)
    save("classic-hog-logreg", scheme, task, m, {"C": best[1], "val_accuracy": best[0], "limit": limit})


def model(name: str, scheme: str | None = None, limit: int | None = None) -> None:
    """Score models/<name> on the test split of its scheme (or another scheme's test split)."""
    from . import train
    net, meta = train.load_model(name)
    scheme = scheme or meta["scheme"]
    x, y, _ = data.load(scheme, "test", meta["task"], limit)
    t0 = time.time()
    proba = train.predict_proba(net, x)
    ms = (time.time() - t0) / len(x) * 1000
    m = metrics(y, proba, meta["classes"])
    save(name, scheme, meta["task"], m, {"arch": meta["arch"], "trained_on": meta["scheme"],
                                         "epochs_run": meta["epochs_run"], "ms_per_image": round(ms, 2),
                                         "limit": limit})


def report() -> str:
    rows = []
    for f in sorted(config.RESULTS.glob("*/*.json")):
        r = json.loads(f.read_text())
        rows.append({"scheme": r["scheme"], "system": r["name"], "task": r["task"], "n": r["n"],
                     "accuracy": f"{r['accuracy']:.3f}",
                     "95% CI": f"{r['accuracy_ci95'][0]:.3f}-{r['accuracy_ci95'][1]:.3f}",
                     "balanced acc": f"{r['balanced_accuracy']:.3f}", "macro F1": f"{r['macro_f1']:.3f}"})
    if not rows:
        return "no results yet"
    md = pd.DataFrame(rows).to_markdown(index=False)
    (config.RESULTS / "summary.md").write_text(md + "\n")
    return md
