"""Oxford-IIIT Pet: download, index, image cache, near-duplicate audit and the three split schemes.

Schemes (column `split_<scheme>` in index.csv, values train/val/test/drop):
- official      : the dataset's own trainval/test; val carved from trainval by breed. Trainval images
                  that are near-duplicates of a test image are dropped.
- unseen-breeds : every image of the config's held-out breeds is test; the rest is train/val. Can a
                  cat/dog model generalise to breeds it never saw? (the hard benchmark)
- leaky-2023    : the 2023 Azure notebook's bug: train_loader = DataLoader(test_set). Train, val and
                  test are all the official test set. Kept only as the contrast.
"""
from __future__ import annotations

import json
import tarfile
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import config, preprocess

SCHEMES = ("official", "unseen-breeds", "leaky-2023")
SPECIES = ("cat", "dog")
CACHE = f"images_{preprocess.SIZE}.npy"


# --- download -------------------------------------------------------------------------------

def _download(urls: list[str], dest: Path, min_bytes: int, ua: str) -> None:
    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f"have {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for url in urls:
        try:
            print(f"downloading {url}")
            req = urllib.request.Request(url, headers={"User-Agent": ua})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            with open(part, "rb") as f:
                magic = f.read(2)
            if magic != b"\x1f\x8b" or part.stat().st_size < min_bytes:
                raise ValueError(f"not a full gzip ({part.stat().st_size} bytes)")
            part.rename(dest)
            return
        except Exception as e:  # noqa: BLE001 - try the next mirror
            print(f"  failed: {e}")
    raise SystemExit(
        f"Could not download {dest.name}. Fetch it by hand from https://www.robots.ox.ac.uk/~vgg/data/pets/ "
        f"and save it as {dest}."
    )


def _download_magic(url: str, dest: Path, min_bytes: int, ua: str, magic: bytes) -> bool:
    """Single-URL download with size + magic-byte check; False instead of raising."""
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        print(f"downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": ua})
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        with open(part, "rb") as f:
            ok = f.read(len(magic)) == magic
        if not ok or part.stat().st_size < min_bytes:
            raise ValueError(f"bad file ({part.stat().st_size} bytes)")
        part.rename(dest)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  failed: {e}")
        return False


def _images_from_parquet(cfg: dict) -> bool:
    """Write images/<id>.jpg from the Hugging Face parquet files (original JPEG bytes)."""
    import pyarrow.parquet as pq
    files = []
    for i, url in enumerate(cfg["hf_parquet"]):
        dest = config.RAW / "hf" / f"part{i}.parquet"
        if not _download_magic(url, dest, cfg["hf_min_bytes"], cfg["user_agent"], b"PAR1"):
            return False
        files.append(dest)
    out = config.RAW / "images"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in files:
        t = pq.read_table(f, columns=["image", "image_id"])
        for img, iid in zip(t.column("image").to_pylist(), t.column("image_id").to_pylist()):
            (out / f"{iid}.jpg").write_bytes(img["bytes"])
            n += 1
    print(f"wrote {n} images from parquet")
    return True


def fetch() -> None:
    cfg = config.load()["data"]
    tgz = config.RAW / "annotations.tar.gz"
    _download(cfg["urls"]["annotations"], tgz, cfg["min_bytes"]["annotations"], cfg["user_agent"])
    if not (config.RAW / "annotations").is_dir():
        with tarfile.open(tgz) as t:
            t.extractall(config.RAW, filter="data")
    if (config.RAW / "images").is_dir() and len(list((config.RAW / "images").glob("*.jpg"))) >= 7349:
        print("have images/")
        return
    if _images_from_parquet(cfg):
        return
    tgz = config.RAW / "images.tar.gz"
    _download(cfg["urls"]["images"], tgz, cfg["min_bytes"]["images"], cfg["user_agent"])
    print(f"extracting {tgz.name}")
    with tarfile.open(tgz) as t:
        t.extractall(config.RAW, filter="data")


# --- index ----------------------------------------------------------------------------------

def _read_list(path: Path) -> pd.DataFrame:
    rows = [ln.split() for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]
    df = pd.DataFrame(rows, columns=["image", "class_id", "species_id", "breed_in_species"])
    return df.astype({"class_id": int, "species_id": int})


def build_index() -> pd.DataFrame:
    """One row per annotated image: image, breed, breed_id (0..36), species, official split."""
    ann = config.RAW / "annotations"
    df = _read_list(ann / "list.txt")
    test = set(_read_list(ann / "test.txt").image)
    df["breed"] = df.image.str.rsplit("_", n=1).str[0]
    df["breed_id"] = df.class_id - 1
    df["species"] = df.species_id.map({1: "cat", 2: "dog"})
    df["official"] = np.where(df.image.isin(test), "test", "trainval")
    return df[["image", "breed", "breed_id", "species", "official"]].reset_index(drop=True)


# --- cache + near-duplicates ----------------------------------------------------------------

def _load_one(path: str) -> tuple[np.ndarray, int]:
    import imagehash
    with Image.open(path) as im:
        sq = preprocess.square(im)
    h = imagehash.phash(Image.fromarray(sq))
    return sq, int(str(h), 16)


def _hamming_pairs(hashes: np.ndarray, max_d: int) -> list[tuple[int, int]]:
    bits = np.unpackbits(hashes.astype(">u8").view(np.uint8).reshape(-1, 8), axis=1).astype(np.float32)
    d = bits @ (1 - bits).T
    d = d + d.T                                    # Hamming distance matrix
    i, j = np.where(np.triu(d <= max_d, k=1))
    return list(zip(i.tolist(), j.tolist()))


def _groups(n: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        parent[find(a)] = find(b)
    return np.array([find(a) for a in range(n)])


# --- splits ---------------------------------------------------------------------------------

def _train_val(df: pd.DataFrame, pool: pd.Series, val_frac: float, rng) -> pd.Series:
    """Assign pool rows to train/val per breed, whole duplicate groups at a time."""
    out = pd.Series(index=df.index[pool], dtype=object)
    for _, sub in df[pool].groupby("breed"):
        groups = sub.dup_group.unique()
        rng.shuffle(groups)
        target, n_val = val_frac * len(sub), 0
        for g in groups:
            members = sub.index[sub.dup_group == g]
            if n_val < target:
                out[members], n_val = "val", n_val + len(members)
            else:
                out[members] = "train"
    return out


def _drop_touching(df: pd.DataFrame, is_test: pd.Series) -> pd.Series:
    """Non-test rows whose duplicate group contains a test image."""
    test_groups = set(df.dup_group[is_test])
    return ~is_test & df.dup_group.isin(test_groups)


def make_splits(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["seed"])
    unseen = cfg["unseen_breeds"]["cat"] + cfg["unseen_breeds"]["dog"]
    missing = set(unseen) - set(df.breed)
    if missing:
        raise ValueError(f"unseen_breeds not in the data: {sorted(missing)}")

    for scheme, is_test in (("official", df.official == "test"), ("unseen-breeds", df.breed.isin(unseen))):
        col = pd.Series("test", index=df.index, dtype=object)
        drop = _drop_touching(df, is_test)
        col[drop] = "drop"
        pool = ~is_test & ~drop
        col[pool] = _train_val(df, pool, cfg["val_frac"], rng)
        df[f"split_{scheme}"] = col

    # leaky-2023: every role played by the official test set (one row per role, see load()).
    df["split_leaky-2023"] = np.where(df.official == "test", "test", "drop")
    return df


def prepare(workers: int = 8) -> None:
    cfg = config.load()["data"]
    df = build_index()
    paths = [str(config.RAW / "images" / f"{i}.jpg") for i in df.image]
    print(f"caching {len(paths)} images at {preprocess.SIZE}px + pHash ({workers} workers)")
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(_load_one, paths, chunksize=64))
    imgs = np.stack([o[0] for o in out])
    hashes = np.array([o[1] for o in out], dtype=np.uint64)

    pairs = _hamming_pairs(hashes, cfg["dup_hamming"])
    df["dup_group"] = _groups(len(df), pairs)
    df["phash"] = [f"{h:016x}" for h in hashes]
    df = make_splits(df, cfg)

    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    np.save(config.PROCESSED / CACHE, imgs)
    df.to_csv(config.PROCESSED / "index.csv", index=False)
    audit = duplicate_audit(df, pairs)
    (config.PROCESSED / "duplicates.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit["summary"], indent=2))
    for s in SCHEMES:
        print(s, df[f"split_{s}"].value_counts().to_dict())


def duplicate_audit(df: pd.DataFrame, pairs: list[tuple[int, int]]) -> dict:
    off = df.official.to_numpy()
    cross = [(a, b) for a, b in pairs if off[a] != off[b]]
    return {
        "summary": {
            "images": len(df),
            "near_duplicate_pairs": len(pairs),
            "pairs_crossing_official_trainval_test": len(cross),
            "pairs_with_different_breed_label": sum(df.breed[a] != df.breed[b] for a, b in pairs),
            "dropped_official": int((df["split_official"] == "drop").sum()),
            "dropped_unseen_breeds": int((df["split_unseen-breeds"] == "drop").sum()),
        },
        "pairs": [[df.image[a], df.image[b]] for a, b in pairs],
    }


# --- loading --------------------------------------------------------------------------------

def class_names(task: str) -> list[str]:
    if task == "species":
        return list(SPECIES)
    df = load_index()
    return df.drop_duplicates("breed_id").sort_values("breed_id").breed.tolist()


def labels(df: pd.DataFrame, task: str) -> np.ndarray:
    if task == "species":
        return (df.species == "dog").to_numpy(np.int64)
    return df.breed_id.to_numpy(np.int64)


def load_index() -> pd.DataFrame:
    return pd.read_csv(config.PROCESSED / "index.csv")


def load(scheme: str, split: str, task: str, limit: int | None = None, seed: int = 0):
    """(images uint8 N x SIZE x SIZE x 3, labels int64, index rows) for one split of a scheme."""
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}")
    df = load_index()
    col = f"split_{scheme}"
    rows = df[df[col] == "test"] if scheme == "leaky-2023" else df[df[col] == split]
    if limit and limit < len(rows):
        rows = rows.sample(limit, random_state=seed)          # random, never a prefix
    imgs = np.load(config.PROCESSED / CACHE, mmap_mode="r")
    return np.asarray(imgs[rows.index.to_numpy()]), labels(rows, task), rows.reset_index(drop=True)
