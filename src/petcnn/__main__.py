"""CLI: python -m petcnn <command>."""
from __future__ import annotations

import argparse

from . import data, evaluate, models


def main(argv=None):
    p = argparse.ArgumentParser(prog="petcnn")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="download Oxford-IIIT Pet (~800 MB)")
    s = sub.add_parser("prepare", help="index, 144 px cache, near-duplicate audit, split schemes")
    s.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("classic", help="colour histogram + HOG -> logistic regression baseline")
    s.add_argument("--scheme", default="official", choices=data.SCHEMES)
    s.add_argument("--task")
    s.add_argument("--limit", type=int)
    s = sub.add_parser("train", help="train a CNN locally (GPU if it works, else CPU)")
    s.add_argument("--arch", default="smallcnn", choices=models.ARCHS)
    s.add_argument("--scheme", default="official", choices=data.SCHEMES)
    s.add_argument("--name", help="default <arch>-<scheme>")
    s.add_argument("--task")
    s.add_argument("--epochs", type=int)
    s.add_argument("--limit", type=int, help="random N training images (smoke test)")
    s.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"),
                   help="cuda = GPU only, stop if unavailable (no silent CPU fallback)")
    s = sub.add_parser("eval", help="score models/<name> on a test split")
    s.add_argument("--name", required=True)
    s.add_argument("--scheme", choices=data.SCHEMES, help="default: the scheme it was trained on")
    s.add_argument("--limit", type=int)
    s = sub.add_parser("predict", help="classify image files locally, through the endpoint's handler")
    s.add_argument("--name", required=True)
    s.add_argument("images", nargs="+")
    sub.add_parser("report", help="markdown table of every scored system")
    s = sub.add_parser("sm-train", help="SageMaker spot GPU training job (costs money)")
    s.add_argument("--arch", default="resnet18", choices=models.ARCHS)
    s.add_argument("--scheme", default="official", choices=data.SCHEMES)
    s.add_argument("--name")
    s.add_argument("--task")
    s.add_argument("--epochs", type=int)
    s.add_argument("--max-usd", type=float)
    s.add_argument("--no-wait", action="store_true")
    sub.add_parser("sm-preflight", help="print the worst-case SageMaker job cost; no AWS calls")
    s = sub.add_parser("sm-status")
    s.add_argument("job")
    s.add_argument("--wait", action="store_true")
    s = sub.add_parser("sm-fetch", help="download a finished job's model into models/<name>")
    s.add_argument("job")
    s.add_argument("--name", required=True)
    s = sub.add_parser("sm-package", help="upload models/<name> as model.tar.gz; prints the S3 URI")
    s.add_argument("--name", required=True)
    s = sub.add_parser("sm-invoke", help="call the deployed serverless endpoint with image files")
    s.add_argument("images", nargs="+")
    a = p.parse_args(argv)

    if a.cmd == "fetch":
        data.fetch()
    elif a.cmd == "prepare":
        data.prepare(a.workers)
    elif a.cmd == "classic":
        evaluate.classic(a.scheme, a.task, a.limit)
    elif a.cmd == "train":
        from . import train
        train.train(a.arch, a.scheme, a.name or f"{a.arch}-{a.scheme}", a.task, a.epochs, a.limit,
                    device_pref=a.device)
    elif a.cmd == "eval":
        evaluate.model(a.name, a.scheme, a.limit)
    elif a.cmd == "predict":
        _predict(a.name, a.images)
    elif a.cmd == "report":
        print(evaluate.report())
    elif a.cmd == "sm-train":
        from . import sagemaker
        sagemaker.launch(a.arch, a.scheme, a.name or f"{a.arch}-{a.scheme}-sm", a.task, a.epochs, a.max_usd, not a.no_wait)
    elif a.cmd == "sm-preflight":
        from . import sagemaker
        sagemaker.preflight()
    elif a.cmd == "sm-status":
        from . import sagemaker
        sagemaker.status(a.job, a.wait)
    elif a.cmd == "sm-fetch":
        from . import sagemaker
        sagemaker.fetch(a.job, a.name)
    elif a.cmd == "sm-package":
        from . import sagemaker
        sagemaker.package(a.name)
    elif a.cmd == "sm-invoke":
        from . import sagemaker
        sagemaker.invoke(a.images)


def _predict(name: str, images: list[str]) -> None:
    """Runs serve/inference.py's model_fn -> input_fn -> predict_fn -> output_fn, as the endpoint does."""
    import importlib.util
    import mimetypes
    import sys

    from . import config, models, preprocess
    sys.modules.setdefault("models", models)          # the handler imports these as top-level modules
    sys.modules.setdefault("preprocess", preprocess)
    spec = importlib.util.spec_from_file_location("inference", config.ROOT / "serve" / "inference.py")
    inf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inf)
    model = inf.model_fn(str(config.MODELS / name))
    for path in images:
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        x = inf.input_fn(open(path, "rb").read(), ctype)
        print(f"{path}: {inf.output_fn(inf.predict_fn(x, model), 'application/json')}")

if __name__ == "__main__":
    main()
