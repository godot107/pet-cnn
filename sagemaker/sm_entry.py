"""SageMaker entry point. The PyTorch container passes hyperparameters as --key value arguments
and sets PETCNN_PROCESSED / PETCNN_MODELS (see petcnn.sagemaker.launch)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from petcnn import train  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--arch", required=True)
p.add_argument("--scheme", required=True)
p.add_argument("--name", required=True)
p.add_argument("--task")
p.add_argument("--epochs", type=int)
a, _ = p.parse_known_args()
train.train(a.arch, a.scheme, a.name, a.task, a.epochs, out_dir=train.config.MODELS)
