# pet-cnn

Trello "Azure Image Classification with CNN" (Vtxdy5AV), restarted from scratch on **AWS** on
2026-10-14. The 2023 Azure ML attempt stalled on blob unzip, out-of-memory errors (data held as
float tensors) and cost; its screenshot also shows `train_loader = DataLoader(test_set)`.

Task: cat vs dog (and optionally 37 breeds) on Oxford-IIIT Pet (CC BY-SA 4.0, 7,349 images).
Audience: general readers and recruiters, so README, blog and notebook stay plain-language.

## Run

```bash
source .venv/bin/activate          # own venv: torch cu130 + requirements-dev.txt, pip install -e .
pytest -q                          # 13 tests, no AWS, ~5 s
python -m petcnn fetch             # annotations from Oxford, images from the HF parquet copy (~800 MB)
python -m petcnn prepare           # 144 px uint8 cache, pHash duplicate audit, 3 split schemes (~8 s)
python -m petcnn classic --scheme official
python -m petcnn train --arch smallcnn --scheme official     # CPU ~75 s/epoch; --device cuda = GPU or stop
scripts/train_scratch_gpu.sh                # from-scratch SmallCNN reserved for the GPU (all 3 schemes, 60 epochs)
python -m petcnn eval --name smallcnn-official
python -m petcnn predict --name smallcnn-official some.jpg   # runs serve/inference.py locally
python -m petcnn report
python notebooks/build_walkthrough.py && jupyter nbconvert --to notebook --execute --inplace notebooks/walkthrough.ipynb
```

AWS (all `aws login` as WillieMaize; `require_iam_user` refuses root):

```bash
ALERT_EMAIL=<alerts address> deploy/deploy_cost.sh   # once: $10/month budget on project=pet-cnn
deploy/deploy_lab.sh                        # S3 bucket + SageMaker execution role
deploy/deploy_serve.sh [name]               # serverless endpoint + REST API (API key, 500/day quota)
DOMAIN=pets.willieman.com deploy/deploy_serve.sh   # + custom domain (2 CNAMEs at Spaceship)
deploy/down.sh [--all]                      # serve stack; --all also empties/deletes the lab stack
```

## Key decisions

- **Splits** (`data.make_splits`, index.csv `split_<scheme>`): `official` (dataset's trainval/test,
  val = 15% of trainval by breed), `unseen-breeds` (4 cat + 8 dog breeds held out entirely),
  `leaky-2023` (train = val = test = official test; the contrast). Near-duplicates (pHash Hamming
  ≤ 4) are kept in one split as whole groups; trainval images duplicating a test image are dropped (24).
- **Memory:** images stay uint8 144×144 (<0.5 GB total); crop/flip/normalise per batch.
- **Preprocessing single source:** `src/petcnn/preprocess.py` (numpy + Pillow) is used by the cache,
  training, evaluation and, copied into model.tar.gz, by the endpoint.
- **No pickle:** weights are safetensors + meta.json.
- **Training is local** (Willie's call, 2026-10-14): CPU now, the GTX 1660 after a reboot. At 7k images
  of 128 px both models are CPU-practical; ResNet-18 at 128 px costs ~0.6 GFLOPs/img, less than SmallCNN's
  ~0.85. `sm-train`/`sm-preflight`/`sm-fetch` (spot ml.g4dn.xlarge, cost guard) stay in the repo as an
  optional, **unused** path; don't run them without asking.
- **Serving:** SageMaker **Serverless** Inference (CPU, scales to zero) running `serve/inference.py` in
  the PyTorch inference container. A REST API calls `runtime.sagemaker` directly with an IAM role
  (no Lambda). It needs an `x-api-key`, and a usage plan sets 2 req/s, burst 5 and 500/day. HTTP APIs can't
  do API keys, which is why it's a REST API. Cold starts can exceed API Gateway's 29 s limit, so the first
  call after idle may return 503; retry.
- **Unverified until the first AWS run:** DLC image tags (`image_verified: false`) and prices
  (`prices.verified: false`). Check both before spending.
- **Not chosen:** a SageMaker notebook instance or Studio domain (always-on compute was the 2023
  cost and OOM trap; notebooks run locally). Studio is an optional later stack if wanted.
- **Tags:** shared/TAGGING.md; `project=pet-cnn`. CloudBurn: `.cloudburn.yml` + CI. One medium
  finding (S3 storage-class transitions) is accepted: <1 GB, objects expire.

## Environment notes

- GPU (GTX 1660) shows a "driver/library version mismatch" until a reboot; `train.device()` falls back
  to CPU (verified), so everything trains on CPU until then (SmallCNN ~75 s/epoch).
- boto3 needs `botocore[crt]` for `aws login` credentials.
