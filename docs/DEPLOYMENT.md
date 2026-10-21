# Deployment record

## 2026-10-21: serverless endpoint + API-key REST API (deployed, verified, torn down the same day)

**What ran:** `resnet18-official` (test accuracy 0.972 on the official split) behind a SageMaker Serverless endpoint
(CPU, 3,072 MB, max concurrency 2), fronted by an API Gateway REST API that requires an API key (usage plan with a
500 requests/day quota). Region us-east-1, deployed as IAM user WillieMaize.

| stack | resources |
|---|---|
| pet-cnn-cost | 1 budget ($10/month alert on `project=pet-cnn`) |
| pet-cnn-lab | S3 bucket (model.tar.gz), SageMaker execution role |
| pet-cnn-serve | SageMaker model + endpoint config + serverless endpoint; REST API (resource, method, stage `v1`, deployment), usage plan, API key, IAM role for the API → SageMaker integration |

Steps: `ALERT_EMAIL=... deploy/deploy_cost.sh`, `deploy/deploy_lab.sh`, `deploy/deploy_serve.sh resnet18-official`.
Image: `pytorch-inference:2.6.0-cpu-py312-ubuntu22.04-sagemaker`, checked in ECR before deploying.

**Smoke test** (real Oxford-IIIT Pet photos, `curl --data-binary @img -H 'x-api-key: ...' .../v1/predict`):

| photo | answer | confidence | time (warm) |
|---|---|---|---|
| Abyssinian_120 | cat | 1.000 | 1.3 s |
| beagle_185 | dog | 0.999 | 1.6 s |
| Bengal_10 | cat | 1.000 | 1.5 s |
| pug_12 | dog | 0.988 | 1.4 s |
| (no API key) | HTTP 403 | | |

The first call with a new key also returned 403 for about a minute until the key propagated.

**Fix needed to deploy:** `infra/lab.yaml` used to create the log group `/aws/sagemaker/TrainingJobs`. That name is
account-wide and another project had already created it, so CloudFormation's early name-conflict check failed the
stack. Removed (training is local). Commit 684ee39.

**Cost:** serverless endpoints bill per millisecond of inference only; nothing while idle. The test was a handful of
calls (well under $0.01).

**Teardown:** see below.


## Teardown (2026-10-21, same session)

1. `deploy/down.sh --all`: deleted the serving/app stacks, emptied and deleted the bucket stack.
2. Deleted the budget stack too (down.sh keeps it by design; nothing was left to watch).
3. Deleted what CloudFormation does not own:
   - the CloudWatch log groups SageMaker creates per endpoint (`/aws/sagemaker/Endpoints/<name>`);
   - the SageMaker lineage records it creates per endpoint. Order matters: associations first, then the
     actions and contexts, then the artifacts (model.tar.gz and container image), or AWS refuses with
     "Cannot delete entity with associations".
4. Verified: 0 stacks, 0 endpoints, 0 resources tagged with this project (Resource Groups Tagging API), no buckets.

To redeploy, run the deploy steps above again; nothing from this run is reused.
