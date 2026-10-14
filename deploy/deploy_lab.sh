#!/usr/bin/env bash
# S3 bucket + SageMaker execution role. Needed by `python -m petcnn sm-train` and deploy_serve.sh.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
require_iam_user
# shellcheck disable=SC2046
aws cloudformation deploy --stack-name "$LAB_STACK" --template-file "$ROOT/infra/lab.yaml" \
  --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset --tags $(tags lab)
echo "Lab stack ready: bucket $(out "$LAB_STACK" Bucket)"
