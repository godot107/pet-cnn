#!/usr/bin/env bash
# Package models/<name> for SageMaker, then deploy the serverless endpoint + API-key REST API.
# Usage: deploy/deploy_serve.sh [model-name]   (default: serve.model in config.yaml)
#        DOMAIN=pets.willieman.com deploy/deploy_serve.sh   (custom domain; see infra/serve.yaml)
# The endpoint costs nothing while idle, but tear it down with deploy/down.sh when done anyway.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
require_iam_user
PY=$ROOT/.venv/bin/python
cfg() { "$PY" -c "import yaml;print(yaml.safe_load(open('$ROOT/config.yaml'))['serve']['$1'])"; }
NAME=${1:-$(cfg model)}
[ -f "$ROOT/models/$NAME/model.safetensors" ] || { echo "No models/$NAME: train it first." >&2; exit 1; }

MODEL_URL=$("$PY" -m petcnn sm-package --name "$NAME" | tail -1)
# shellcheck disable=SC2046
aws cloudformation deploy --stack-name "$SERVE_STACK" --template-file "$ROOT/infra/serve.yaml" \
  --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset --tags $(tags app) \
  --parameter-overrides ModelDataUrl="$MODEL_URL" ImageUri="$(cfg image_uri)" \
    ExecutionRoleArn="$(out "$LAB_STACK" RoleArn)" DomainName="${DOMAIN:-}" \
    MemoryMb="$(cfg memory_mb)" MaxConcurrency="$(cfg max_concurrency)"
KEY=$(aws apigateway get-api-key --api-key "$(out "$SERVE_STACK" ApiKeyId)" --include-value --query value --output text)
echo "Endpoint: $(out "$SERVE_STACK" EndpointName)   (IAM: python -m petcnn sm-invoke cat.jpg)"
echo "Public:   curl --data-binary @cat.jpg -H 'content-type: image/jpeg' -H 'x-api-key: $KEY' $(out "$SERVE_STACK" ApiUrl)/predict"
[ -n "${DOMAIN:-}" ] && echo "DNS:      CNAME $DOMAIN -> $(out "$SERVE_STACK" CustomDomainTarget) at Spaceship"
exit 0
