#!/usr/bin/env bash
# Delete the serve stack, then (with --all) empty and delete the lab stack. The budget stack is kept.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
require_iam_user
if aws cloudformation describe-stacks --stack-name "$SERVE_STACK" >/dev/null 2>&1; then
  echo "==> Deleting $SERVE_STACK"
  aws cloudformation delete-stack --stack-name "$SERVE_STACK"
  aws cloudformation wait stack-delete-complete --stack-name "$SERVE_STACK"
fi
if [ "${1:-}" = "--all" ]; then
  BUCKET=$(out "$LAB_STACK" Bucket)
  echo "==> Emptying $BUCKET and deleting $LAB_STACK"
  aws s3 rm "s3://$BUCKET" --recursive --only-show-errors
  aws cloudformation delete-stack --stack-name "$LAB_STACK"
  aws cloudformation wait stack-delete-complete --stack-name "$LAB_STACK"
fi
echo "Done. Training jobs stop on their own (max runtime in config.yaml); the budget stack is kept."
