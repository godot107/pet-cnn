# Shared settings for deploy scripts. Source, don't run.
LAB_STACK=${LAB_STACK:-pet-cnn-lab}
SERVE_STACK=${SERVE_STACK:-pet-cnn-serve}
COST_STACK=${COST_STACK:-pet-cnn-cost}
export AWS_REGION=${AWS_REGION:-us-east-1}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# shared/TAGGING.md: CloudFormation copies stack tags to every taggable resource.
tags() {
  echo project=pet-cnn environment=dev owner=williemaize cost-center=personal \
    managed-by=cloudformation component="$1" repository=github.com/godot107/pet-cnn
}

require_iam_user() {
  local arn
  arn=$(aws sts get-caller-identity --query Arn --output text) || {
    echo "No AWS session: run 'aws login' first." >&2; exit 1; }
  case "$arn" in
    *:root) echo "Refusing to deploy as the root user. 'aws login' as WillieMaize." >&2; exit 1 ;;
  esac
}

out() {  # out <stack> <OutputKey>
  aws cloudformation describe-stacks --stack-name "$1" \
    --query "Stacks[0].Outputs[?OutputKey=='$2'].OutputValue" --output text
}
