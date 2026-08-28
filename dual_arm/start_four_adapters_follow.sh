#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../deploy/runtime_env.sh
source "$SCRIPT_DIR/../deploy/runtime_env.sh"

cd "$PROJECT_DIR"
exec "$ENV_DIR/bin/python" dual_arm/test_four_adapters_follow.py
