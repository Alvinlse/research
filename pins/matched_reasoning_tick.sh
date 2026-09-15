#!/usr/bin/env bash
# One bounded cell per invocation; safe to call repeatedly from a scheduler.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
PYTHON_BIN="${PINS_PYTHON:-$REPO_DIR/.venv/bin/python}"
exec "$PYTHON_BIN" -m pins.run_matched_reasoning "$@"
