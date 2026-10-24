#!/usr/bin/env bash
# Create the reference Python environment and run the tests.
#   scripts/setup_env.sh [ENV_DIR]          default ENV_DIR = .venv inside the repository
# Uses $PYTHON (default python3), which must be >= 3.11 for the pinned versions in requirements-lock.txt.
# With an older Python, set LOCK=requirements.txt or install the minimum versions (pip install -e .[test]).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${1:-$REPO/.venv}"
PY="${PYTHON:-python3}"
LOCK="${LOCK:-requirements-lock.txt}"
"$PY" - <<'PYEOF'
import sys
if sys.version_info < (3, 11):
    sys.exit(f"Python {sys.version.split()[0]} found; the reference environment needs >= 3.11 (set PYTHON=...)")
PYEOF
"$PY" -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install -q --upgrade pip
"$ENV_DIR/bin/python" -m pip install -q -r "$REPO/$LOCK"
"$ENV_DIR/bin/python" -m pip install -q -e "$REPO" --no-deps
cd "$REPO" && "$ENV_DIR/bin/python" -m pytest -q
echo "environment ready: source $ENV_DIR/bin/activate"
