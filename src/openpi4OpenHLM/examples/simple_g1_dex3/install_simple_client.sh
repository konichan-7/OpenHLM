#!/usr/bin/env bash
# Install the patched SIMPLE client into an existing SIMPLE checkout.
#
#   bash install_simple_client.sh /path/to/SIMPLE
#
# Only copies one new file; the original `pi05.py` baseline is left untouched.
set -euo pipefail

SIMPLE_ROOT="${1:-${SIMPLE_ROOT:-}}"
if [[ -z "$SIMPLE_ROOT" ]]; then
  echo "usage: $0 /path/to/SIMPLE" >&2
  exit 1
fi
if [[ ! -f "$SIMPLE_ROOT/src/simple/baselines/pi05.py" ]]; then
  echo "not a SIMPLE checkout: $SIMPLE_ROOT" >&2
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$SIMPLE_ROOT/src/simple/baselines/pi05_openhlm.py"
cp "$HERE/simple_client/pi05_openhlm.py" "$TARGET"

cd "$SIMPLE_ROOT"
python - <<'PY'
import pathlib, sys
sys.path.insert(0, "src")
from simple.utils import snake_to_pascal
assert snake_to_pascal("pi05_openhlm") == "Pi05Openhlm", snake_to_pascal("pi05_openhlm")
src = pathlib.Path("src/simple/baselines/pi05_openhlm.py").read_text()
assert "class Pi05OpenhlmAgent" in src
print("client installed and class name resolves")
PY
