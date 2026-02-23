#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

MODE="${1:-all}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

: "${SMOKE_TEST:=1}"
: "${SMOKE_IMG_SIZE:=224}"
: "${SMOKE_MAX_SAMPLES:=64}"

echo "REPO_DIR=$SCRIPT_DIR"
echo "SMOKE_TEST=$SMOKE_TEST"
echo "SMOKE_IMG_SIZE=$SMOKE_IMG_SIZE"
echo "SMOKE_MAX_SAMPLES=$SMOKE_MAX_SAMPLES"

usage() {
  echo "Usage: $0 [setup|smoke|all]"
}

if [ ! -f "SETTINGS.json" ]; then
  echo "ERROR: SETTINGS.json not found in $SCRIPT_DIR"
  exit 1
fi

RAW_DATA_DIR=$(python - <<'PY'
import json
from pathlib import Path
settings = json.load(open('SETTINGS.json','r',encoding='utf-8'))
raw = settings.get('RAW_DATA_DIR','./data/raw')
p = Path(raw).expanduser()
if not p.is_absolute():
    p = (Path('.').resolve() / p).resolve()
print(str(p))
PY
)

echo "RAW_DATA_DIR=$RAW_DATA_DIR"

do_setup() {
  echo "Installing Python dependencies..."
  if grep -qE '^torch([<>=!~]=|==|>=|<=|~=|!=)' "$SCRIPT_DIR/requirements.txt"; then
    REQ_NO_TORCH="/tmp/requirements_no_torch.txt"
    grep -vE '^torch([<>=!~]=|==|>=|<=|~=|!=)' "$SCRIPT_DIR/requirements.txt" > "$REQ_NO_TORCH"
    python -m pip -q install -r "$REQ_NO_TORCH"
  else
    python -m pip -q install -r "$SCRIPT_DIR/requirements.txt"
  fi

  echo "Installing Kaggle CLI..."
  python -m pip -q install kaggle

  mkdir -p ~/.kaggle
  if [ ! -f ~/.kaggle/kaggle.json ]; then
    if [ -f /content/kaggle.json ]; then
      cp /content/kaggle.json ~/.kaggle/kaggle.json
      chmod 600 ~/.kaggle/kaggle.json
    else
      echo "ERROR: Kaggle token not found. Upload kaggle.json to /content (Kaggle -> Account -> API -> Create New Token)."
      exit 1
    fi
  fi

  mkdir -p "$RAW_DATA_DIR"

  if [ -f "$RAW_DATA_DIR/train.csv" ] && [ -d "$RAW_DATA_DIR/train" ]; then
    echo "Dataset already present, skipping download."
  else
    echo "Downloading Kaggle competition data (csiro-biomass)..."
    kaggle competitions download -c csiro-biomass -p "$RAW_DATA_DIR"

    echo "Unzipping downloaded archives..."
    cd "$RAW_DATA_DIR"
    for z in *.zip; do
      unzip -q "$z"
    done
    cd "$SCRIPT_DIR"
  fi

  if [ ! -f "$RAW_DATA_DIR/train.csv" ] || [ ! -f "$RAW_DATA_DIR/test.csv" ] || [ ! -d "$RAW_DATA_DIR/train" ] || [ ! -d "$RAW_DATA_DIR/test" ]; then
    echo "ERROR: dataset layout invalid under $RAW_DATA_DIR"
    echo "Expected: train.csv, test.csv, train/, test/"
    ls -la "$RAW_DATA_DIR" || true
    exit 1
  fi
}

do_smoke() {
  if [ ! -f "$RAW_DATA_DIR/train.csv" ] || [ ! -f "$RAW_DATA_DIR/test.csv" ] || [ ! -d "$RAW_DATA_DIR/train" ] || [ ! -d "$RAW_DATA_DIR/test" ]; then
    echo "ERROR: dataset layout invalid under $RAW_DATA_DIR"
    echo "Expected: train.csv, test.csv, train/, test/"
    echo "Run: $0 setup"
    exit 1
  fi

  echo "Running smoke tests..."
  export SMOKE_TEST SMOKE_IMG_SIZE SMOKE_MAX_SAMPLES

  python modelA_and_specialist_train.py
  python modelB_and_specialist_train.py
  python model_high_specialist_train.py
}

case "$MODE" in
  setup)
    do_setup
    ;;
  smoke)
    do_smoke
    ;;
  all)
    do_setup
    do_smoke
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac

echo "DONE"
