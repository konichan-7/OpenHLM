#!/usr/bin/env bash
# Provision + verify the SIMPLE G1/Dex3 training box, from a bare A100 to a
# running policy server.
#
# Usage (on the server, from openpi4OpenHLM/):
#   bash examples/simple_g1_dex3/deploy_server.sh data      # download + audit + split
#   bash examples/simple_g1_dex3/deploy_server.sh env       # uv sync (skips evdev/pynput)
#   bash examples/simple_g1_dex3/deploy_server.sh weights   # pi0.5 base -> PyTorch
#   bash examples/simple_g1_dex3/deploy_server.sh stats     # norm stats on the train split
#   bash examples/simple_g1_dex3/deploy_server.sh check     # protocol + e2e pipeline
#   bash examples/simple_g1_dex3/deploy_server.sh smoke     # 100-step training smoke
#   bash examples/simple_g1_dex3/deploy_server.sh train     # full run (resumable)
#   bash examples/simple_g1_dex3/deploy_server.sh serve     # websocket policy server
#
# Everything is idempotent; nothing is deleted.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/data/psi_simple_tabletop/lerobot}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$ROOT}"
REPO_ID="${REPO_ID:-local/g1_dex3_tabletop}"
DATASET_DIR="$HF_LEROBOT_HOME/$REPO_ID"
DATA_ROOT="${DATA_ROOT:-/data/psi_simple_tabletop}"
MANIFEST="${MANIFEST:-assets/simple_g1_dex3/tabletop_split.json}"
BASE_URL="https://huggingface.co/datasets/USC-PSI-Lab/psi-data/resolve/main"
CONFIG=pi05_simple_g1_dex3_tabletop
PI05_DIR="$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch"
PI05_SHA256=0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f

log() { printf '\n=== %s ===\n' "$*"; }

download() {
  local name="$1" url="$2" dest="$3"
  if [[ -f "$dest" ]]; then echo "have $dest"; return; fi
  echo "downloading $url"
  mkdir -p "$(dirname "$dest")"
  curl -fL --retry 3 -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

case "${1:-help}" in
data)
  log "download dataset"
  mkdir -p "$DATA_ROOT"
  # Accept a package uploaded by hand (the server may not reach huggingface.co).
  if [[ -f "$DATA_ROOT/train.zip" && ! -f "$DATA_ROOT/G1WholebodyTabletopGraspMP-v0.zip" ]]; then
    mv "$DATA_ROOT/train.zip" "$DATA_ROOT/G1WholebodyTabletopGraspMP-v0.zip"
  fi
  download train "$BASE_URL/simple/G1WholebodyTabletopGraspMP-v0.zip" "$DATA_ROOT/G1WholebodyTabletopGraspMP-v0.zip"
  download eval "$BASE_URL/simple-eval/G1WholebodyTabletopGraspMP-v0.zip" "$DATA_ROOT/G1WholebodyTabletopGraspMP-v0.eval.zip"

  log "unpack into LeRobot layout"
  if [[ ! -d "$DATASET_DIR" ]]; then
    mkdir -p "$HF_LEROBOT_HOME/local"
    tmp="$DATA_ROOT/_unpack.$$"
    mkdir -p "$tmp"
    unzip -q -o "$DATA_ROOT/G1WholebodyTabletopGraspMP-v0.zip" -d "$tmp"
    mv "$tmp/G1WholebodyTabletopGraspMP-v0" "$DATASET_DIR"
    rmdir "$tmp"
  fi
  ls "$DATASET_DIR/meta"

  log "audit"
  python examples/simple_g1_dex3/audit_data.py --dataset-root "$DATASET_DIR" --check-videos

  log "split"
  if [[ ! -f "$MANIFEST" ]]; then
    python examples/simple_g1_dex3/make_split.py --dataset-root "$DATASET_DIR" --out "$MANIFEST" --val-fraction 0.2 --seed 0
  else
    echo "have $MANIFEST"
  fi
  ;;

env)
  log "uv sync"
  uv sync --no-install-package evdev --no-install-package pynput
  log "apply OpenHLM transformers patch"
  cp -r src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
  uv run python -c "import torch, jax; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('jax', jax.__version__)"
  ;;

weights)
  log "pi0.5 base PyTorch weights"
  mkdir -p "$PI05_DIR"
  model="$PI05_DIR/model.safetensors"
  if [[ ! -f "$model" ]]; then
    # huggingface.co is not reachable from the training cluster; ModelScope
    # mirrors the identical lerobot/pi05_base safetensors.
    curl -fL --retry 3 -C - -o "$model.part" \
      "https://modelscope.cn/models/lerobot/pi05_base/resolve/master/model.safetensors"
    mv "$model.part" "$model"
  fi
  echo "$PI05_SHA256  $model" | sha256sum -c -
  ;;

stats)
  log "norm stats (train split only)"
  PYTHONPATH=src uv run python scripts/compute_norm_stats.py --config-name "$CONFIG" --episode-manifest "$MANIFEST"
  ;;

check)
  log "protocol tests"
  PYTHONPATH=src uv run python examples/simple_g1_dex3/test_protocol.py
  log "e2e pipeline"
  HF_LEROBOT_HOME="$HF_LEROBOT_HOME" PYTHONPATH=src uv run python examples/simple_g1_dex3/test_pipeline_e2e.py \
    --repo-id "$REPO_ID" --config-name "$CONFIG" --episode-manifest "$MANIFEST"
  log "websocket serve smoke"
  PYTHONPATH=src uv run python examples/simple_g1_dex3/test_serve.py
  ;;

smoke)
  log "100-step smoke (batch 1, no wandb)"
  PYTHONPATH=src uv run torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/train_pytorch.py \
    "$CONFIG" --exp_name smoke \
    --episode-manifest "$MANIFEST" \
    --batch_size 1 --num_train_steps 100 --save_interval 50 --log_interval 10 \
    --no-wandb-enabled --overwrite
  ;;

train)
  log "training"
  resume_flag=""
  if [[ "${RESUME:-0}" == "1" ]]; then resume_flag="--resume"; fi
  PYTHONPATH=src uv run torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/train_pytorch.py \
    "$CONFIG" --exp_name "${EXP_NAME:-run1}" \
    --episode-manifest "$MANIFEST" \
    --no-wandb-enabled $resume_flag
  ;;

serve)
  log "policy server"
  CKPT="${CKPT:?set CKPT=checkpoints/$CONFIG/<exp>/<step>}"
  PYTHONPATH=src uv run python scripts/serve_policy.py \
    --port "${PORT:-8000}" policy:checkpoint --policy.config "$CONFIG" --policy.dir "$CKPT"
  ;;

*)
  sed -n '2,20p' "$0"
  exit 1
  ;;
esac
