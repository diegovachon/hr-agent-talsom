#!/usr/bin/env bash
# scripts/download_model.sh
#
# Downloads the Llama 3 8B Instruct Q4_K_M GGUF from HuggingFace.
#
# Usage :
#   bash scripts/download_model.sh
#
# Requirements :
#   pip install huggingface_hub
#   huggingface-cli login   (only needed for gated models)

set -euo pipefail

MODEL_DIR="./models"
MODEL_FILE="Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
HF_REPO="bartowski/Meta-Llama-3-8B-Instruct-GGUF"

echo "=================================================="
echo "Llama 3 8B Instruct — Q4_K_M GGUF downloader"
echo "=================================================="
echo ""
echo "Repo  : $HF_REPO"
echo "File  : $MODEL_FILE"
echo "Dest  : $MODEL_DIR/$MODEL_FILE"
echo ""

# ── Checks ────────────────────────────────────────────────────────────────────

if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "Model already exists at $MODEL_DIR/$MODEL_FILE"
    echo "Delete the file and re-run to force a fresh download."
    exit 0
fi

if ! command -v python3 &> /dev/null; then
    echo "Error : python3 not found. Install Python 3.10+ and try again."
    exit 1
fi

if ! python3 -c "import huggingface_hub" &> /dev/null; then
    echo "huggingface_hub not found. Installing..."
    pip install huggingface_hub
fi

# ── Download ──────────────────────────────────────────────────────────────────

mkdir -p "$MODEL_DIR"

echo "Starting download (~4.6 GB). This will take a few minutes..."
echo ""

python3 - <<EOF
from huggingface_hub import hf_hub_download
import shutil, pathlib

path = hf_hub_download(
    repo_id   = "$HF_REPO",
    filename  = "$MODEL_FILE",
    local_dir = "$MODEL_DIR",
)
print(f"Downloaded to : {path}")
EOF

echo ""
echo "=================================================="
echo "Done. Model saved to $MODEL_DIR/$MODEL_FILE"
echo ""
echo "Next steps :"
echo "  1. python scripts/ingest_all.py"
echo "  2. python -m src.ui.app"
echo "=================================================="
