#!/usr/bin/env bash
# Semantic evaluation (single GPU / single job). Set TC_DN_MT_LANG_PAIR to match your data (en-zh, en-de, ...).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

INPUT="${TC_DN_MT_SEMANTIC_INPUT:-./translation_results}"
OUTPUT="${TC_DN_MT_SEMANTIC_OUTPUT:-./evaluation_results_semantic}"
STEP="${TC_DN_MT_SEMANTIC_STEP:-10}"
LANG_PAIR="${TC_DN_MT_LANG_PAIR:-en-zh}"
GPU="${TC_DN_MT_SEMANTIC_GPU:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"
echo "Semantic eval: input=$INPUT output=$OUTPUT step=$STEP lang_pair=$LANG_PAIR CUDA_VISIBLE_DEVICES=$GPU"
python before_semantic.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --step "$STEP" \
  --lang_pair "$LANG_PAIR"

echo "Semantic evaluation finished."
