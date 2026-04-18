#!/usr/bin/env bash
# Batch lexical evaluation (single job). Override paths with env vars.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

INPUT="${TC_DN_MT_LEXICAL_INPUT:-./translation_results}"
OUTPUT="${TC_DN_MT_LEXICAL_OUTPUT:-./evaluation_results_lexical}"
DEVICE="${TC_DN_MT_DEVICE:-cuda:0}"
SKIP="${TC_DN_MT_SKIP_EXISTING:-true}"

echo "Lexical eval: input=$INPUT output=$OUTPUT device=$DEVICE"
python before_lexical.py \
  --input_folder "$INPUT" \
  --output_folder "$OUTPUT" \
  --device "$DEVICE" \
  --skip_existing "$SKIP"

echo "Lexical evaluation finished."
