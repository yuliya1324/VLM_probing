#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_vrd_pipeline.sh qwen2
#   bash scripts/run_vrd_pipeline.sh vila
#   bash scripts/run_vrd_pipeline.sh llava15
#
# If no argument is given, default to qwen2.

MODELTAG="${1:-qwen2}"

case "$MODELTAG" in
    qwen2|vila|llava15)
        ;;
    *)
        echo "Error: unsupported model_tag '$MODELTAG'"
        echo "Supported values: qwen2, vila, llava15"
        exit 1
        ;;
esac

LOG="logs_vrd_pipeline_${MODELTAG}.txt"

echo "=============================" | tee -a "$LOG"
date | tee -a "$LOG"
echo "Starting VRD pipeline for model: $MODELTAG" | tee -a "$LOG"
echo "=============================" | tee -a "$LOG"

run() {
    echo "" | tee -a "$LOG"
    echo ">>> RUN: $*" | tee -a "$LOG"
    "$@" 2>&1 | tee -a "$LOG"
}

delete_existing_extract_outputs() {
    local task="$1"
    local base_dir="results/vrd_${task}/${MODELTAG}"
    local repr_path="${base_dir}/representations.npz"
    local chunk_dir="${base_dir}/representations_chunks"

    if [[ -e "$repr_path" ]]; then
        echo "Removing old file: $repr_path" | tee -a "$LOG"
        rm -f "$repr_path"
    fi

    if [[ -d "$chunk_dir" ]]; then
        echo "Removing old chunk directory: $chunk_dir" | tee -a "$LOG"
        rm -rf "$chunk_dir"
    fi
}

# -----------------------------
# Optional cache cleanup
# -----------------------------
rm -rf ~/.cache/*

: << 'SKIP_COLOR'

# -----------------------------
# COLOR
# -----------------------------
run python scripts/evaluate_vrd_raw.py \
    --task color \
    --model_tag "$MODELTAG" \
    --max_new_tokens 4

run python scripts/make_correct_subset.py \
    --pred_csv "results/vrd_color/${MODELTAG}/raw_response_predictions.csv" \
    --repr_npz "results/vrd_color/${MODELTAG}/representations.npz" \
    --out_npz "results/vrd_color/${MODELTAG}/correct/representations.npz"

run python scripts/evaluate.py \
    --probes_dir "results/color/${MODELTAG}/probes" \
    --representations "results/vrd_color/${MODELTAG}/correct/representations.npz" \
    --split all \
    --output "results/vrd_color/${MODELTAG}/correct/eval_with_synth_probes.png"

run python scripts/extract_and_probe.py \
    --task color \
    --model_tag "$MODELTAG" \
    --output_dir "results/vrd_color/${MODELTAG}/correct" \
    --skip_extraction

SKIP_COLOR


# -----------------------------
# SPATIAL
# -----------------------------
: << 'SKIP_SPATIAL'
delete_existing_extract_outputs spatial

run python scripts/extract_vrd.py \
    --task spatial \
    --model_tag "$MODELTAG"

run python scripts/evaluate_vrd_raw.py \
    --task spatial \
    --model_tag "$MODELTAG" \
    --max_new_tokens 6

run python scripts/make_correct_subset.py \
    --pred_csv "results/vrd_spatial/${MODELTAG}/raw_response_predictions.csv" \
    --repr_npz "results/vrd_spatial/${MODELTAG}/representations.npz" \
    --out_npz "results/vrd_spatial/${MODELTAG}/correct/representations.npz"

run python scripts/evaluate.py \
    --probes_dir "results/spatial/${MODELTAG}/probes" \
    --representations "results/vrd_spatial/${MODELTAG}/correct/representations.npz" \
    --split all \
    --output "results/vrd_spatial/${MODELTAG}/correct/eval_with_synth_probes.png"

run python scripts/extract_and_probe.py \
    --task spatial \
    --model_tag "$MODELTAG" \
    --output_dir "results/vrd_spatial/${MODELTAG}/correct" \
    --skip_extraction

SKIP_SPATIAL

# -----------------------------
# SHAPE
# -----------------------------
delete_existing_extract_outputs shape

run python scripts/extract_vrd.py \
     --task shape \
     --model_tag "$MODELTAG"

run python scripts/evaluate_vrd_raw.py \
    --task shape \
    --model_tag "$MODELTAG" \
    --max_new_tokens 4 \
    --resume

run python scripts/make_correct_subset.py \
    --pred_csv "results/vrd_shape/${MODELTAG}/raw_response_predictions.csv" \
    --repr_npz "results/vrd_shape/${MODELTAG}/representations.npz" \
    --out_npz "results/vrd_shape/${MODELTAG}/correct/representations.npz"

run python scripts/evaluate.py \
    --probes_dir "results/shape/${MODELTAG}/probes" \
    --representations "results/vrd_shape/${MODELTAG}/correct/representations.npz" \
    --split all \
    --output "results/vrd_shape/${MODELTAG}/correct/eval_with_synth_probes.png"

run python scripts/extract_and_probe.py \
    --task shape \
    --model_tag "$MODELTAG" \
    --output_dir "results/vrd_shape/${MODELTAG}/correct" \
    --skip_extraction

echo "" | tee -a "$LOG"
echo "Pipeline finished for model: $MODELTAG" | tee -a "$LOG"
date | tee -a "$LOG"