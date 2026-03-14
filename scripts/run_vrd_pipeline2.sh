#!/usr/bin/env bash
set -euo pipefail

LOG="logs_vrd_pipeline.txt"

echo "=============================" | tee -a $LOG
date | tee -a $LOG
echo "Starting VRD pipeline" | tee -a $LOG
echo "=============================" | tee -a $LOG


run() {
    echo "" | tee -a $LOG
    echo ">>> RUN: $*" | tee -a $LOG
    "$@" 2>&1 | tee -a $LOG
}


# -----------------------------
# SPATIAL
# -----------------------------
run python scripts/evaluate_vrd_raw.py \
    --task spatial \
    --model_tag qwen2 \
    --max_new_tokens 6 \
    --resume

run python scripts/make_correct_subset.py \
    --pred_csv results/vrd_spatial/qwen2/raw_response_predictions.csv \
    --repr_npz results/vrd_spatial/qwen2/representations.npz \
    --out_npz results/vrd_spatial/qwen2/correct/representations.npz

run python scripts/evaluate.py \
    --probes_dir results/spatial/qwen2/probes \
    --representations results/vrd_spatial/qwen2/correct/representations.npz \
    --split all \
    --output results/vrd_spatial/qwen2/correct/eval_with_synth_probes.png

run python scripts/extract_and_probe.py \
    --task spatial \
    --model_tag qwen2 \
    --output_dir results/vrd_spatial/qwen2/correct \
    --skip_extraction


# -----------------------------
# SHAPE
# -----------------------------
run python scripts/extract_vrd.py \
    --task shape \
    --model_tag qwen2

run python scripts/evaluate_vrd_raw.py \
    --task shape \
    --model_tag qwen2 \
    --max_new_tokens 4 \
    --resume

run python scripts/make_correct_subset.py \
    --pred_csv results/vrd_shape/qwen2/raw_response_predictions.csv \
    --repr_npz results/vrd_shape/qwen2/representations.npz \
    --out_npz results/vrd_shape/qwen2/correct/representations.npz

run python scripts/evaluate.py \
    --probes_dir results/shape/qwen2/probes \
    --representations results/vrd_shape/qwen2/correct/representations.npz \
    --split all \
    --output results/vrd_shape/qwen2/correct/eval_with_synth_probes.png

run python scripts/extract_and_probe.py \
    --task shape \
    --model_tag qwen2 \
    --output_dir results/vrd_shape/qwen2/correct \
    --skip_extraction


echo "" | tee -a $LOG
echo "Pipeline finished" | tee -a $LOG
date | tee -a $LOG