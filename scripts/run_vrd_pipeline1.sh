#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# SPATIAL
# -----------------------------
python scripts/extract_vrd.py --task spatial --model_tag qwen2 &&

python scripts/evaluate_vrd_raw.py \
    --task spatial \
    --model_tag qwen2 \
    --max_new_tokens 6 &&

python scripts/make_correct_subset.py \
    --pred_csv results/vrd_spatial/qwen2/raw_response_predictions.csv \
    --repr_npz results/vrd_spatial/qwen2/representations.npz \
    --out_npz  results/vrd_spatial/qwen2/correct/representations.npz &&

python scripts/evaluate.py \
    --probes_dir results/spatial/qwen2/probes \
    --representations results/vrd_spatial/qwen2/correct/representations.npz \
    --split all \
    --output results/vrd_spatial/qwen2/correct/eval_with_synth_probes.png &&

python scripts/extract_and_probe.py \
    --task spatial \
    --model_tag qwen2 \
    --output_dir results/vrd_spatial/qwen2/correct \
    --skip_extraction ;

# -----------------------------
# SHAPE
# -----------------------------
python scripts/extract_vrd.py --task shape --model_tag qwen2 &&

python scripts/evaluate_vrd_raw.py \
    --task shape \
    --model_tag qwen2 \
    --max_new_tokens 4 &&

python scripts/make_correct_subset.py \
    --pred_csv results/vrd_shape/qwen2/raw_response_predictions.csv \
    --repr_npz results/vrd_shape/qwen2/representations.npz \
    --out_npz  results/vrd_shape/qwen2/correct/representations.npz &&

python scripts/evaluate.py \
    --probes_dir results/shape/qwen2/probes \
    --representations results/vrd_shape/qwen2/correct/representations.npz \
    --split all \
    --output results/vrd_shape/qwen2/correct/eval_with_synth_probes.png &&

python scripts/extract_and_probe.py \
    --task shape \
    --model_tag qwen2 \
    --output_dir results/vrd_shape/qwen2/correct \
    --skip_extraction