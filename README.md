# VLM Spatial Probing

Probing Vision-Language Models' internal representations for visual knowledge.

*The work is done by [Iuliia Korotkova](https://github.com/yuliya1324) and [Masayo Tomita](https://github.com/MTomita143)*

## Project Structure

```
./
├── src/
│   ├── dataset_generation/     # Synthetic image + label generation
│   ├── data_preprocessing/     # VRD CSV helpers and preprocessing
│   ├── data_postprocessing/    # VRD postprocessing
│   ├── extraction/             # Hidden state extraction from VLMs
│   │   ├── extract.py
│   │   └── io.py
│   ├── probing/                # Linear probe training & evaluation
│   │   └── probe.py
│   └── steering/               # Steering code
│       └── steer.py
├── scripts/                    # Entry-point scripts
│   ├── generate_dataset.py     # Script for generating synthetic dataset
│   ├── extract_and_probe.py    # Script for the whole pipeline extract & probe
│   ├── evaluate.py             # Script for probing evaluation
│   ├── evaluate_steering.py    # Script for steering evaluation
│   └── run_steering.py         # Script for steering
├── notebooks/                  
├── configs/                    # YAML configs for dataset generation & experiments
│   ├── spatial_dataset.yaml
│   └── color_dataset.yaml
├── requirements.txt
└── README.md
```

## Pipeline

1. **Generate synthetic datasets** (`scripts/generate_dataset.py`)
   - Spatial relations, color and shape identification

2. **Extract hidden representations** (`src/extraction/extract.py`)
   - Feed each image + prompt into a VLM
   - Save residual stream activations from all layers at the last prompt token

3. **Train linear probes** (`src/probing/probe.py`)
   - One-vs-rest logistic regression with L2 regularization
   - Trained per-layer on 80/20 train/val split

4. **Steering** (`src/steering/steer.py`)
    - Uses probes' weights to steer the VLMs' hidden representations.

## Environment Usage

```bash
uv venv -p 3.11 venv
source venv/bin/activate
python -m ensurepip
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --no-deps
python -m pip install -e "git+https://github.com/NVlabs/VILA.git@b760c34b9487fd736b4075f5111fbef3d80a37e9#egg=vila" --no-deps
```

Model Setup (SpacialRGBT)

```bash
git submodule update --init --recursive
python -m pip install -e ./VILA --no-deps
git apply patches/vila_local.patch
```

## 1. Dataset Preparation

Generate synthetic dataset:

```bash
python scripts/generate_dataset.py \
    --task {shape, color, spatial} \
    --config configs/{task}_dataset.yaml
```

Prepare VRD dataset:

1. Download VRD
This project uses the Visual Relationship Detection dataset from Kaggle.
```bash
kaggle datasets download apoorvshekher/visual-relationship-detection-vrd-dataset
```
Then unzip it under `data/raw/vrd/`.

2. Build VRD CSV files

We flatten VRD annotations into task-specific CSV files used for extraction and evaluation.

```bash
python src/data_preprocessing/build_base_csv.py
python src/data_preprocessing/build_task_csv.py --task {shape, color, spatial}
```
The resulting files are stored in: `data/processed/vrd/csv/`.

## 2. Extract Hiddens and Train the Probes

```bash
python scripts/extract_and_probe.py \
    --task=spatial \
    --data_dir=data/raw/spatial \
    --model_tag=llava15 \ # Choose between llava15 and qwen2
    --output_dir=path_to_the_output_dir \
    --train_split_path=path_to_the_train_split_json \ # optional
    --val_split_path=path_to_the_val_split_json \ # optional
    --skip_extraction # Skip extraction, use existing .npz (for re-running probes only)
```

Extract hidden states for VRD dataset:

```bash
python scripts/extract_vrd.py \
    --task spatial \
    --model_tag qwen2
```
This writes by default to: `results/vrd/spatial/qwen2/representations.npz`. Likewise for color and shape.

## 3. Probes Evaluation

```bash
# Single model — auto-finds representations.npz next to probes/
python scripts/evaluate.py \
    --probes_dir results/qwen2_spatial/probes \
    --split_json data/splits/spatial/val.json \
    --output results/qwen2_spatial/eval_plot.png \
    --per_class

# Compare models
python scripts/evaluate.py \
    --probes_dir results/qwen2_spatial/probes results/vila_spatial/probes \
    --labels "Qwen2-VL" "SpatialRGPT-VILA" "LLaVA-1.5" \
    --split_json data/splits/spatial/val.json \
    --output results/comparison.png
```

## 4. Steering


```bash
# Basic steering
python scripts/steer.py \
    --model_tag qwen2 \
    --image_path data/raw/spatial/images/spatial_00042.png \
    --prompt "Where is the red circle relative to the blue square?" \
    --probes_dir results/qwen2_spatial/probes \
    --layers 11 18 20 \ # can be a list or a single layer
    --target left_of \
    --alpha 10

# Contrast: push left, suppress right
python scripts/steer.py \
    --model_tag qwen2 \
    --image_path data/raw/spatial/images/spatial_00042.png \
    --prompt "Where is the red circle relative to the blue square?" \
    --probes_dir results/qwen2_spatial/probes \
    --layers 20 \
    --target left_of --source right_of \
    --alpha 10

# Sweep alpha to find the sweet spot
python scripts/steer.py \
    --model_tag qwen2 \
    --image_path data/raw/spatial/images/spatial_00042.png \
    --prompt "Where is the red circle relative to the blue square?" \
    --probes_dir results/qwen2_spatial/probes \
    --layers 20 \
    --target left_of \
    --sweep
```

Usage in a notebook (check `notebooks/steering`):

```python
from src.steering.steer import steer_and_generate, SteeringManager

# Quick one-liner
result = steer_and_generate(
    model, processor, "qwen2", image, prompt,
    probes_dir="results/qwen2_spatial/probes",
    layers=[18, 20, 22],
    target_class="left_of",
    alpha=5.0,
    when="prefill",
)

# Or manual control with context manager
with SteeringManager.from_probes(
    model, "qwen2", "results/qwen2_spatial/probes",
    layers=[18, 20, 22],
    target_class="left_of", alpha=5.0, when="prefill",
):
    output = _generate(model, processor, "qwen2", image, prompt, 50)
```


<details>

<summary>Usage of other scripts</summary>

--------------------------------------------------------------------------------
Evaluate Raw VLM Accuracy on VRD
--------------------------------------------------------------------------------

This evaluates the model’s generated answer directly, without probes.
```
python scripts/evaluate_vrd_raw.py \
    --task color \
    --model_tag qwen2 \
    --max_new_tokens 4
```
For spatial tasks, a larger max_new_tokens may be needed.
```
python scripts/evaluate_vrd_raw.py \
    --task spatial \
    --model_tag qwen2 \
    --max_new_tokens 6
```
Default output: `results/vrd/<task>/<model_tag>/raw_response_predictions.csv`

--------------------------------------------------------------------------------
Evaluate Probe Predictions on VRD
--------------------------------------------------------------------------------

This applies a trained probe to VRD representations and saves per-sample
predictions.
```
python scripts/evaluate_vrd_probe.py \
    --task spatial \
    --model_tag qwen2 \
    --probes_dir results/synthetic/spatial/qwen2/probes
```
Evaluate a specific layer:
```
python scripts/evaluate_vrd_probe.py \
    --task color \
    --model_tag qwen2 \
    --probes_dir results/synthetic/color/qwen2/probes \
    --layer 24
```
Default output: `results/vrd/<task>/<model_tag>/probe_predictions.csv`

--------------------------------------------------------------------------------
Re-evaluate Synthetic Probes on the VRD Correct Subset
--------------------------------------------------------------------------------

After running evaluate_vrd_raw.py, you can create a correct-only VRD subset and
evaluate synthetic probes on it.
```
python scripts/make_correct_subset.py \
    --pred_csv results/vrd/color/qwen2/raw_response_predictions.csv \
    --repr_npz results/vrd/color/qwen2/representations.npz \
    --out_npz results/vrd/color/qwen2/correct/representations.npz
```
Then evaluate:
```
python scripts/evaluate.py \
    --probes_dir results/synthetic/color/qwen2/probes \
    --representations results/vrd/color/qwen2/correct/representations.npz \
    --split all \
    --output results/vrd/color/qwen2/correct/eval_with_synth_probes.png
```
--------------------------------------------------------------------------------
Train Probes on the VRD Correct Subset
--------------------------------------------------------------------------------
```
python scripts/extract_and_probe.py \
    --task color \
    --model_tag qwen2 \
    --output_dir results/vrd/color/qwen2/correct \
    --skip_extraction \
    --representations_path results/vrd/color/qwen2/correct/representations.npz
```
--------------------------------------------------------------------------------
Mixed Probes on VRD
--------------------------------------------------------------------------------

1. Evaluate a probe trained on the correct subset
```
python scripts/evaluate.py \
    --probes_dir results/vrd/spatial/qwen2/correct/probes \
    --representations results/vrd/spatial/qwen2/representations.npz \
    --split all \
    --output results/vrd/spatial/qwen2/correct/eval_on_full_vrd.png
```
2. Build a mixed .npz
```
python scripts/make_mixed_npz.py \
    --vrd results/vrd/spatial/qwen2/correct/representations.npz \
    --synthetic results/synthetic/spatial/qwen2/representations.npz \
    --output results/vrd/spatial/qwen2/mixed/representations.npz
```
3. Train probes on the mixed dataset
```
python scripts/extract_and_probe.py \
    --task spatial \
    --model_tag qwen2 \
    --output_dir results/vrd/spatial/qwen2/mixed \
    --skip_extraction \
    --representations_path results/vrd/spatial/qwen2/mixed/representations.npz
```
4. Evaluate on full VRD
```
python scripts/evaluate.py \
    --probes_dir results/vrd/spatial/qwen2/mixed/probes \
    --representations results/vrd/spatial/qwen2/representations.npz \
    --split all \
    --output results/vrd/spatial/qwen2/mixed/eval_on_full_vrd.png
```

</details>