# VLM Spatial Probing

Probing Vision-Language Models' internal representations for spatial relationship knowledge.

*The work is done by [Iuliia Korotkova](https://github.com/yuliya1324) and [Masayo Tomita](https://github.com/MTomita143)*

## Project Structure

```
.
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
   - Spatial relations: images of geometric shapes with ground-truth relations
   - Color identification: simpler task as a sanity check for the probing pipeline

2. **Extract hidden representations** (`src/extraction/extract.py`)
   - Feed each image + prompt into a VLM
   - Save residual stream activations from all layers at the last prompt token

3. **Train linear probes** (`src/probing/probe.py`)
   - One-vs-rest logistic regression with L2 regularization
   - Trained per-layer on 80/20 train/val split

# Quick Start

## Environment Usage

We use Python 3.11.

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

## Generate dataset

```bash
# Generate spatial dataset (default: 3000 samples)
python scripts/generate_dataset.py --config configs/spatial_dataset.yaml

# Generate color dataset (default: 1000 samples)
python scripts/generate_dataset.py --config configs/color_dataset.yaml

# Generate shape dataset (default: 1000 samples)
python scripts/generate_dataset.py --task shape
```

## Download VRD Dataset
This project uses the Visual Relationship Detection (VRD) dataset from Kaggle.
To download it:
```bash
kaggle datasets download apoorvshekher/visual-relationship-detection-vrd-dataset
```
Then unzip

## Create CSV from VRD Dataset
We convert the Visual Relationship Detection (VRD) annotations into a flat CSV file used for probing Vision-Language Models.
Run:
```bash
python src/dataset_to_csv.py
```
Output:
`~/data/vrd_relationships.csv`
Notes
- Only samples with existing images are included.
- One image may produce multiple rows (one per relationship).

## Extract Hiddens and Train the Probes

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

## Extract Hidden States for VRD dataset

```bash
python src/lasttoken/extract_{llava15, qwen2, spatialRGBT}.py
```

## Evaluate

Example of evaluation with representations in `representations.npz`

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

Example of evaluation with representations in `.pt` files

```bash
python scripts/evaluate.py \
    --probes_dir results/qwen2_spatial/probes \
    --pt_dir features/Qwen2-VL \
    --output results/qwen2_spatial/eval_plot.png
```

## Steering


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