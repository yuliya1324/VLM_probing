# VLM Spatial Probing

Probing Vision-Language Models' internal representations for spatial relationship knowledge.

*The work is done by [Iuliia Korotkova](https://github.com/yuliya1324) and [Masayo Tomita](https://github.com/MTomita143)*

## Project Structure

```
vlm-spatial-probing/
├── configs/                    # YAML configs for dataset generation & experiments
│   ├── spatial_dataset.yaml
│   └── color_dataset.yaml
├── data/
│   ├── raw/                    # Generated images
│        └── vrd                # Dataset - Visual Relationship Detection
│   └── splits/                 # Train/val JSON splits
├── src/
│   ├── dataset_generation/     # Synthetic image + label generation
│   │   ├── spatial.py          # Spatial relationship dataset
│   │   ├── color.py            # Color identification dataset (sanity check)
│   │   ├── renderer.py         # Shape rendering engine
│   │   └── schema.py           # Data schemas / types
│   ├── extraction/             # Hidden state extraction from VLMs
│       └── extract.py
│   └── probing/                # Linear probe training & evaluation
│       └── probe.py
├── scripts/                    # Entry-point scripts
│   ├── extract_and_probe.py    # Script for the whole pipeline extract & probe
│   └── generate_dataset.py     # Script for generating the dataset
├── notebooks/                  # Exploratory notebooks
├── results/                    # Saved probe results, plots
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

This project uses TWO environments depending on the task.

### Default environment (development)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### Extract (hidden states, Python 3.11)
We use Python 3.11 for extraction (cluster default is 3.9).

Reason:
Some models require Python 3.11 + specific CUDA/bitsandbytes versions.

```bash
uv venv -p 3.11 venv-extract
source venv-extract/bin/activate
python -m ensurepip
python -m pip install --upgrade pip
python -m pip install -r requirements-extract.txt --no-deps
python -m pip install -e "git+https://github.com/NVlabs/VILA.git@b760c34b9487fd736b4075f5111fbef3d80a37e9#egg=vila" --no-deps
# pip install -r requirements-extract.txt
```
Run:
```bash
./scripts/run_extract.sh
```

⚠️ Do NOT run extraction scripts inside venv.

## Generate dataset

```bash
# Generate spatial dataset (default: 3000 samples)
python scripts/generate_dataset.py --config configs/spatial_dataset.yaml

# Generate color dataset (sanity check, default: 1000 samples)
python scripts/generate_dataset.py --config configs/color_dataset.yaml
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

## Model Setup (SpacialRGBT)

```bash
git submodule update --init --recursive
python -m pip install -e ./VILA --no-deps
git apply patches/vila_local.patch
```

## Extract Hidden States

```bash
python src/lasttoken/extract_{llava15, qwen2, spatialRGBT}.py
```
⚠️ Make sure you are on `venv-extract`

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