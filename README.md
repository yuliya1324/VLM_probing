# VLM Spatial Probing

Probing Vision-Language Models' internal representations for spatial relationship knowledge.

*The work is done by [Iuliia Korotkova](https://github.com/yuliya1324) and [Masayo Tomita](https://github.com/MTomita143)*

--------------------------------------------------------------------------------
Project Structure
--------------------------------------------------------------------------------

```bash
.
├── configs/                         # YAML configs for synthetic dataset generation
│   ├── spatial_dataset.yaml
│   └── color_dataset.yaml
│
├── data/
│   ├── raw/
│   │   ├── synthetic/              # Generated synthetic datasets
│   │   └── vrd/                    # Raw VRD dataset
│   ├── processed/
│   │   └── vrd/
│   │       ├── csv/                # Task-specific VRD CSVs
│   │       │   ├── vrd_base.csv
│   │       │   ├── vrd_spatial.csv
│   │       │   ├── vrd_color.csv
│   │       │   └── vrd_shape.csv
│   │       └── metadata/           # Optional metadata files
│   └── splits/                     # Train/val JSON splits
│
├── src/
│   ├── dataset_generation/         # Synthetic dataset generation
│   │   ├── spatial.py
│   │   ├── color.py
│   │   ├── renderer.py
│   │   └── schema.py
│   ├── data_preprocessing/         # VRD CSV helpers and preprocessing
│   │   ├── build_base_csv.py
│   │   ├── build_task_csv.py
│   │   └── vrd.py
│   ├── extraction/                 # Hidden-state extraction
│   │   ├── extract.py
│   │   └── io.py
│   ├── probing/                    # Probe training and inference
│   │   └── probe.py
│   ├── evaluation/
│   ├── plot/
│   ├── steering/
│   └── archive/
│
├── scripts/                        # Entry-point scripts
│   ├── generate_dataset.py
│   ├── extract_and_probe.py        # Synthetic / metadata-based pipeline
│   ├── extract_vrd.py              # VRD extraction
│   ├── evaluate.py                 # Probe evaluation from .npz
│   ├── evaluate_vrd_probe.py       # Probe predictions on VRD
│   ├── evaluate_vrd_raw.py         # Raw VLM response evaluation on VRD
│   ├── steer.py                    # Representation steering
│   └── evaluate_steering.py        # Quantitative steering evaluation
│
├── results/
│   ├── synthetic/
│   │   ├── spatial/
│   │   ├── color/
│   │   └── shape/
│   └── vrd/
│       ├── spatial/
|       |    ├── qwen2/
|       |    |    ├── representations.npz
|       |    |    ├── raw_response_predictions.csv
|       |    |    ├── probe_predictions.csv
|       |    |    └── correct/
|       |    |      ├── representations.npz
|       |    |      └── probes/
|       |    └── llava/
│       ├── color/
│       └── shape/
├── notebooks/
├── requirements.txt
├── requirements-extract.txt
└── README.md
```
--------------------------------------------------------------------------------
Pipeline
--------------------------------------------------------------------------------

1. Generate synthetic datasets
   Synthetic datasets are used for controlled probing experiments.

   - geometric images with ground-truth labels
   - tasks: spatial, color, shape

2. Extract hidden representations
   For each image and prompt:

   - run the VLM
   - collect hidden states from all layers
   - save last-token representations into representations.npz

3. Train linear probes
   For each layer:

   - train a logistic-regression probe
   - evaluate on the validation split
   - save the probe files and metadata

4. Evaluate
   We support several evaluation modes:

   - scripts/evaluate.py
     Evaluate probes across all layers from .npz representations

   - scripts/evaluate_vrd_raw.py
     Evaluate raw VLM responses on VRD

   - scripts/evaluate_vrd_probe.py
     Evaluate probe predictions on VRD representations

5. Steering
   Probe-derived directions can also be used to intervene on hidden states and
   steer model outputs toward target concepts.

--------------------------------------------------------------------------------
Environment Setup
--------------------------------------------------------------------------------

This project uses two environments.

1. Default environment
Use this for dataset generation, probe training, evaluation, and general
development.
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
2. Extraction environment
Use this for hidden-state extraction.

We use Python 3.11 for extraction because some model dependencies require newer
versions than the cluster default.
```
uv venv -p 3.11 venv-extract
source venv-extract/bin/activate
python -m ensurepip
python -m pip install --upgrade pip
python -m pip install -r requirements-extract.txt --no-deps
python -m pip install -e "git+https://github.com/NVlabs/VILA.git@b760c34b9487fd736b4075f5111fbef3d80a37e9#egg=vila" --no-deps
```
--------------------------------------------------------------------------------
Synthetic Dataset Generation
--------------------------------------------------------------------------------

Spatial dataset:
```
python scripts/generate_dataset.py --config configs/spatial_dataset.yaml
```
Color dataset:
```
python scripts/generate_dataset.py --config configs/color_dataset.yaml
```
--------------------------------------------------------------------------------
VRD Dataset Preparation
--------------------------------------------------------------------------------

1. Download VRD

This project uses the Visual Relationship Detection dataset from Kaggle.
```
kaggle datasets download apoorvshekher/visual-relationship-detection-vrd-dataset
```
Then unzip it under data/raw/vrd/.

2. Build VRD CSV files

We flatten VRD annotations into task-specific CSV files used for extraction and
evaluation.
```
python src/data_preprocessing/build_base_csv.py
python src/data_preprocessing/build_task_csv.py --task spatial
python src/data_preprocessing/build_task_csv.py --task color
python src/data_preprocessing/build_task_csv.py --task shape
```
The resulting files are stored in: `data/processed/vrd/csv/`

Notes:
- spatial labels are already normalized to left_of, right_of, above, below
- shape labels are already normalized, e.g. round -> circular
- too small objects are excluded

--------------------------------------------------------------------------------
Model Setup for VILA / SpatialRGPT
--------------------------------------------------------------------------------
```
git submodule update --init --recursive
python -m pip install -e ./VILA --no-deps
git apply patches/vila_local.patch
```
--------------------------------------------------------------------------------
Extraction and Probe Training
--------------------------------------------------------------------------------

Synthetic / metadata-based pipeline:
```
python scripts/extract_and_probe.py \
    --task spatial \
    --data_dir data/raw/synthetic/spatial \
    --model_tag qwen2 \
    --output_dir results/synthetic/spatial/qwen2
```

Example: retrain probes from an existing .npz
```
python scripts/extract_and_probe.py \
    --task color \
    --model_tag qwen2 \
    --output_dir results/vrd/color/qwen2/correct \
    --skip_extraction \
    --representations_path results/vrd/color/qwen2/correct/representations.npz
```
--------------------------------------------------------------------------------
Extract Hidden States on VRD
--------------------------------------------------------------------------------
```
python scripts/extract_vrd.py \
    --task spatial \
    --model_tag qwen2
```
This writes by default to: `results/vrd/spatial/qwen2/representations.npz`

Likewise for color and shape.

--------------------------------------------------------------------------------
Evaluate Probes Across Layers
--------------------------------------------------------------------------------
```
python scripts/evaluate.py \
    --probes_dir results/synthetic/spatial/qwen2/probes \
    --representations results/synthetic/spatial/qwen2/representations.npz \
    --output results/synthetic/spatial/qwen2/eval_all_layers.png \
    --per_class
```
Compare multiple runs:
```
python scripts/evaluate.py \
    --probes_dir \
        results/synthetic/spatial/qwen2/probes \
        results/synthetic/spatial/vila/probes \
    --representations \
        results/synthetic/spatial/qwen2/representations.npz \
        results/synthetic/spatial/vila/representations.npz \
    --labels "Qwen2-VL" "SpatialRGPT-VILA" \
    --output results/synthetic/spatial/comparison.png
```
If --representations is omitted, the script will look for:`<Path to probes parent>/representations.npz`

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
--------------------------------------------------------------------------------
Steering Evaluation
--------------------------------------------------------------------------------
```
python scripts/evaluate_steering.py \
    --probes_dir results/synthetic/spatial/qwen2/probes \
    --data_dir data/raw/synthetic/spatial \
    --task spatial \
    --model_tag qwen2 \
    --layers 20 \
    --alphas 0 1 2 5 10 20 \
    --output results/synthetic/spatial/qwen2/steering_eval.json \
    --plot results/synthetic/spatial/qwen2/steering_eval.png \
    --limit 100
```

--------------------------------------------------------------------------------
Notes
--------------------------------------------------------------------------------

- scripts/evaluate.py now supports .npz representations only
- VRD preprocessing is centralized in src/data_preprocessing/vrd.py
- VRD labels are assumed to be normalized at CSV creation time
- VRD image resizing is handled consistently during extraction and raw evaluation
- probe directions can also be used for representation steering via scripts/steer.py
