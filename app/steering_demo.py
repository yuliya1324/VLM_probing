#!/usr/bin/env python3
"""Streamlit interface for interactive VLM steering.

Upload an image, type a prompt, pick a probe (color/shape/spatial),
choose a target class, adjust alpha, and see how the model's response changes.

Usage:
    streamlit run app/steering_demo.py -- \
        --model_tag qwen2 \
        --probes_color results/qwen2_color/probes \
        --probes_shape results/qwen2_shape/probes \
        --probes_spatial results/qwen2_spatial/probes

Or configure via the sidebar after launch.
"""

import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import torch
from PIL import Image

from src.extraction.extract import MODEL_REGISTRY
from src.steering.steer import SteeringManager, _generate, get_steering_vector


# ============================================================
# Task / class definitions
# ============================================================

TASK_CLASSES = {
    "color": [
        "white", "black", "blue", "brown", "green",
        "red", "yellow", "gray", "orange", "pink", "purple",
    ],
    "shape": [
        "circular", "oval", "square", "rectangular", "triangular",
    ],
    "spatial": [
        "left_of", "right_of", "above", "below",
    ],
}

PROBE_DIRS = {
    "qwen2": {
            "color": "/Data/iuliia.korotkova/VLM_probing/color/qwen2/probes",
            "shape": "/Data/iuliia.korotkova/VLM_probing/shape/qwen2/probes",
            "spatial": "/Data/iuliia.korotkova/VLM_probing/spatial/qwen2/probes",
        },
    "llava15": {
            "color": "/Data/iuliia.korotkova/VLM_probing/color/llava15/probes",
            "shape": "/Data/iuliia.korotkova/VLM_probing/shape/llava15/probes",
            "spatial": "/Data/iuliia.korotkova/VLM_probing/spatial/llava15/probes",
        },
    "vila": {
            "color": "/Data/iuliia.korotkova/VLM_probing/color/vila/probes",
            "shape": "/Data/iuliia.korotkova/VLM_probing/shape/vila/probes",
            "spatial": "/Data/iuliia.korotkova/VLM_probing/spatial/vila/probes",
        },
}

DEFAULTS = {
    ("qwen2", "color"): {
        "layers": (20, 28),
        "alpha": 10.0,
        "when": "all",
        "strategy": "push",
    },
    ("qwen2", "shape"): {
        "layers": (20, 28),
        "alpha": 10.0,
        "when": "all",
        "strategy": "push",
    },
    ("qwen2", "spatial"): {
        "layers": (20, 28),
        "alpha": 10.0,
        "when": "all",
        "strategy": "push",
    },
    ("llava15", "color"): {
        "layers": (20, 32),
        "alpha": 3.0,
        "when": "all",
        "strategy": "push",
    },
    ("llava15", "shape"): {
        "layers": (15, 32),
        "alpha": 3.0,
        "when": "all",
        "strategy": "push",
    },
    ("llava15", "spatial"): {
        "layers": (15, 32),
        "alpha": 3.0,
        "when": "all",
        "strategy": "push",
    },
    ("vila", "color"): {
        "layers": (24, 30),
        "alpha": 5.0,
        "when": "all",
        "strategy": "push",
    },
    ("vila", "shape"): {
        "layers": (23, 32),
        "alpha": 5.0,
        "when": "all",
        "strategy": "push",
    },
    ("vila", "spatial"): {
        "layers": (25, 30),
        "alpha": 5.0,
        "when": "all",
        "strategy": "push",
    },
}

# Fallback if no specific default exists
DEFAULT_FALLBACK = {
    "layers": (20, 28),
    "alpha": 10.0,
    "when": "all",
    "strategy": "push",
}

def get_defaults(model_tag: str, task: str) -> dict:
    return DEFAULTS.get((model_tag, task), DEFAULT_FALLBACK)

# ============================================================
# Model loading (cached so it only loads once)
# ============================================================

@st.cache_resource
def load_model(model_tag: str, model_id: str = None):
    """Load model and processor, cached across reruns."""
    registry_entry = MODEL_REGISTRY[model_tag]
    if model_id is None:
        model_id = registry_entry["default_id"]
    print(f"Loading {model_id}...")
    model, processor = registry_entry["loader"](model_id)
    return model, processor, model_id


# ============================================================
# Main app
# ============================================================

def main():
    st.set_page_config(
        page_title="VLM Steering Demo",
        page_icon="🧭",
        layout="wide",
    )

    st.title("🧭 VLM Steering with Probes")
    st.caption(
        "Upload an image, ask a question, and steer the model's internal representations "
        "using trained linear probes."
    )

    # ----------------------------------------------------------
    # Sidebar: model & probe configuration
    # ----------------------------------------------------------
    with st.sidebar:
        st.header("Model")

        model_tag = st.selectbox(
            "Model",
            options=list(MODEL_REGISTRY.keys()),
            index=0,
        )
        model_id = st.text_input(
            "Model ID (blank = default)",
            value="",
            help="Leave blank to use the default model for this tag",
        )
        model_id = model_id.strip() or None

        probe_dirs = PROBE_DIRS[model_tag]

        # Check which probes are available
        available_tasks = []
        for task, pdir in probe_dirs.items():
            manifest = Path(pdir) / "manifest.json"
            if manifest.exists():
                available_tasks.append(task)

        if not available_tasks:
            st.warning("No probe directories found. Train probes first.")

        st.divider()
        st.header("Generation")
        max_new_tokens = st.slider("Max new tokens", 10, 200, 50)

    # ----------------------------------------------------------
    # Load model
    # ----------------------------------------------------------
    model, processor, loaded_id = load_model(model_tag, model_id)
    st.sidebar.success(f"Model loaded: {loaded_id}")

    # ----------------------------------------------------------
    # Main area: image upload + prompt
    # ----------------------------------------------------------
    col_input, col_output = st.columns([1, 1])

    with col_input:
        st.subheader("Input")

        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["png", "jpg", "jpeg", "webp"],
        )

        # Also allow picking from synthetic dataset
        use_synthetic = st.checkbox("Or pick from synthetic dataset")
        synthetic_path = None
        if use_synthetic:
            synthetic_dir = st.text_input("Images directory", value="data/raw/spatial/images")
            if Path(synthetic_dir).exists():
                images = sorted(Path(synthetic_dir).glob("*.png"))[:50]
                if images:
                    selected = st.selectbox(
                        "Select image",
                        options=images,
                        format_func=lambda p: p.name,
                    )
                    synthetic_path = selected

        prompt = st.text_area(
            "Prompt",
            value="Describe what you see in this image.",
            height=80,
        )

        # Display image
        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
        elif synthetic_path is not None:
            image = Image.open(synthetic_path).convert("RGB")

        if image is not None:
            st.image(image, use_container_width=True)

    with col_output:
        st.subheader("Steering Controls")

        if not available_tasks:
            st.info("No probes available. Configure probe directories in the sidebar.")
            return

        # Task / probe selection
        task = st.selectbox("Probe type", options=available_tasks)
        classes = TASK_CLASSES[task]
        probes_dir = probe_dirs[task]

        # Load manifest for layer info
        manifest_path = Path(probes_dir) / "manifest.json"
        try:
            import json
            with open(manifest_path) as f:
                manifest = json.load(f)
            n_layers = manifest["n_layers"]
            best_layer = manifest["best_layer"]
        except Exception:
            n_layers = 32
            best_layer = 20

        target_class = st.selectbox("Target class", options=classes)

        # Get defaults for this model+task combo
        defs = get_defaults(model_tag, task)

        # Strategy
        strategy_options = ["push", "contrast"]
        strategy = st.radio("Strategy", strategy_options, index=strategy_options.index(defs["strategy"]), horizontal=True)
        source_class = None
        if strategy == "contrast":
            source_options = [c for c in classes if c != target_class]
            source_class = st.selectbox("Source class (steer away from)", options=source_options)

        # Layer selection
        def_l_start, def_l_end = defs["layers"]
        def_l_start = max(1, min(def_l_start, n_layers - 1))
        def_l_end = max(1, min(def_l_end, n_layers - 1))

        layer_mode = st.radio("Layers", ["Best layer", "Single layer", "Layer range"], index=2, horizontal=True)
        if layer_mode == "Best layer":
            layers = [best_layer]
            st.caption(f"Using best layer: {best_layer}")
        elif layer_mode == "Single layer":
            layer = st.slider("Layer", 1, n_layers - 1, best_layer)
            layers = [layer]
        else:
            l_start, l_end = st.slider("Layer range", 1, n_layers - 1,
                                       (def_l_start, def_l_end))
            layers = list(range(l_start, l_end + 1))
            st.caption(f"Steering on layers: {layers}")

        # Alpha
        alpha = st.slider("α (steering strength)", 0.0, 100.0, defs["alpha"], step=0.5)

        # When
        when = st.radio("Intervention timing", ["all", "prefill"], horizontal=True,
                        help="'all': steer every forward pass. 'prefill': only during prompt processing.")

        st.divider()

        # ----------------------------------------------------------
        # Generate button
        # ----------------------------------------------------------
        if image is None:
            st.info("Upload an image to get started.")
            return

        if st.button("🚀 Generate", type="primary", use_container_width=True):
            with st.spinner("Generating baseline..."):
                baseline = _generate(model, processor, model_tag, image, prompt, max_new_tokens)

            with st.spinner(f"Generating with steering (α={alpha})..."):
                if alpha == 0:
                    steered = baseline
                else:
                    mgr = SteeringManager.from_probes(
                        model, model_tag, probes_dir, layers,
                        target_class=target_class,
                        source_class=source_class,
                        alpha=alpha,
                        strategy=strategy,
                        when=when,
                    )
                    with mgr:
                        steered = _generate(model, processor, model_tag, image, prompt, max_new_tokens)

            # Display results
            st.subheader("Results")

            col_b, col_s = st.columns(2)
            with col_b:
                st.markdown("**Baseline** (no steering)")
                st.info(baseline)
            with col_s:
                st.markdown(f"**Steered** → `{target_class}` (α={alpha})")
                st.success(steered)

            # Show diff
            if baseline != steered:
                st.caption("✅ Steering changed the response.")
            else:
                st.caption("⚠️ Response unchanged. Try increasing α or using a different layer.")

            # Store in session for history
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append({
                "task": task,
                "target": target_class,
                "source": source_class,
                "alpha": alpha,
                "layers": layers,
                "when": when,
                "strategy": strategy,
                "baseline": baseline,
                "steered": steered,
            })

    # ----------------------------------------------------------
    # History panel
    # ----------------------------------------------------------
    if "history" in st.session_state and st.session_state.history:
        st.divider()
        st.subheader("History")
        for i, entry in enumerate(reversed(st.session_state.history[-10:])):
            with st.expander(
                f"#{len(st.session_state.history) - i}: "
                f"{entry['task']} → {entry['target']} (α={entry['alpha']})",
                expanded=(i == 0),
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Baseline:**")
                    st.write(entry["baseline"])
                with col2:
                    st.markdown("**Steered:**")
                    st.write(entry["steered"])
                st.caption(
                    f"Layers: {entry['layers']} | Strategy: {entry['strategy']} | "
                    f"When: {entry['when']}"
                )


if __name__ == "__main__":
    main()