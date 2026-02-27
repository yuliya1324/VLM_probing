"""Extract last-token hidden states from VLMs for probing.

Core function borrowed from Masayo's extract_qwen2.py, generalized to:
  - Read directly from our metadata.json (no CSV conversion needed)
  - Support multiple models via a registry
  - Use task-appropriate prompt templates (spatial / color)

Usage:
    from src.extraction.extract import extract_dataset

    extract_dataset(
        metadata_path="data/raw/spatial/metadata.json",
        images_dir="data/raw/spatial/images",
        output_path="data/processed/qwen2_spatial.npz",
        model_id="Qwen/Qwen2-VL-7B-Instruct",
        task="spatial",
    )
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Callable, Dict, Any

import numpy as np
import torch
from PIL import Image


# ============================================================
# Prompt templates
# ============================================================

PROMPT_TEMPLATES = [
    "The spatial relationship of {subj} to {obj} is",
    "Considering the image, the {subj} is positioned ___ the {obj}. The answer is",
    "Where is the {subj} relative to the {obj}? The {subj} is",
]

SPATIAL_PROMPT = (
    "Determine the spatial relationship of '{subj}' relative to '{obj}'.\n"
    "Choose ONE label from:\n"
    "[left of, right of, above, below]\n"
    "Respond with ONLY the label. No explanation."
)

COLOR_PROMPT = (
    "What is the color of the {subj} in the image?\n"
    "Respond with ONLY the color name. No explanation."
)

PROMPT_TEMPLATES = {
    "spatial": SPATIAL_PROMPT,
    "color": COLOR_PROMPT,
}


def build_prompt(sample: dict, task: str) -> str:
    """Build a prompt string from a metadata sample."""
    template = PROMPT_TEMPLATES[task]

    if task == "spatial":
        subj = f"{sample['subject_color']} {sample['subject_shape']}"
        obj = f"{sample['reference_color']} {sample['reference_shape']}"
        return template.format(subj=subj, obj=obj)
    elif task == "color":
        subj = sample["shape_type"]
        return template.format(subj=subj)
    else:
        raise ValueError(f"Unknown task: {task}")


def get_label(sample: dict, task: str) -> str:
    """Extract the ground-truth label from a metadata sample."""
    if task == "spatial":
        return sample["relation"]
    elif task == "color":
        return sample["color_label"]
    else:
        raise ValueError(f"Unknown task: {task}")


# ============================================================
# Model loading registry
# ============================================================

def load_qwen2vl(model_id: str):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", torch_dtype="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def load_llava(model_id: str):
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", torch_dtype="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def load_vila(model_id: str):
    """Load VILA / SpatialRGPT model using VILA's custom builder.

    Requires VILA to be installed: pip install -e ./VILA --no-deps
    Returns (model, (device, tokenizer, image_processor)) — note the
    different return shape, handled by the VILA-specific extract path.
    """
    import os
    os.environ["FLASH_ATTENTION_2"] = "0"
    os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "sdpa"

    from llava.model.builder import load_pretrained_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_id,
        model_name=model_id,
        model_base=None,
        device=device,
        device_map="auto",
    )
    model.eval()
    # Pack extras into processor slot for registry compatibility
    return model, (device, tokenizer, image_processor)


# Maps a short tag → (loader_fn, input_builder_fn)
# input_builder_fn: (processor, prompt, image) → dict of model inputs
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "qwen2": {
        "loader": load_qwen2vl,
        "default_id": "Qwen/Qwen2-VL-7B-Instruct",
    },
    "llava15": {
        "loader": load_llava,
        "default_id": "llava-hf/llava-1.5-7b-hf",
    },
    "vila": {
        "loader": load_vila,
        "default_id": "a8cheng/SpatialRGPT-VILA1.5-8B",
    },
}


def prepare_inputs_qwen2(processor, prompt: str, image: Image.Image, device: torch.device) -> dict:
    """Prepare inputs for Qwen2-VL using its chat template."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}


def prepare_inputs_llava(processor, prompt: str, image: Image.Image, device: torch.device) -> dict:
    """Prepare inputs for LLaVA-1.5."""
    chat_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
    inputs = processor(text=chat_prompt, images=image, return_tensors="pt")
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}


INPUT_BUILDERS = {
    "qwen2": prepare_inputs_qwen2,
    "llava15": prepare_inputs_llava,
}


# ============================================================
# VILA-specific extraction
# ============================================================

@torch.no_grad()
def _extract_single_vila(
    model,
    processor_tuple,
    image: Image.Image,
    prompt: str,
) -> np.ndarray:
    """Extract last-token hidden states from VILA / SpatialRGPT.

    VILA uses a custom forward path:
      1. Tokenize with image tokens via tokenizer_image_token
      2. Embed via model._embed (fuses image + text)
      3. Forward through model.llm.model (the inner LLM)
    """
    import inspect
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import DEFAULT_IMAGE_TOKEN

    device, tokenizer, image_processor = processor_tuple

    # Process image
    img_t = image_processor(image, return_tensors="pt")["pixel_values"][0].to(device).half()
    media = {"image": [img_t]}
    media_config = {"image": {}}

    # Wrap prompt in LLaVA chat format
    chat_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

    # Tokenize with image token handling
    image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)

    sig = inspect.signature(tokenizer_image_token)
    param_names = list(sig.parameters.keys())

    args = [chat_prompt, tokenizer]
    kwargs = {}
    if "return_tensors" in sig.parameters:
        kwargs["return_tensors"] = "pt"

    for name in ("image_token_index", "image_token", "image_token_id"):
        if name in sig.parameters:
            pos = param_names.index(name)
            while len(args) < pos:
                args.append(None)
            args.append(image_token_id)
            break

    input_ids = tokenizer_image_token(*args, **kwargs)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)

    attention_mask = torch.ones_like(input_ids, device=device)

    # VILA custom embed: fuses image patches into the token sequence
    inputs_embeds, _, attention_mask = model._embed(
        input_ids=input_ids,
        media=media,
        media_config=media_config,
        labels=None,
        attention_mask=attention_mask,
    )

    # Ensure embeds match the LLM's expected dtype
    llm_dtype = next(model.llm.model.parameters()).dtype
    inputs_embeds = inputs_embeds.to(dtype=llm_dtype)

    # Forward through the inner LLM (not the outer wrapper)
    out = model.llm.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    hs = out.hidden_states
    last_token_per_layer = [hs[l][0, -1].detach().cpu().float().numpy() for l in range(len(hs))]
    return np.stack(last_token_per_layer, axis=0)


# ============================================================
# Core extraction (single sample) — dispatches by model_tag
# ============================================================

@torch.no_grad()
def extract_single(
    model,
    processor,
    model_tag: str,
    image: Image.Image,
    prompt: str,
) -> np.ndarray:
    """Extract last-token hidden state from all layers.

    Returns: np.ndarray of shape (n_layers, hidden_dim)
    """
    # VILA has a completely different forward path
    if model_tag == "vila":
        return _extract_single_vila(model, processor, image, prompt)

    # Standard HuggingFace models
    try:
        target_device = model.get_input_embeddings().weight.device
    except Exception:
        target_device = next(model.parameters()).device

    prepare_fn = INPUT_BUILDERS[model_tag]
    inputs = prepare_fn(processor, prompt, image, target_device)

    out = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )

    hs = out.hidden_states
    last_token_per_layer = [hs[l][0, -1].detach().cpu().float().numpy() for l in range(len(hs))]
    return np.stack(last_token_per_layer, axis=0)


# ============================================================
# Dataset-level extraction
# ============================================================

def extract_dataset(
    metadata_path: str,
    images_dir: str,
    output_path: str,
    model_tag: str = "qwen2",
    model_id: Optional[str] = None,
    task: str = "spatial",
    limit: Optional[int] = None,
    random_prompt: bool = False,
) -> np.ndarray:
    """Extract representations for an entire synthetic dataset.

    Args:
        metadata_path: Path to metadata.json from generate_dataset.py
        images_dir: Path to the images/ directory
        output_path: Where to save the .npz file
        model_tag: Key in MODEL_REGISTRY ("qwen2", "llava15", ...)
        model_id: HuggingFace model ID (defaults to registry default)
        task: "spatial" or "color" (determines prompt template & label key)
        limit: Max samples to process (None = all)

    Saves .npz with:
        representations: (n_samples, n_layers, hidden_dim)
        labels: (n_samples,)
        image_ids: (n_samples,)
    """
    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)

    if limit is not None:
        metadata = metadata[:limit]

    # Load model
    registry_entry = MODEL_REGISTRY[model_tag]
    if model_id is None:
        model_id = registry_entry["default_id"]

    print(f"Loading model: {model_id}")
    model, processor = registry_entry["loader"](model_id)

    # Extract
    all_reprs = []
    all_labels = []
    all_ids = []

    n = len(metadata)
    for i, sample in enumerate(metadata):
        image_path = Path(images_dir) / sample["image_filename"]
        image = Image.open(image_path).convert("RGB")
        # if random_prompt:
        prompt = build_prompt(sample, task)
        # else:
            # prompt = sample["prompt"] + " "
        label = get_label(sample, task)

        try:
            repr_array = extract_single(model, processor, model_tag, image, prompt)
            all_reprs.append(repr_array)
            all_labels.append(label)
            all_ids.append(sample["image_id"])

            if (i + 1) % 50 == 0 or (i + 1) == n:
                print(f"  [{i+1}/{n}] extracted")
        except torch.cuda.OutOfMemoryError:
            print(f"  [{i+1}/{n}] OOM — skipping")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [{i+1}/{n}] error: {e} — skipping")

    if not all_reprs:
        raise RuntimeError("No samples were successfully extracted!")

    representations = np.stack(all_reprs, axis=0)
    labels = np.array(all_labels)
    image_ids = np.array(all_ids)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, representations=representations, labels=labels, image_ids=image_ids)

    print(f"\nSaved: {representations.shape} → {output_path}")
    print(f"  {len(all_labels)} samples, {representations.shape[1]} layers, {representations.shape[2]} hidden dim")
    return representations