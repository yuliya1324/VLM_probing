"""Steer VLM outputs by intervening on hidden states using probe directions.

The core idea: if a linear probe θ can decode spatial information from layer
activations, then adding α·θ to the activations at that layer should push
the model toward a target spatial relation.

Supports:
  - Single or multiple layers
  - Intervention timing: "prefill" (prompt only) vs "all" (prompt + generation)
  - Two strategies: "push" and "contrast"

Usage:
    from src.steering.steer import steer_and_generate

    # Single layer
    result = steer_and_generate(
        model, processor, "qwen2", image, prompt,
        probes_dir="results/qwen2_spatial/probes",
        layers=[20],
        target_class="left_of",
        alpha=5.0,
    )

    # Multiple layers
    result = steer_and_generate(
        model, processor, "qwen2", image, prompt,
        probes_dir="results/qwen2_spatial/probes",
        layers=[15, 18, 20, 22, 25],
        target_class="left_of",
        alpha=3.0,
    )

    # Only steer during prompt processing, not during generation
    result = steer_and_generate(
        model, processor, "qwen2", image, prompt,
        probes_dir="results/qwen2_spatial/probes",
        layers=[20],
        target_class="left_of",
        alpha=10.0,
        when="prefill",
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List, Union

import joblib
import numpy as np
import torch
from PIL import Image


# ============================================================
# Steering hook
# ============================================================

class SteeringHook:
    """Hook that adds a steering vector to hidden states at a transformer layer.

    Supports:
      - when: "all" (every forward call) or "prefill" (only the first forward,
        i.e. the prompt processing pass; disabled during autoregressive generation)
    """

    def __init__(
        self,
        steering_vector: torch.Tensor,
        alpha: float = 5.0,
        when: str = "all",
    ):
        self.steering_vector = steering_vector
        self.alpha = alpha
        self.when = when
        self.handle = None
        self._call_count = 0
        self._active = True

    def hook_fn(self, module, input, output):
        if not self._active:
            return output

        self._call_count += 1

        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        # "prefill" mode: only intervene when seq_len > 1 (prompt pass).
        # During autoregressive generation, seq_len == 1 (single new token).
        if self.when == "prefill" and hidden_states.shape[1] == 1:
            if rest is not None:
                return (hidden_states,) + rest
            return hidden_states

        device = hidden_states.device
        dtype = hidden_states.dtype
        sv = self.steering_vector.to(device=device, dtype=dtype)

        hidden_states = hidden_states + self.alpha * sv

        if rest is not None:
            return (hidden_states,) + rest
        return hidden_states

    def attach(self, layer_module):
        self._call_count = 0
        self._active = True
        self.handle = layer_module.register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# ============================================================
# Steering vector extraction from probes
# ============================================================

def get_steering_vector(
    probes_dir: str,
    layer: int,
    target_class: str,
    source_class: Optional[str] = None,
    strategy: str = "push",
) -> torch.Tensor:
    """Extract a steering vector from a trained probe.

    Returns: torch.Tensor of shape (hidden_dim,)
    """
    probes_dir = Path(probes_dir)

    probe = joblib.load(probes_dir / f"probe_layer_{layer:03d}.joblib")
    le = joblib.load(probes_dir / "label_encoder.joblib")

    classes = list(le.classes_)
    target_idx = classes.index(target_class)

    if hasattr(probe, "estimators_"):
        target_weights = probe.estimators_[target_idx].coef_[0]
    else:
        target_weights = probe.coef_[target_idx]

    target_vec = torch.tensor(target_weights, dtype=torch.float32)

    if strategy == "contrast" and source_class is not None:
        source_idx = classes.index(source_class)
        if hasattr(probe, "estimators_"):
            source_weights = probe.estimators_[source_idx].coef_[0]
        else:
            source_weights = probe.coef_[source_idx]
        source_vec = torch.tensor(source_weights, dtype=torch.float32)
        return target_vec - source_vec

    return target_vec


# ============================================================
# Model layer access
# ============================================================

def get_model_layer(model, model_tag: str, layer_idx: int):
    """Get the nn.Module for a specific transformer layer."""
    if model_tag == "qwen2":
        return model.model.layers[layer_idx]
    elif model_tag == "llava15":
        return model.language_model.model.layers[layer_idx]
    elif model_tag == "vila":
        return model.llm.model.layers[layer_idx]
    else:
        raise ValueError(f"Unknown model_tag: {model_tag}. Add layer access pattern.")


def get_num_layers(model, model_tag: str) -> int:
    """Get the total number of transformer layers."""
    if model_tag == "qwen2":
        return len(model.model.layers)
    elif model_tag == "llava15":
        return len(model.language_model.model.layers)
    elif model_tag == "vila":
        return len(model.llm.model.layers)
    else:
        raise ValueError(f"Unknown model_tag: {model_tag}")


# ============================================================
# Multi-layer hook manager
# ============================================================

class SteeringManager:
    """Manage steering hooks across multiple layers.

    Can be used as a context manager:
        with SteeringManager.from_probes(...) as mgr:
            output = generate(...)
    """

    def __init__(self, hooks: List[tuple]):
        """hooks: List of (layer_module, SteeringHook) pairs."""
        self.hooks = hooks

    @classmethod
    def from_probes(
        cls,
        model,
        model_tag: str,
        probes_dir: str,
        layers: List[int],
        target_class: str,
        source_class: Optional[str] = None,
        alpha: float = 5.0,
        strategy: str = "push",
        when: str = "all",
    ) -> "SteeringManager":
        """Create a SteeringManager from trained probes.

        Args:
            layers: Probe layer indices. Layer 0 = embedding output,
                    layer 1 = first transformer block output, etc.
                    Each layer gets its own steering vector from its own probe.
            alpha: Scaling factor (shared across all layers)
            when: "all" — steer every forward pass (prompt + generation tokens)
                  "prefill" — steer only during prompt processing
        """
        hooks = []
        for layer in layers:
            sv = get_steering_vector(probes_dir, layer, target_class, source_class, strategy)
            hook = SteeringHook(sv, alpha=alpha, when=when)

            # Probe layer 0 = embeddings, layer 1 = first transformer block
            transformer_idx = layer - 1
            if transformer_idx < 0:
                print(f"  Skipping layer {layer} (embedding layer, cannot hook)")
                continue

            layer_module = get_model_layer(model, model_tag, transformer_idx)
            hooks.append((layer_module, hook))

        return cls(hooks)

    def attach(self):
        for layer_module, hook in self.hooks:
            hook.attach(layer_module)
        return self

    def remove(self):
        for _, hook in self.hooks:
            hook.remove()

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *args):
        self.remove()


# ============================================================
# Generation helpers
# ============================================================

def _generate(model, processor, model_tag: str, image: Image.Image,
              prompt: str, max_new_tokens: int) -> str:
    """Generate text from a VLM (handles model-specific dispatch)."""

    if model_tag == "vila":
        return _generate_vila(model, processor, image, prompt, max_new_tokens)

    from src.extraction.extract import INPUT_BUILDERS

    try:
        device = model.get_input_embeddings().weight.device
    except Exception:
        device = next(model.parameters()).device

    prepare_fn = INPUT_BUILDERS[model_tag]
    inputs = prepare_fn(processor, prompt, image, device)

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    if hasattr(processor, "batch_decode"):
        input_len = inputs.get("input_ids", inputs.get("inputs_embeds")).shape[1]
        text = processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0]
    else:
        text = processor.decode(output_ids[0], skip_special_tokens=True)

    return text.strip()


def _generate_vila(model, processor_tuple, image: Image.Image,
                   prompt: str, max_new_tokens: int) -> str:
    """Generate text from VILA model."""
    import inspect
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import DEFAULT_IMAGE_TOKEN

    device, tokenizer, image_processor = processor_tuple

    img_t = image_processor(image, return_tensors="pt")["pixel_values"][0].to(device).half()
    media = {"image": [img_t]}
    media_config = {"image": {}}

    chat_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

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

    output_ids = model.generate(
        input_ids=input_ids,
        media=media,
        media_config=media_config,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    input_len = input_ids.shape[1]
    text = tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True)
    return text.strip()


# ============================================================
# High-level API
# ============================================================

@torch.no_grad()
def steer_and_generate(
    model,
    processor,
    model_tag: str,
    image: Image.Image,
    prompt: str,
    probes_dir: str,
    layers: Union[int, List[int]],
    target_class: str,
    source_class: Optional[str] = None,
    alpha: float = 5.0,
    strategy: str = "push",
    when: str = "all",
    max_new_tokens: int = 50,
) -> dict:
    """Run generation with steering intervention.

    Args:
        layers: Single layer or list of layers to steer on.
        when: "all" — steer every forward pass (prompt + each generated token)
              "prefill" — steer only during prompt processing
        (other args same as before)
    """
    if isinstance(layers, int):
        layers = [layers]

    # Baseline
    baseline_output = _generate(model, processor, model_tag, image, prompt, max_new_tokens)

    # Steered
    with SteeringManager.from_probes(
        model, model_tag, probes_dir, layers,
        target_class=target_class, source_class=source_class,
        alpha=alpha, strategy=strategy,
        when=when,
    ):
        steered_output = _generate(model, processor, model_tag, image, prompt, max_new_tokens)

    return {
        "steered_output": steered_output,
        "baseline_output": baseline_output,
        "target_class": target_class,
        "source_class": source_class,
        "alpha": alpha,
        "layers": layers,
        "strategy": strategy,
        "when": when,
    }


@torch.no_grad()
def sweep_alpha(
    model, processor, model_tag: str,
    image: Image.Image, prompt: str,
    probes_dir: str,
    layers: Union[int, List[int]],
    target_class: str,
    source_class: Optional[str] = None,
    alphas: list = None,
    strategy: str = "push",
    when: str = "all",
    max_new_tokens: int = 50,
) -> list:
    """Run steering at multiple alpha values."""
    if isinstance(layers, int):
        layers = [layers]
    if alphas is None:
        alphas = [0, 1, 2, 5, 10, 20, 50]

    results = []
    for alpha in alphas:
        if alpha == 0:
            output = _generate(model, processor, model_tag, image, prompt, max_new_tokens)
        else:
            with SteeringManager.from_probes(
                model, model_tag, probes_dir, layers,
                target_class=target_class, source_class=source_class,
                alpha=alpha, strategy=strategy, when=when,
            ):
                output = _generate(model, processor, model_tag, image, prompt, max_new_tokens)

        results.append({"alpha": alpha, "output": output})
        print(f"  α={alpha:6.1f}  →  {output}")

    return results