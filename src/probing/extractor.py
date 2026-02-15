"""Extract hidden-state representations from VLMs.

For each (image, prompt) pair, we:
1. Feed the image + prompt into the VLM
2. Run a forward pass with output_hidden_states=True
3. Save the last-token hidden state from every layer

This gives us a tensor of shape (n_layers, hidden_dim) per sample,
which the probing classifier will use.
"""

import json
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from tqdm import tqdm


def extract_representations(
    model,
    processor,
    metadata_path: str,
    images_dir: str,
    output_path: str,
    device: str = "cuda",
    batch_size: int = 1,
    max_samples: Optional[int] = None,
):
    """Extract last-token hidden states from all layers.

    Args:
        model: A VLM (e.g., LlavaForConditionalGeneration) with output_hidden_states support.
        processor: Corresponding processor/tokenizer.
        metadata_path: Path to metadata.json from dataset generation.
        images_dir: Path to directory containing images.
        output_path: Where to save the .npz file with representations.
        device: torch device.
        batch_size: Currently only 1 is supported for simplicity.
        max_samples: Limit number of samples (for debugging).

    Saves:
        <output_path>.npz with keys:
            - "representations": np.array of shape (n_samples, n_layers, hidden_dim)
            - "labels": np.array of shape (n_samples,) — string labels
            - "image_ids": list of image_id strings
    """
    from PIL import Image

    with open(metadata_path) as f:
        metadata = json.load(f)

    if max_samples:
        metadata = metadata[:max_samples]

    all_reprs = []
    all_labels = []
    all_ids = []

    # Determine label key
    label_key = "relation" if "relation" in metadata[0] else "color_label"

    model.eval()
    with torch.no_grad():
        for sample in tqdm(metadata, desc="Extracting representations"):
            image_path = Path(images_dir) / sample["image_filename"]
            image = Image.open(image_path).convert("RGB")
            prompt = sample["prompt"]

            # Process inputs (model-specific — this is a generic template)
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs, output_hidden_states=True)

            # outputs.hidden_states is a tuple of (n_layers+1,) tensors
            # each of shape (batch, seq_len, hidden_dim)
            # We take the last token from each layer
            hidden_states = outputs.hidden_states  # tuple of tensors
            last_token_reprs = []
            for layer_hs in hidden_states:
                # layer_hs: (1, seq_len, hidden_dim)
                last_token_reprs.append(layer_hs[0, -1, :].cpu().numpy())

            # Stack: (n_layers, hidden_dim)
            sample_repr = np.stack(last_token_reprs, axis=0)
            all_reprs.append(sample_repr)
            all_labels.append(sample[label_key])
            all_ids.append(sample["image_id"])

    representations = np.stack(all_reprs, axis=0)  # (n_samples, n_layers, hidden_dim)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        representations=representations,
        labels=np.array(all_labels),
        image_ids=np.array(all_ids),
    )
    print(f"Saved representations: {representations.shape} → {output_path}")
    return representations