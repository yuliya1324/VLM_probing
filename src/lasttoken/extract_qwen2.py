from __future__ import annotations

from pathlib import Path
import re
from typing import Union, Iterable, Dict

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


# =========================
# prompt (same spec)
# =========================
LLaVA_SPATIAL_PROMPT = (
    "Determine the spatial relationship of '{obj1}' relative to '{obj2}'.\n"
    "Choose ONE label from:\n"
    "[left of, right of, above, below, in front of, behind]\n"
    "Respond with ONLY the label. No explanation."
)
LLaVA_CHAT_WRAPPER = "USER: <image>\n{instruction}\nASSISTANT:"


def build_prompt(obj1: str, obj2: str) -> str:
    instruction = LLaVA_SPATIAL_PROMPT.format(obj1=obj1, obj2=obj2)
    return LLaVA_CHAT_WRAPPER.format(instruction=instruction)


def _safe(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "", s)
    return s[:80] if len(s) > 80 else s


def extract_qwen2vl_lasttoken_layers_from_ex(
    ex: dict,
    save_dir: Union[str, Path],
    model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
    selected_layers: Iterable[int] = (4, 12, 20, 28, 32),
    prompt_builder=build_prompt,
) -> Path:
    """
    ex: {"img_path": Path/str, "subj": str, "obj": str, "rel": str (optional)}
    Saves: <image_stem>__<subj>__<obj>.pt

    Extracts: last-token hidden state at selected transformer layers.
    """

    # --- fixed fields from ex ---
    img_path = Path(ex["img_path"])
    obj1 = ex["subj"]
    obj2 = ex["obj"]

    # --- prep IO ---
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- load model + processor ---
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(model_id)

    # --- build messages (Qwen2-VL chat format) ---
    prompt = prompt_builder(obj1, obj2)
    # NOTE: Qwen2-VL expects messages, not raw "<image>" token in text;
    # we still keep 'prompt' in meta for spec consistency.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(img_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # --- make text via chat template, then tokenize with image ---
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,  # important: we will call processor(...) to attach image tensors
    )

    image = Image.open(img_path).convert("RGB")
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    )

    # --- move tensors to model's main device (safe with device_map="auto") ---
    # Prefer the device of input embeddings weight if available.
    try:
        target_device = model.get_input_embeddings().weight.device
    except Exception:
        target_device = next(model.parameters()).device

    inputs = {k: (v.to(target_device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    # --- forward + extract ---
    with torch.no_grad():
        out = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    hs = out.hidden_states  # tuple: (embeddings + each layer)
    max_layer = len(hs) - 1

    layers: Dict[int, torch.Tensor] = {}
    for l in selected_layers:
        l = int(l)
        if not (0 <= l <= max_layer):
            raise ValueError(f"selected_layers contains {l}, but hidden_states has layers 0..{max_layer}")
        layers[l] = hs[l][0, -1].detach().cpu()

    fname = f"{img_path.stem}__{_safe(obj1)}__{_safe(obj2)}.pt"
    save_path = save_dir / fname

    payload = {
        "layers": layers,
        "meta": {
            "img_path": str(img_path),
            "obj1": obj1,
            "obj2": obj2,
            "rel": ex.get("rel", None),
            "prompt": prompt,
            "selected_layers": list(map(int, selected_layers)),
            "model_id": model_id,
        },
    }
    torch.save(payload, save_path)
    return save_path


if __name__ == "__main__":
    ex = {
        "img_path": "/Data/masayo.tomita/VLM_probing/data/raw/vrd/sg_train_images/9561419764_c12dc30e76_b.jpg",
        "subj": "the wall",
        "obj": "people",
        "rel": "in front of",
    }

    save_path = extract_qwen2vl_lasttoken_layers_from_ex(
        ex,
        save_dir="/Data/masayo.tomita/VLM_probing/features/Qwen2-VL",
        model_id="Qwen/Qwen2-VL-7B-Instruct",
        selected_layers=(4, 12, 20, 28),
    )
    print("saved:", save_path)