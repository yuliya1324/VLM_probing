from pathlib import Path
import re
import sys
import inspect
from typing import Union

import torch
from PIL import Image

# =========================
# import path (move script)
# =========================
REPO_ROOT = Path(__file__).resolve().parents[2]  # extract_lsttoken -> src -> VLM_probing
VILA_ROOT = REPO_ROOT / "VILA"
sys.path.insert(0, str(VILA_ROOT))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import DEFAULT_IMAGE_TOKEN


# =========================
# prompt (same as HF version)
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


def extract_specialrgbt_lasttoken_layers_from_ex(
    ex: dict,
    save_dir: Union[str, Path],
    model_id: str = "a8cheng/SpatialRGPT-VILA1.5-8B",
    prompt_builder=build_prompt,
) -> Path:
    """
    ex: {"img_path": Path/str, "subj": str, "obj": str, "rel": str}
    Saves: <image_stem>__<subj>__<obj>.pt
    """

    # --- fixed fields from ex ---
    img_path = Path(ex["img_path"])
    obj1 = ex["subj"]
    obj2 = ex["obj"]

    # --- prep IO ---
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- device + load model (same behavior as your script) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_id,
        model_name=model_id,
        model_base=None,
        device=device,
        device_map="auto",
    )
    model.eval()

    # --- image (same as your script) ---
    img = Image.open(img_path).convert("RGB")
    img_t = image_processor(img, return_tensors="pt")["pixel_values"][0].to(device)
    media = {"image": [img_t]}
    media_config = {"image": {}}

    # --- prompt (aligned to HF version) ---
    prompt = prompt_builder(obj1, obj2)

    # --- robust tokenizer_image_token (your original logic 그대로) ---
    image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    assert isinstance(image_token_id, int) and image_token_id >= 0

    sig = inspect.signature(tokenizer_image_token)
    param_names = list(sig.parameters.keys())

    args = [prompt, tokenizer]
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

    # --- forward (same as your script; no extra logic) ---
    with torch.no_grad():
        inputs_embeds, _, attention_mask = model._embed(
            input_ids=input_ids,
            media=media,
            media_config=media_config,
            labels=None,
            attention_mask=attention_mask,
        )

        out = model.llm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    # --- extract (aligned payload structure) ---
    hs = out.hidden_states
    layers = {l: hs[l][0, -1].detach().cpu() for l in range(len(hs))}

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

    save_path = extract_specialrgbt_lasttoken_layers_from_ex(
        ex,
        save_dir="/Data/masayo.tomita/VLM_probing/features/SpecialRGBT",
        model_id="a8cheng/SpatialRGPT-VILA1.5-8B"
    )
    print("saved:", save_path)