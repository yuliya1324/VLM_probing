import os
os.environ["ACCELERATE_USE_META_DEVICE"] = "0"

import re
import inspect
from pathlib import Path
from typing import Union, Iterable, Dict

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, BitsAndBytesConfig

# transformers / InternVL2 compatibility patch
if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    PreTrainedModel.all_tied_weights_keys = set()

# =========================
# prompt (same as LLaVA1.5 spec)
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


def build_internvl2_query(obj1: str, obj2: str) -> str:
    instruction = LLaVA_SPATIAL_PROMPT.format(obj1=obj1, obj2=obj2)
    return f"<image>\n{instruction}"


def _safe(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "", s)
    return s[:80] if len(s) > 80 else s


def extract_internvl2_lasttoken_layers_from_ex(
    ex: dict,
    save_dir: Union[str, Path],
    model_id: str = "OpenGVLab/InternVL2-8B",
    selected_layers: Iterable[int] = (4, 12, 20, 28, 32),
    max_new_tokens: int = 32,
) -> Path:
    img_path = Path(ex["img_path"])
    obj1 = ex["subj"]
    obj2 = ex["obj"]

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 4bit quantization (needs: pip install bitsandbytes)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # ---- WORKAROUND: avoid recursion + avoid meta linspace during init ----
    _orig_linspace = None
    if getattr(torch.linspace, "__name__", "") != "_linspace_cpu":
        _orig_linspace = torch.linspace

        def _linspace_cpu(*args, **kwargs):
            kwargs.setdefault("device", "cpu")
            return _orig_linspace(*args, **kwargs)

        torch.linspace = _linspace_cpu

    try:
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map={"": "cuda:0"} if torch.cuda.is_available() else None,
            quantization_config=bnb if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True if torch.cuda.is_available() else False,
            torch_dtype="auto",
        )
    finally:
        if _orig_linspace is not None:
            torch.linspace = _orig_linspace

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False,
    )

    img = Image.open(img_path).convert("RGB")

    prompt_llava_style = build_prompt(obj1, obj2)
    query = build_internvl2_query(obj1, obj2)

    # ---- generation_config dict (InternVL2 chat expects dict-like) ----
    gen_cfg_obj = getattr(model, "generation_config", None)
    if gen_cfg_obj is None:
        gen_cfg = {}
    elif isinstance(gen_cfg_obj, dict):
        gen_cfg = dict(gen_cfg_obj)
    else:
        gen_cfg = gen_cfg_obj.to_dict()

    # avoid duplicates / weird length behavior
    gen_cfg.pop("use_cache", None)
    gen_cfg.pop("max_length", None)
    gen_cfg.pop("min_length", None)
    gen_cfg["max_new_tokens"] = int(max_new_tokens)
    gen_cfg["do_sample"] = False

    # ==========================================================
    # 핵: chat() は str しか返さないので hidden_states を “横取り”
    # - language_model.forward を一時的にラップして
    #   past_key_values=None の最初の forward だけ hidden_states を保存
    # ==========================================================
    if not hasattr(model, "language_model"):
        raise RuntimeError("This InternVL2 model has no .language_model; cannot hook hidden states.")

    captured: Dict[str, object] = {"hs": None}

    orig_forward = model.language_model.forward

    def forward_capture_hidden_states(*args, **kwargs):
        # force hidden states
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True

        out = orig_forward(*args, **kwargs)

        # capture ONLY the first step (prompt pass)
        # During generation, later steps pass past_key_values != None
        if captured["hs"] is None and kwargs.get("past_key_values", None) is None:
            hs = getattr(out, "hidden_states", None)
            if hs is not None:
                captured["hs"] = hs
        return out

    model.language_model.forward = forward_capture_hidden_states

    try:
        # call chat with correct positional order for this revision:
        # chat(tokenizer, pixel_values, question, generation_config, ...)
        with torch.no_grad():
            _ = model.chat(tokenizer, None, query, gen_cfg)
    finally:
        # restore no matter what
        model.language_model.forward = orig_forward

    hs = captured["hs"]
    if hs is None:
        raise RuntimeError(
            "Failed to capture hidden_states from the first forward pass. "
            "This InternVL2 revision may bypass language_model.forward in generate()."
        )

    # ---- extract selected layers last token ----
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
            "prompt": prompt_llava_style,
            "query_used": query,
            "selected_layers": list(map(int, selected_layers)),
            "model_id": model_id,
            "note": "hidden_states captured by wrapping model.language_model.forward at first generate step",
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

    save_path = extract_internvl2_lasttoken_layers_from_ex(
        ex,
        save_dir="/Data/masayo.tomita/VLM_probing/features/internvl2",
        model_id="OpenGVLab/InternVL2-8B",
        selected_layers=(4, 12, 20, 28, 32),
        max_new_tokens=8,  # 生成は短くてOK（hidden_states横取りが目的）
    )
    print("saved:", save_path)