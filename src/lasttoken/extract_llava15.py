from pathlib import Path
import re
import torch
from transformers import LlavaForConditionalGeneration, LlavaProcessor, BitsAndBytesConfig
from PIL import Image
from typing import Union

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

def extract_llava_lasttoken_layers_from_ex(
    ex: dict,
    save_dir: Union[str, Path],
    model_id: str = "llava-hf/llava-1.5-7b-hf",
    selected_layers=(4, 12, 20, 28, 32),
    prompt_builder=build_prompt
):
    """
    ex: {"img_path": Path/str, "subj": str, "obj": str, "rel": str (optional)}
    Saves: <image_stem>__<subj>__<obj>.pt
    """

    # --- load model + processor ---
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    processor = LlavaProcessor.from_pretrained(model_id)

    # --- fixed fields from ex ---
    img_path = Path(ex["img_path"])
    obj1 = ex["subj"]
    obj2 = ex["obj"]

    # --- prep IO ---
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(img_path).convert("RGB")
    prompt = prompt_builder(obj1, obj2)
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    # --- move all tensors to the same device as input embeddings ---
    device = model.get_input_embeddings().weight.device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    # --- forward + extract ---
    with torch.no_grad():
        out = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
            return_dict=True
        )

    hs = out.hidden_states  # tuple: (embeddings + each layer)
    max_layer = len(hs) - 1

    layers = {}
    for l in selected_layers:
        l = int(l)
        if not (0 <= l <= max_layer):
            raise ValueError(f"selected_layers contains {l}, but hidden_states has layers 0..{max_layer}")
        layers[l] = hs[l][0, -1].detach().cpu()  # [H]

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
        }
    }
    torch.save(payload, save_path)
    return save_path

if __name__ == "__main__":
    """
    ex = {
        "img_path": "/Data/masayo.tomita/VLM_probing/data/raw/vrd/sg_train_images/5516535415_0bae63e88d_b.jpg",
        "subj": "truck",
        "obj": "bus",
        "rel": "right of",
    }
    """
    ex = {
        "img_path": '/Data/masayo.tomita/VLM_probing/data/raw/vrd/sg_train_images/9561419764_c12dc30e76_b.jpg',
        "subj": "the wall",
        "obj": "people",
        "rel": "in front of",
    }
    
    save_path = extract_llava_lasttoken_layers_from_ex(
        ex,
        save_dir="/Data/masayo.tomita/VLM_probing/features/spacellava",
        model_id="salma-remyx/spacellava-1.5-7b"
    )
    print("saved:", save_path)