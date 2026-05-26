import torch
from collections import Counter
from pathlib import Path


CKPT_PATH = "checkpoints/foundation_x.pth"
OUT_PATH = "foundationx_checkpoint_inventory.txt"


def get_state_dict(obj):
    if not isinstance(obj, dict):
        return obj, "raw_object"

    candidate_keys = [
        "state_dict",
        "model",
        "teacher_model",
        "student_model",
        "module",
        "net",
    ]

    for key in candidate_keys:
        if key in obj and isinstance(obj[key], dict):
            return obj[key], key

    tensor_like = [k for k, v in obj.items() if hasattr(v, "shape")]
    if len(tensor_like) > 10:
        return obj, "root_state_dict"

    return obj, "unknown_dict"


ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
state, selected_key = get_state_dict(ckpt)

lines = []
lines.append("=" * 80)
lines.append("FOUNDATION X CHECKPOINT INVENTORY")
lines.append("=" * 80)
lines.append(f"Checkpoint path: {CKPT_PATH}")
lines.append(f"Top-level type: {type(ckpt)}")

if isinstance(ckpt, dict):
    lines.append(f"Top-level keys: {list(ckpt.keys())}")

lines.append(f"Selected state dict key: {selected_key}")
lines.append(f"State type: {type(state)}")

if not isinstance(state, dict):
    lines.append("ERROR: selected state is not a dict.")
else:
    keys = list(state.keys())
    lines.append(f"Total keys: {len(keys)}")

    prefix1 = Counter(k.split(".")[0] for k in keys)
    prefix2 = Counter(".".join(k.split(".")[:2]) for k in keys)
    prefix3 = Counter(".".join(k.split(".")[:3]) for k in keys)

    lines.append("\n--- Prefix level 1 ---")
    for k, v in prefix1.most_common(100):
        lines.append(f"{k}: {v}")

    lines.append("\n--- Prefix level 2 ---")
    for k, v in prefix2.most_common(150):
        lines.append(f"{k}: {v}")

    lines.append("\n--- Prefix level 3 ---")
    for k, v in prefix3.most_common(200):
        lines.append(f"{k}: {v}")

    search_terms = [
        "seg",
        "segment",
        "segmentation",
        "decoder",
        "decode",
        "head",
        "heads",
        "s1",
        "s2",
        "s3",
        "uper",
        "psp",
        "fpn",
        "cls",
        "class",
        "classification",
        "loc",
        "localization",
        "bbox",
        "dino",
        "backbone",
        "teacher",
        "student",
    ]

    lines.append("\n--- Keyword matches ---")
    for term in search_terms:
        matches = [k for k in keys if term.lower() in k.lower()]
        lines.append(f"\n[{term}] matches: {len(matches)}")
        for k in matches[:120]:
            shape = tuple(state[k].shape) if hasattr(state[k], "shape") else "no_shape"
            lines.append(f"  {k} | shape={shape}")

    lines.append("\n--- First 300 keys ---")
    for k in keys[:300]:
        shape = tuple(state[k].shape) if hasattr(state[k], "shape") else "no_shape"
        lines.append(f"{k} | shape={shape}")

Path(OUT_PATH).write_text("\n".join(lines), encoding="utf-8")
print(f"Saved inventory to: {OUT_PATH}")
