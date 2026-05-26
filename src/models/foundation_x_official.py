"""
Official Foundation_X segmentation adapter for PR-9A-FIX Stage A/B.

This module intentionally wraps the vendored upstream Swin cyclic segmentation
implementation instead of the reverse-engineered `FoundationXFullNet`. It loads
only the `backbone.0.*` branch from a Foundation X checkpoint into the official
Swin module and exposes direct segmentation logits/probabilities for the
official pneumothorax-related heads.

It does not instantiate the full DINO detection stack, train, or export priors.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDORED_FOUNDATION_X_ROOT = REPO_ROOT / "third_party" / "foundation_x"
VENDORED_SWIN_PATH = (
    VENDORED_FOUNDATION_X_ROOT
    / "models"
    / "dino"
    / "swin_transformer_CyclicSegmentation.py"
)
OFFICIAL_FOUNDATION_X_COMMIT = "5e0b473cf3f595320ce982697d4365e1d9a4f642"

HEAD_SIIM_ACR_PNEUMOTHORAX = 5
HEAD_CANDID_PTX = 2
HEAD_CHESTX_DET = 4
CHESTX_DET_PNEUMOTHORAX_CHANNEL = 12

HEAD_DESCRIPTIONS: dict[int, str] = {
    HEAD_CANDID_PTX: "official CANDID-PTX pneumothorax diagnostic head",
    HEAD_CHESTX_DET: "official ChestX-Det multiclass segmentation head",
    HEAD_SIIM_ACR_PNEUMOTHORAX: "official SIIM-ACR pneumothorax segmentation head",
}


@dataclass(frozen=True)
class SegmentationHeadSpec:
    """A concrete segmentation output to evaluate."""

    head_idx: int
    channel: int | None = None

    @property
    def key(self) -> str:
        if self.channel is None:
            return f"head_{self.head_idx}"
        return f"head_{self.head_idx}_ch{self.channel}"

    @property
    def label(self) -> str:
        base = HEAD_DESCRIPTIONS.get(self.head_idx, f"segmentation head {self.head_idx}")
        if self.channel is None:
            return base
        return f"{base}, channel {self.channel}"


def default_head_specs(heads: list[int] | tuple[int, ...], head4_channel: int) -> list[SegmentationHeadSpec]:
    """Return requested binary heads plus the ChestX-Det channel diagnostic."""
    specs: list[SegmentationHeadSpec] = []
    seen: set[tuple[int, int | None]] = set()
    for head in heads:
        spec = SegmentationHeadSpec(int(head), None)
        if (spec.head_idx, spec.channel) not in seen:
            specs.append(spec)
            seen.add((spec.head_idx, spec.channel))
    head4 = SegmentationHeadSpec(HEAD_CHESTX_DET, int(head4_channel))
    if (head4.head_idx, head4.channel) not in seen:
        specs.append(head4)
    return specs


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    """Small JSON-serializable tensor stats block."""
    detached = tensor.detach()
    if detached.numel() == 0:
        return {
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "numel": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    cpu = detached.float().cpu()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "numel": int(detached.numel()),
        "min": float(cpu.min().item()),
        "max": float(cpu.max().item()),
        "mean": float(cpu.mean().item()),
        "std": float(cpu.std(unbiased=False).item()),
    }


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return tensor_stats(value)
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    return value


def _load_vendored_swin_module() -> Any:
    """Import the official vendored Swin cyclic segmentation module by path."""
    if not VENDORED_SWIN_PATH.exists():
        raise FileNotFoundError(
            f"Vendored Foundation_X Swin file not found: {VENDORED_SWIN_PATH}"
        )
    if not (VENDORED_FOUNDATION_X_ROOT / "VENDORED_COMMIT.txt").exists():
        raise FileNotFoundError(
            f"Vendored Foundation_X commit marker missing under {VENDORED_FOUNDATION_X_ROOT}"
        )

    module_name = "_foundation_x_official_swin_transformer_cyclic_segmentation"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(module_name, VENDORED_SWIN_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {VENDORED_SWIN_PATH}")

    inserted = False
    vendor_str = str(VENDORED_FOUNDATION_X_ROOT)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)
        inserted = True
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        if inserted:
            try:
                sys.path.remove(vendor_str)
            except ValueError:
                pass


def instantiate_official_swin_segmentation(
    model_name: str = "swin_B_224_22k",
    pretrain_img_size: int = 224,
    num_classes: int = 1,
    use_checkpoint: bool = False,
) -> torch.nn.Module:
    """
    Instantiate the official Foundation_X Swin segmentation branch.

    `num_classes=1` is required for binary segmentation heads 0/1/2/3/5 to
    match the released checkpoint. ChestX-Det head 4 is hard-coded upstream to
    13 output channels.
    """
    module = _load_vendored_swin_module()
    return module.build_swin_transformer(
        model_name,
        pretrain_img_size=pretrain_img_size,
        num_classes=num_classes,
        out_indices=(0, 1, 2, 3),
        dilation=False,
        use_checkpoint=use_checkpoint,
    )


def load_checkpoint_state_dict(checkpoint_path: Path, state_key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a Foundation X checkpoint and select `model` or `teacher_model`."""
    checkpoint_path = Path(checkpoint_path)
    obj = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    info: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "top_level_type": type(obj).__name__,
    }
    if isinstance(obj, dict):
        info["top_level_keys"] = list(obj.keys())
        if state_key not in obj:
            raise KeyError(
                f"state_key={state_key!r} not found in checkpoint keys {info['top_level_keys']}"
            )
        state = obj[state_key]
        info["selected_state_key"] = state_key
    else:
        state = obj
        info["top_level_keys"] = []
        info["selected_state_key"] = "raw"

    if not isinstance(state, dict):
        raise TypeError(
            f"Checkpoint entry {state_key!r} must be a state_dict-like dict; got {type(state).__name__}"
        )

    info["state_dict_len"] = len(state)
    info["state_dict_key_sample"] = list(state.keys())[:20]
    info["prefix_counts"] = _prefix_counts(state)
    return state, info


def _prefix_counts(state_dict: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in state_dict:
        prefix = str(key).split(".")[0]
        counts[prefix] = counts.get(prefix, 0) + 1
    return counts


def strip_backbone0_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Strip `backbone.0.` from official checkpoint keys."""
    prefix = "backbone.0."
    stripped: dict[str, torch.Tensor] = {}
    stats = {
        "raw_total": 0,
        "raw_with_backbone0_prefix": 0,
        "routed_swin": 0,
        "routed_segnorm": 0,
        "routed_locnorm": 0,
        "routed_ppn": 0,
        "routed_fpn": 0,
        "routed_segmentation_heads": 0,
        "routed_classification_heads": 0,
        "routed_swin_head": 0,
        "skipped_non_backbone0": 0,
        "routed_other_backbone0": 0,
    }
    for raw_key, value in state_dict.items():
        key = str(raw_key)
        stats["raw_total"] += 1
        if key.startswith("module."):
            key = key[len("module.") :]
        if not key.startswith(prefix):
            stats["skipped_non_backbone0"] += 1
            continue

        stats["raw_with_backbone0_prefix"] += 1
        inner = key[len(prefix) :]
        stripped[inner] = value
        if inner.startswith(("patch_embed", "layers.", "norm.")):
            stats["routed_swin"] += 1
        elif inner.startswith("Segnorm"):
            stats["routed_segnorm"] += 1
        elif inner.startswith("Locnorm"):
            stats["routed_locnorm"] += 1
        elif inner.startswith("segmentation_PPN"):
            stats["routed_ppn"] += 1
        elif inner.startswith("segmentation_FPN"):
            stats["routed_fpn"] += 1
        elif inner.startswith("segmentation_heads"):
            stats["routed_segmentation_heads"] += 1
        elif inner.startswith("classification_heads"):
            stats["routed_classification_heads"] += 1
        elif inner.startswith("head."):
            stats["routed_swin_head"] += 1
        else:
            stats["routed_other_backbone0"] += 1
    return stripped, stats


def filter_shape_compatible_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], list[str]]:
    """
    Keep only keys present in the official model with exactly matching shapes.

    PyTorch raises on size mismatches even with `strict=False`; diagnostics keep
    these separate from true missing/unexpected keys.
    """
    model_state = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    shape_mismatches: list[dict[str, Any]] = []
    unexpected_before_filter: list[str] = []
    for key, value in state_dict.items():
        if key not in model_state:
            unexpected_before_filter.append(key)
            continue
        expected_shape = tuple(model_state[key].shape)
        got_shape = tuple(value.shape) if hasattr(value, "shape") else None
        if got_shape != expected_shape:
            shape_mismatches.append(
                {
                    "key": key,
                    "checkpoint_shape": list(got_shape) if got_shape is not None else None,
                    "model_shape": list(expected_shape),
                }
            )
            continue
        compatible[key] = value
    return compatible, shape_mismatches, unexpected_before_filter


class OfficialFoundationXSegmentationModel:
    """
    Small wrapper around the official vendored Swin segmentation branch.

    The wrapper owns exactly one official Swin model instance and can load one
    checkpoint state key at a time.
    """

    def __init__(
        self,
        device: str | torch.device = "cpu",
        model_name: str = "swin_B_224_22k",
        pretrain_img_size: int = 224,
        num_classes: int = 1,
        use_checkpoint: bool = False,
    ):
        self.device = torch.device(device)
        self.model_name = model_name
        self.pretrain_img_size = int(pretrain_img_size)
        self.num_classes = int(num_classes)
        self.use_checkpoint = bool(use_checkpoint)
        self.instantiation_error: str | None = None
        self.load_diagnostics: dict[str, Any] | None = None
        try:
            self.model = instantiate_official_swin_segmentation(
                model_name=self.model_name,
                pretrain_img_size=self.pretrain_img_size,
                num_classes=self.num_classes,
                use_checkpoint=self.use_checkpoint,
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as exc:
            self.instantiation_error = f"{type(exc).__name__}: {exc}"
            self.model = None
            raise

    def load_checkpoint(self, checkpoint_path: Path, state_key: str) -> dict[str, Any]:
        """Load `checkpoint[state_key]` into the official Swin module."""
        if self.model is None:
            raise RuntimeError(f"official model was not instantiated: {self.instantiation_error}")

        state_dict, ckpt_info = load_checkpoint_state_dict(Path(checkpoint_path), state_key)
        stripped, route_stats = strip_backbone0_state_dict(state_dict)
        compatible, shape_mismatches, unexpected_before_filter = filter_shape_compatible_state_dict(
            self.model,
            stripped,
        )
        missing, unexpected = self.model.load_state_dict(compatible, strict=False)
        self.model.to(self.device)
        self.model.eval()

        diagnostics: dict[str, Any] = {
            "status": "loaded",
            "official_commit": OFFICIAL_FOUNDATION_X_COMMIT,
            "vendored_root": str(VENDORED_FOUNDATION_X_ROOT),
            "vendored_swin_path": str(VENDORED_SWIN_PATH),
            "model_name": self.model_name,
            "pretrain_img_size": self.pretrain_img_size,
            "num_classes": self.num_classes,
            "checkpoint_info": ckpt_info,
            "route_stats": route_stats,
            "stripped_backbone0_key_count": len(stripped),
            "shape_compatible_key_count": len(compatible),
            "missing": [str(k) for k in missing],
            "unexpected": [str(k) for k in unexpected],
            "unexpected_before_shape_filter": [str(k) for k in unexpected_before_filter],
            "shape_mismatches": shape_mismatches,
            "missing_count": len(missing),
            "unexpected_count": len(unexpected),
            "unexpected_before_shape_filter_count": len(unexpected_before_filter),
            "shape_mismatch_count": len(shape_mismatches),
        }
        self.load_diagnostics = diagnostics
        return diagnostics

    @torch.no_grad()
    def forward_logits(
        self,
        image_tensor: torch.Tensor,
        head_idx: int,
        channel: int | None = None,
    ) -> torch.Tensor:
        """Run official `extra_features_seg` and return raw logits."""
        if self.model is None:
            raise RuntimeError(f"official model was not instantiated: {self.instantiation_error}")
        x = image_tensor.to(self.device, non_blocking=False).float()
        logits, _, _ = self.model.extra_features_seg(x, head_n=int(head_idx))
        if channel is not None:
            if logits.ndim != 4:
                raise AssertionError(f"Expected BCHW logits, got {tuple(logits.shape)}")
            if channel < 0 or channel >= logits.shape[1]:
                raise IndexError(
                    f"Head {head_idx} produced {logits.shape[1]} channels; "
                    f"cannot select channel {channel}"
                )
            logits = logits[:, int(channel) : int(channel) + 1]
        return logits

    @torch.no_grad()
    def predict_probability(
        self,
        image_tensor: torch.Tensor,
        head_idx: int,
        channel: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(logits, sigmoid_probability)` for a head/channel."""
        logits = self.forward_logits(image_tensor, head_idx=head_idx, channel=channel)
        prob = torch.sigmoid(logits)
        return logits, prob

    @torch.no_grad()
    def run_synthetic_probe(
        self,
        image_size: int,
        heads: list[int] | tuple[int, ...] = (HEAD_SIIM_ACR_PNEUMOTHORAX, HEAD_CANDID_PTX),
        head4_channel: int = CHESTX_DET_PNEUMOTHORAX_CHANNEL,
    ) -> dict[str, Any]:
        """Run one zero tensor through the official segmentation branch."""
        image_size = int(image_size)
        probe = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32, device=self.device)
        results: dict[str, Any] = {
            "synthetic_input": tensor_stats(probe),
            "outputs": {},
        }
        for spec in default_head_specs(list(heads), int(head4_channel)):
            try:
                logits, prob = self.predict_probability(
                    probe,
                    head_idx=spec.head_idx,
                    channel=spec.channel,
                )
                raw_logits = self.forward_logits(probe, head_idx=spec.head_idx, channel=None)
                results["outputs"][spec.key] = {
                    "label": spec.label,
                    "raw_head_logits": tensor_stats(raw_logits),
                    "selected_logits": tensor_stats(logits),
                    "probability": tensor_stats(prob),
                }
            except Exception as exc:
                results["outputs"][spec.key] = {
                    "label": spec.label,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=5),
                }
        return results

    def save_load_diagnostics(self, path: Path) -> None:
        """Persist the latest load diagnostics as JSON."""
        if self.load_diagnostics is None:
            raise RuntimeError("No checkpoint has been loaded yet.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_sanitize_for_json(self.load_diagnostics), indent=2),
            encoding="utf-8",
        )


__all__ = [
    "CHESTX_DET_PNEUMOTHORAX_CHANNEL",
    "HEAD_CANDID_PTX",
    "HEAD_CHESTX_DET",
    "HEAD_DESCRIPTIONS",
    "HEAD_SIIM_ACR_PNEUMOTHORAX",
    "OFFICIAL_FOUNDATION_X_COMMIT",
    "OfficialFoundationXSegmentationModel",
    "SegmentationHeadSpec",
    "VENDORED_FOUNDATION_X_ROOT",
    "VENDORED_SWIN_PATH",
    "default_head_specs",
    "filter_shape_compatible_state_dict",
    "instantiate_official_swin_segmentation",
    "load_checkpoint_state_dict",
    "strip_backbone0_state_dict",
    "tensor_stats",
]
