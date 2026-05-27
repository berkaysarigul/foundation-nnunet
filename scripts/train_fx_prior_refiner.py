"""Train a lightweight Foundation X prior-guided segmentation refiner."""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.fx_prior_refiner_dataset import (  # noqa: E402
    INPUT_MODES,
    FoundationXPriorRefinerDataset,
    input_channels_for_mode,
)
from src.models.refiner_unet import FoundationXPriorRefinerUNet  # noqa: E402
from src.training.losses import BCEDiceLoss  # noqa: E402
from src.training.metrics import (  # noqa: E402
    aggregate_per_case_binary_metrics,
    compute_per_case_binary_metrics,
)


BASELINE_POSITIVE_DICE = 0.5235
SUCCESS_POSITIVE_DICE = 0.5335
SUCCESS_ABLATION_DELTA = 0.005
THRESHOLD = 0.5


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the PR-10C Foundation X prior refiner.",
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--input-mode", choices=sorted(INPUT_MODES), required=True)
    parser.add_argument("--target-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--amp", type=parse_bool, default=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    return parser.parse_args(argv)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _validate_run_dir(out_dir: Path) -> Path:
    resolved = _resolve_path(out_dir).resolve()
    normalized = resolved.as_posix().lower()
    if "artifacts/refiner/runs" not in normalized:
        raise ValueError(
            f"--out must be under artifacts/refiner/runs/<run_id>; got {out_dir}"
        )
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty run directory: {resolved}"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Manifest contains no rows: {path}")
    if "gt_is_positive" not in fieldnames:
        raise ValueError("train manifest must contain gt_is_positive for stratified split.")
    return fieldnames, rows


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _is_positive(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _validate_train_rows_do_not_use_heldout(rows: list[dict[str, str]]) -> None:
    offenders: list[str] = []
    for row in rows:
        joined = " ".join(
            [
                str(row.get("image_path", "")),
                str(row.get("label_path", "")),
            ]
        ).replace("\\", "/").lower()
        if "imagests" in joined or "heldout_labelsts" in joined:
            offenders.append(str(row.get("case_id", "")))
    if offenders:
        raise ValueError(
            "Training manifest must not reference imagesTs or heldout_labelsTs. "
            f"Offending cases: {offenders[:10]}"
        )


def _stratified_split(
    rows: list[dict[str, str]],
    *,
    val_split: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0.0 < val_split < 1.0:
        raise ValueError("--val-split must be strictly between 0 and 1.")

    labels = [1 if _is_positive(row["gt_is_positive"]) else 0 for row in rows]
    label_counts = {label: labels.count(label) for label in sorted(set(labels))}
    if len(label_counts) < 2 or min(label_counts.values()) < 2:
        raise ValueError(
            "Stratified split requires at least two positive and two negative cases "
            f"in the train manifest; got counts {label_counts}."
        )

    indices = list(range(len(rows)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        random_state=seed,
        stratify=labels,
    )
    train_rows = [rows[idx] for idx in sorted(train_idx)]
    val_rows = [rows[idx] for idx in sorted(val_idx)]
    return train_rows, val_rows


def _make_loader(
    manifest_path: Path,
    *,
    input_mode: str,
    target_size: int,
    batch_size: int,
    num_workers: int,
    augment: bool,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = FoundationXPriorRefinerDataset(
        manifest_path,
        input_mode=input_mode,
        target_size=target_size,
        augment=augment,
        normalization="zero_one",
        repo_root=REPO_ROOT,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def _evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_cases = 0
    case_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for x, y, _metadata in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            logits = model(x)
            loss = criterion(logits, y)
            probs = torch.sigmoid(logits)
            rows = compute_per_case_binary_metrics(
                probs.detach().cpu(),
                y.detach().cpu(),
                threshold=THRESHOLD,
            )
            case_rows.extend(rows)
            total_loss += float(loss.item()) * int(x.shape[0])
            total_cases += int(x.shape[0])

    aggregate = aggregate_per_case_binary_metrics(case_rows)
    aggregate["loss"] = total_loss / max(total_cases, 1)
    return aggregate


def _save_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_report(config: dict[str, Any], best_epoch: int, best_score: float) -> str:
    best_text = "nan" if math.isnan(best_score) else f"{best_score:.6f}"
    return "\n".join(
        [
            "# PR-10C Foundation X Prior Refiner Training Report",
            "",
            f"- input mode: `{config['input_mode']}`",
            f"- target size: `{config['target_size']}`",
            f"- train manifest: `{config['train_manifest']}`",
            f"- test manifest recorded only: `{config.get('test_manifest') or ''}`",
            f"- best epoch by validation positive Dice: `{best_epoch}`",
            f"- best validation positive Dice: `{best_text}`",
            "",
            "## Baseline Targets",
            "",
            (
                "- Foundation X direct teacher/head5 official_siim_224 baseline "
                f"positive Dice ~= {BASELINE_POSITIVE_DICE:.4f}"
            ),
            f"- Success: image_prior heldout positive Dice >= {SUCCESS_POSITIVE_DICE:.4f}",
            (
                "- Success: image_prior beats image_only by at least "
                f"{SUCCESS_ABLATION_DELTA:.3f} absolute positive Dice"
            ),
            "",
            "## Split Policy",
            "",
            "- Validation was stratified from the train manifest only.",
            "- Heldout/test manifest was not used for training or validation.",
        ]
    )


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    _set_seed(args.seed)
    run_dir = _validate_run_dir(args.out)
    train_manifest = _resolve_path(args.train_manifest).resolve()
    test_manifest = _resolve_path(args.test_manifest).resolve() if args.test_manifest else None
    if not train_manifest.exists():
        raise FileNotFoundError(f"train manifest not found: {train_manifest}")
    if test_manifest is not None and not test_manifest.exists():
        raise FileNotFoundError(f"test manifest not found: {test_manifest}")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.target_size <= 0:
        raise ValueError("--target-size must be > 0.")

    fieldnames, rows = _read_manifest(train_manifest)
    _validate_train_rows_do_not_use_heldout(rows)
    train_rows, val_rows = _stratified_split(rows, val_split=args.val_split, seed=args.seed)
    split_train_path = run_dir / "split_train.csv"
    split_val_path = run_dir / "split_val.csv"
    _write_rows(split_train_path, fieldnames, train_rows)
    _write_rows(split_val_path, fieldnames, val_rows)

    device = _resolve_device(args.device)
    amp_enabled = bool(args.amp) and device.type == "cuda"
    config = {
        "schema_version": 1,
        "created_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "train_manifest": str(train_manifest),
        "test_manifest": str(test_manifest) if test_manifest is not None else None,
        "out": str(run_dir),
        "input_mode": args.input_mode,
        "target_size": int(args.target_size),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "val_split": float(args.val_split),
        "seed": int(args.seed),
        "device": str(device),
        "amp": bool(args.amp),
        "amp_enabled": amp_enabled,
        "num_workers": int(args.num_workers),
        "early_stopping_patience": int(args.early_stopping_patience),
        "model": {
            "name": "FoundationXPriorRefinerUNet",
            "in_channels": input_channels_for_mode(args.input_mode),
            "out_channels": 1,
            "base_channels": 32,
            "norm": "batch",
        },
        "loss": "BCEWithLogitsLoss + DiceLoss",
        "selection_metric": "val_positive_dice",
        "threshold": THRESHOLD,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
    }
    _save_yaml(run_dir / "config.yaml", config)

    train_loader = _make_loader(
        split_train_path,
        input_mode=args.input_mode,
        target_size=args.target_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
        shuffle=True,
        seed=args.seed,
    )
    val_loader = _make_loader(
        split_val_path,
        input_mode=args.input_mode,
        target_size=args.target_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
        shuffle=False,
        seed=args.seed,
    )

    model = FoundationXPriorRefinerUNet(
        in_channels=input_channels_for_mode(args.input_mode),
        out_channels=1,
        base_channels=32,
        norm="batch",
    ).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    train_log_rows: list[dict[str, Any]] = []
    val_metric_rows: list[dict[str, Any]] = []
    best_score = float("nan")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for x, y, _metadata in progress:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * int(x.shape[0])
            seen += int(x.shape[0])
            progress.set_postfix(loss=f"{float(loss.item()):.4f}")

        train_loss = running_loss / max(seen, 1)
        val_metrics = _evaluate_model(model, val_loader, criterion, device)
        val_positive_dice = float(val_metrics.get("positive_dice", float("nan")))

        train_log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        val_row = {
            "epoch": epoch,
            "val_loss": float(val_metrics["loss"]),
            "val_dice": float(val_metrics["mean_dice_all_cases"]),
            "val_positive_dice": val_positive_dice,
            "val_iou": float(val_metrics["mean_iou_all_cases"]),
            "val_precision": float(val_metrics["mean_precision_all_cases"]),
            "val_recall": float(val_metrics["mean_recall_all_cases"]),
            "val_specificity": float(val_metrics["mean_specificity_all_cases"]),
            "val_case_level_f1": float(val_metrics["case_level_f1"]),
            "val_negative_fpr": float(val_metrics["negative_case_false_positive_rate"]),
            "val_tp": int(val_metrics["tp"]),
            "val_fn": int(val_metrics["fn"]),
            "val_fp": int(val_metrics["fp"]),
            "val_tn": int(val_metrics["tn"]),
        }
        val_metric_rows.append(val_row)

        improved = False
        if best_epoch == 0:
            improved = True
        elif not math.isnan(val_positive_dice):
            improved = math.isnan(best_score) or val_positive_dice > best_score

        if improved:
            best_score = val_positive_dice
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "schema_version": 1,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "best_val_positive_dice": best_score,
                },
                run_dir / "best_model.pth",
            )
        else:
            stale_epochs += 1

        _write_dict_csv(run_dir / "train_log.csv", train_log_rows)
        _write_dict_csv(run_dir / "val_metrics.csv", val_metric_rows)

        if args.early_stopping_patience >= 0 and stale_epochs >= args.early_stopping_patience:
            break

    (run_dir / "report.md").write_text(
        _build_report(config, best_epoch=best_epoch, best_score=best_score),
        encoding="utf-8",
    )

    return {
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_val_positive_dice": best_score,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
