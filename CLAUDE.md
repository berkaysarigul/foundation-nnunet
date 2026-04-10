# Foundation-nnU-Net

Hybrid deep learning model for pneumothorax segmentation in chest X-rays.
Injects Foundation X (Swin-B backbone) multi-scale features into a U-Net decoder to improve segmentation performance.

## Tech Stack

Python 3.10+, PyTorch, timm, pydicom, albumentations, numpy, pandas, matplotlib, scipy, PIL, scikit-learn, pyyaml, tqdm, opencv-python-headless.

## Architecture

```
src/
├── data/           ← Data processing and loading
│   ├── preprocess.py    (DICOM→PNG, RLE→mask, train/val/test split)
│   ├── dataset.py       (PyTorch Dataset class)
│   └── augmentations.py (albumentations pipeline)
├── models/         ← Model definitions
│   ├── backbone.py      (Foundation X Swin-B feature extractor, frozen)
│   ├── unet.py          (Baseline U-Net)
│   └── hybrid.py        (Hybrid Foundation-nnU-Net = U-Net + Foundation X features)
├── training/       ← Training
│   ├── trainer.py       (Main training loop, CLI entry point)
│   ├── losses.py        (DiceLoss + BCELoss)
│   └── metrics.py       (Dice, IoU, Hausdorff, Precision, Recall, F1)
└── evaluation/     ← Evaluation
    ├── evaluate.py      (Test set final metrics)
    └── visualize.py     (Training curves, prediction visuals, comparison)
```

## Commands

```bash
# Data preprocessing
python -m src.data.preprocess --raw_dir data/raw/SIIM-ACR --output_dir data/processed/pneumothorax --img_size 512

# Baseline training
python -m src.training.trainer --config configs/config.yaml  # model.type: "baseline" in config

# Hybrid model training
python -m src.training.trainer --config configs/config.yaml  # model.type: "hybrid" in config

# Test evaluation
python -m src.evaluation.evaluate --config configs/config.yaml --checkpoint checkpoints/best_hybrid.pth

# Visualization
python -m src.evaluation.visualize --results_dir results/
```

## Development Guide

For detailed development guide, pseudo-code, hyperparameters, and all technical decisions:
**Read `docs/foundation_nnunet_dev_guide.md`**. Review the relevant section before coding each phase.
@.claude/skills/pytorch-patterns/SKILL.md use this skill when you developing with pytorch

## Coding Rules

- Each file has a single responsibility — do not mix concerns.
- Use type hints: `def forward(self, x: torch.Tensor) -> torch.Tensor`.
- All model forward outputs must be post-sigmoid (0-1 range).
- Never hardcode config values, always read from config.yaml.
- Seed consistency: numpy, torch, random all use `seed=42`.
- Prefer tqdm and structured logging over raw `print`.

## Critical Gotchas

- **Mask resize must use NEAREST interpolation.** BILINEAR blurs masks and breaks binary values. Fatal error for this project.
- **Foundation X checkpoint key structure:** First thing to do is `print(list(checkpoint.keys())[:30])` to find the key prefix. Could be "backbone.", "model.", "student." etc.
- **Foundation X backbone must stay in eval() mode at all times.** Even when `model.train()` is called, re-set backbone with `self.foundation_x.backbone.eval()`. Prevents BatchNorm stats corruption.
- **Grayscale → RGB:** Swin-B expects 3 channels. Use `x.repeat(1, 3, 1, 1)` to duplicate.
- **Dice is undefined on empty masks.** Separate pneumothorax-negative samples during evaluation, or use smooth=1e-6.
- **SIIM-ACR RLE format uses Fortran order (column-major).** Use `mask.reshape((H, W), order='F')`.
- **Same ImageId can have multiple RLE rows** (multiple pneumothorax regions). Merge with `np.maximum` (OR).
- **Input size must be a multiple of 32** (Swin-B patch embedding requirement). 256 and 512 are both valid.
- **Foundation X forward must run inside `torch.no_grad()`.** Saves memory and speed.
- **Optimizer must only receive `requires_grad=True` parameters:** `filter(lambda p: p.requires_grad, model.parameters())`.

## Data

- **SIIM-ACR (stage_2):** 3205 DICOM images + stage_2_train.csv (RLE masks). Path: `data/raw/SIIM-ACR/`
- **Processed data:** `data/processed/pneumothorax/images/` and `masks/` (512x512 PNG). Split: `splits.json` (70/15/15, seed=42).
- **Foundation X checkpoint:** `checkpoints/foundation_x.pth` (3GB, Swin-B weights).

## Build Order

Code sequentially, test each step:
1. requirements.txt → pip install
2. Directory structure
3. config.yaml
4. preprocess.py → run it, verify data
5. augmentations.py
6. dataset.py → load a few samples, check shapes
7. losses.py, metrics.py
8. unet.py → test forward pass with random input
9. backbone.py → load checkpoint, check feature shapes
10. hybrid.py → test forward pass with random input
11. trainer.py
12. Train baseline → save results
13. Train hybrid → save results
14. evaluate.py → evaluate both models on test set
15. visualize.py → generate charts and comparisons

## Verification

Per-module validation:
- **preprocess.py:** Equal number of images and masks in processed folder? Are masks binary (only 0 and 255)?
- **dataset.py:** Does `__getitem__` output have shape `(1, H, W)`? Are values in 0-1 range?
- **unet.py:** Does `UNet()(torch.randn(2, 1, 256, 256))` output shape `(2, 1, 256, 256)`?
- **backbone.py:** Does it return 4 feature maps? Do shapes match expected channel counts?
- **hybrid.py:** Does `HybridFoundationUNet()(torch.randn(2, 1, 256, 256))` have correct shape?
- **trainer.py:** Does 1 epoch run without errors? Is loss decreasing?
