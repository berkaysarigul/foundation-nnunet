# Tiny Hybrid Training Smoke Report

## Repository findings

- Existing model path used: `src/models/hybrid.py::HybridFoundationUNet`
- Existing backbone path used: `src/models/backbone.py::FoundationXBackbone`
- Existing losses reused: `src/training/losses.py::DiceFocalLoss` + BCEWithLogits
- Existing full/single-split runner was not reused because it writes under `artifacts/runs/` and exceeds PR-6 smoke scope.

## Setup

```
py scripts/tiny_train_hybrid_smoke.py --input_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\nnUNet_raw\Dataset101_Pneumothorax\imagesTr --labels_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\nnUNet_raw\Dataset101_Pneumothorax\labelsTr --foundation_checkpoint C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\checkpoints\foundation_x.pth --output_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_tiny_train_smoke --img_size 512 --device auto --num_train_cases 8 --num_val_cases 4 --max_steps 5 --batch_size 1 --lr 0.0001 --strict
```

- Device: `cpu`
- Number of training cases: 8 (pos=4, neg=4)
- Number of validation cases: 4 (pos=2, neg=2)
- Max steps: 5
- Batch size: 1
- Learning rate: 0.0001
- Frozen backbone setting: True

## Training smoke results

- Per-step loss values: [1.7968134880065918, 1.7207274436950684, 1.656131386756897, 1.7212010622024536, 1.7106261253356934]
- Probability output shape sample: `[1, 1, 512, 512]`
- Raw logits shape sample (hooked from model.final): `[1, 1, 512, 512]`
- Foundation X stage shape sample: `[[1, 128, 128, 128], [1, 256, 64, 64], [1, 512, 32, 32], [1, 1024, 16, 16]]`
- Gradient norm per step: [2.459545532505268, 2.2607878951799005, 2.0369280150717857, 1.866612550535916, 1.860563554187551]
- Optimizer trainable parameter count: 90
- NaN/Inf totals: prob_nan=0, prob_inf=0, logits_nan=0, logits_inf=0, grad_nonfinite=0

## Validation smoke results

- Validation case IDs: ['siim_000023', 'siim_000028', 'siim_000006', 'siim_000007']
- Validation loss mean (total): 1.7821418046951294
- Validation probability output shape sample: `[1, 1, 512, 512]`
- Validation raw logits shape sample: `[1, 1, 512, 512]`
- NaN/Inf status: prob_nan=0, prob_inf=0, logits_nan=0, logits_inf=0

## Artifact outputs

- Summary JSON: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_tiny_train_smoke\hybrid_tiny_train_smoke_summary.json`
- Summary YAML: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_tiny_train_smoke\hybrid_tiny_train_smoke_summary.yaml`
- Report Markdown: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_tiny_train_smoke\hybrid_tiny_train_smoke_report.md`
- Train-step CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_tiny_train_smoke\train_steps.csv`
- Gradient CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_tiny_train_smoke\gradient_stats.csv`
- Selection CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_tiny_train_smoke\selection_log.csv`
- Visual file count: 24

## Failure cases

None.

## Decision

- Status: `PASS`
- PASS: safe to plan a controlled short hybrid training run.
- PARTIAL_PASS: tiny training works but needs interpretation/fix before controlled training.
- BLOCKED: fix model/dataset/optimizer before proceeding.

## Next recommended PR

- `PR-7 Controlled Short Hybrid Training`

## Gradient routing

- Frozen backbone no-grad check: True
- Trainable non-backbone gradient present check: True
- Trainable non-backbone finite gradient check: True