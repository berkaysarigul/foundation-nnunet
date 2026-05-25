# Controlled Short Hybrid Training Report

## Repository findings

- Existing hybrid path reused: `src/models/hybrid.py::HybridFoundationUNet`.
- Existing backbone path reused: `src/models/backbone.py::FoundationXBackbone`.
- Existing PR-6 diagnostics were extended for longer bounded training and scheduled validation.
- The full single-split runner was intentionally not reused because it writes to `artifacts/runs/` and exceeds PR-7 scope.

## Setup

```
py scripts/controlled_train_hybrid_short.py --input_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\nnUNet_raw\Dataset101_Pneumothorax\imagesTr --labels_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\nnUNet_raw\Dataset101_Pneumothorax\labelsTr --foundation_checkpoint C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\checkpoints\foundation_x.pth --output_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train --img_size 512 --device auto --num_train_cases 32 --num_val_cases 16 --max_steps 25 --batch_size 1 --lr 0.0001 --val_every 5 --strict
```

- Device: `cpu`
- Train cases: 32 (pos=16, neg=16)
- Val cases: 16 (pos=8, neg=8)
- Max steps: 25
- Validation interval: 5
- Batch size: 1
- Learning rate: 0.0001
- Frozen backbone: True

## Dataset selection

- Train/val disjoint overlap count: 0
- Train case IDs count: 32
- Val case IDs count: 16

## Training results

- Steps completed: 25 / 25
- Train total loss series: [1.7671746015548706, 1.7332868576049805, 1.6869170665740967, 1.7560518980026245, 1.7341399192810059, 1.7579238414764404, 1.7407013177871704, 1.7437639236450195, 1.7430323362350464, 1.7240006923675537, 1.6252731084823608, 1.5753823518753052, 1.2595055103302002, 1.1215699911117554, 1.380516529083252, 1.3198399543762207, 1.0147284269332886, 1.0253918170928955, 1.0204685926437378, 1.0432082414627075, 1.1390880346298218, 1.0213254690170288, 1.0070757865905762, 1.004738688468933, 1.0023528337478638]
- Train predicted positive pixel ratio series: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9942817687988281, 0.9782028198242188, 0.0481109619140625, 0.00815582275390625, 0.00035858154296875, 2.6702880859375e-05, 0.0, 0.0, 7.62939453125e-06, 3.4332275390625e-05, 2.288818359375e-05, 4.57763671875e-05, 4.57763671875e-05, 1.9073486328125e-05, 0.0, 3.814697265625e-06, 0.0]
- Train non-empty prediction rate series: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0]
- Probability output shape sample: `[1, 1, 512, 512]`
- Raw logits shape sample: `[1, 1, 512, 512]`
- NaN/Inf totals: prob_nan=0, prob_inf=0, logits_nan=0, logits_inf=0

## Validation results

- Validation schedule steps: [0, 5, 10, 15, 20, 25]
- Validation executed steps: [0, 5, 10, 15, 20, 25]
- Validation mean total loss per step: [1.7704522907733917, 1.7512867525219917, 1.6904266104102135, 1.1168527975678444, 1.0632254853844643, 1.0599572621285915]
- Validation Dice mean per step: [0.009647867700550705, 0.009647867700550705, 0.0, 0.3125, 0.0, 0.5]
- Validation non-empty rate per step: [1.0, 1.0, 1.0, 0.3125, 1.0, 0.0]

## Gradient and frozen-backbone checks

- Frozen backbone no-grad check: True
- Trainable non-backbone gradient present check: True
- Trainable non-backbone finite gradient check: True
- Optimizer excludes frozen backbone: True

## Degeneracy checks

- Any degenerate validation point: True
- Persistent collapse from start: False
- All-background validation points: 1
- All-foreground validation points: 2

## Artifact outputs

- Summary JSON: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train\hybrid_controlled_short_train_summary.json`
- Summary YAML: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train\hybrid_controlled_short_train_summary.yaml`
- Report: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train\hybrid_controlled_short_train_report.md`
- Train steps CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train\train_steps.csv`
- Validation steps CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train\validation_steps.csv`
- Gradient stats CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train\gradient_stats.csv`
- Selection log CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train\selection_log.csv`
- Progress plot: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train\progress.png`
- Visual file count: 24

## Failure cases

None.

## Decision

- Status: `BLOCKED`

## Next recommended PR

- `Blocking fix PR`