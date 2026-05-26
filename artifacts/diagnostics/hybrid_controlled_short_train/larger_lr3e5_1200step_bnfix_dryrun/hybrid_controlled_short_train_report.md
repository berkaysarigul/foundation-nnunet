# Controlled Short Hybrid Training Report

## Repository findings

- Existing hybrid path reused: `src/models/hybrid.py::HybridFoundationUNet`.
- Existing backbone path reused: `src/models/backbone.py::FoundationXBackbone`.
- Existing PR-6 diagnostics were extended for longer bounded training and scheduled validation.
- The full single-split runner was intentionally not reused because it writes to `artifacts/runs/` and exceeds PR-7 scope.

## Setup

```
py scripts/controlled_train_hybrid_short.py --input_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\nnUNet_raw\Dataset101_Pneumothorax\imagesTr --labels_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\nnUNet_raw\Dataset101_Pneumothorax\labelsTr --foundation_checkpoint C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\checkpoints\foundation_x.pth --output_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train\larger_lr3e5_1200step_bnfix_dryrun --img_size 256 --device cpu --num_train_cases 16 --num_val_cases 8 --max_steps 5 --batch_size 1 --lr 3e-05 --bce_loss_weight 1.0 --val_every 2 --no_visuals --save_diagnostic_checkpoint --save_best_checkpoint
```

- Device: `cpu`
- Train cases: 16 (pos=8, neg=8)
- Val cases: 8 (pos=4, neg=4)
- Max steps: 5
- Validation interval: 2
- Batch size: 1
- Learning rate: 3e-05
- BCE loss weight: 1.0
- Frozen backbone: True

## Dataset selection

- Train/val disjoint overlap count: 0
- Train case IDs count: 16
- Val case IDs count: 8

## Training results

- Steps completed: 5 / 5
- Train total loss series: [1.8055492639541626, 1.8059756755828857, 1.7500793933868408, 1.795518159866333, 1.6914544105529785]
- Train predicted positive pixel ratio series: [0.7423095703125, 0.7564849853515625, 0.6896209716796875, 0.6826629638671875, 0.58331298828125]
- Train non-empty prediction rate series: [1.0, 1.0, 1.0, 1.0, 1.0]
- Probability output shape sample: `[1, 1, 256, 256]`
- Raw logits shape sample: `[1, 1, 256, 256]`
- NaN/Inf totals: prob_nan=0, prob_inf=0, logits_nan=0, logits_inf=0

## Validation results

- Validation schedule steps: [0, 2, 4, 5]
- Validation executed steps: [0, 2, 4, 5]
- Validation mean total loss per step: [1.76594977080822, 1.7669825106859207, 1.7721140682697296, 1.7750685065984726]
- Validation Dice mean per step: [0.01431974652223289, 0.01431974652223289, 0.01431974652223289, 0.01431974652223289]
- Validation probability min per step: [0.5201898217201233, 0.521461009979248, 0.5224317908287048, 0.5226927995681763]
- Validation probability max per step: [0.525608479976654, 0.5267863273620605, 0.5303830504417419, 0.5324694514274597]
- Validation probability mean per step: [0.5226819151636164, 0.5231276484569207, 0.5253253062174963, 0.5265841370096496]
- Validation probability std per step: [0.0004402831880879703, 0.0006465780393370841, 0.0011373298564689006, 0.0014134313969223278]
- Validation raw logit min per step: [0.08080326020717621, 0.08589682728052139, 0.08978736400604248, 0.09083361178636551]
- Validation raw logit max per step: [0.10252360999584198, 0.10724812000989914, 0.12168198078870773, 0.13006076216697693]
- Validation raw logit mean per step: [0.0907900437909035, 0.09257681147866492, 0.10138851689023909, 0.1064377775851284]
- Validation raw logit std per step: [0.0017647670866854114, 0.0025919217738416206, 0.004561220578461634, 0.005670067463295992]
- Validation probability foreground ratio @0.5 per step: [1.0, 1.0, 1.0, 1.0]
- Validation non-empty rate per step: [1.0, 1.0, 1.0, 1.0]

## Validation threshold sweep

### threshold = 0.05

- dice_mean per step: [0.01431974652223289, 0.01431974652223289, 0.01431974652223289, 0.01431974652223289]
- dice_pos_mean per step: [0.02863949304446578, 0.02863949304446578, 0.02863949304446578, 0.02863949304446578]
- prediction_non_empty_rate per step: [1.0, 1.0, 1.0, 1.0]
- pred_pos_pixel_ratio per step: [1.0, 1.0, 1.0, 1.0]
- all_background per step: [False, False, False, False]
- all_foreground per step: [True, True, True, True]

### threshold = 0.10

- dice_mean per step: [0.01431974652223289, 0.01431974652223289, 0.01431974652223289, 0.01431974652223289]
- dice_pos_mean per step: [0.02863949304446578, 0.02863949304446578, 0.02863949304446578, 0.02863949304446578]
- prediction_non_empty_rate per step: [1.0, 1.0, 1.0, 1.0]
- pred_pos_pixel_ratio per step: [1.0, 1.0, 1.0, 1.0]
- all_background per step: [False, False, False, False]
- all_foreground per step: [True, True, True, True]

### threshold = 0.20

- dice_mean per step: [0.01431974652223289, 0.01431974652223289, 0.01431974652223289, 0.01431974652223289]
- dice_pos_mean per step: [0.02863949304446578, 0.02863949304446578, 0.02863949304446578, 0.02863949304446578]
- prediction_non_empty_rate per step: [1.0, 1.0, 1.0, 1.0]
- pred_pos_pixel_ratio per step: [1.0, 1.0, 1.0, 1.0]
- all_background per step: [False, False, False, False]
- all_foreground per step: [True, True, True, True]

### threshold = 0.30

- dice_mean per step: [0.01431974652223289, 0.01431974652223289, 0.01431974652223289, 0.01431974652223289]
- dice_pos_mean per step: [0.02863949304446578, 0.02863949304446578, 0.02863949304446578, 0.02863949304446578]
- prediction_non_empty_rate per step: [1.0, 1.0, 1.0, 1.0]
- pred_pos_pixel_ratio per step: [1.0, 1.0, 1.0, 1.0]
- all_background per step: [False, False, False, False]
- all_foreground per step: [True, True, True, True]

### threshold = 0.40

- dice_mean per step: [0.01431974652223289, 0.01431974652223289, 0.01431974652223289, 0.01431974652223289]
- dice_pos_mean per step: [0.02863949304446578, 0.02863949304446578, 0.02863949304446578, 0.02863949304446578]
- prediction_non_empty_rate per step: [1.0, 1.0, 1.0, 1.0]
- pred_pos_pixel_ratio per step: [1.0, 1.0, 1.0, 1.0]
- all_background per step: [False, False, False, False]
- all_foreground per step: [True, True, True, True]

### threshold = 0.50

- dice_mean per step: [0.01431974652223289, 0.01431974652223289, 0.01431974652223289, 0.01431974652223289]
- dice_pos_mean per step: [0.02863949304446578, 0.02863949304446578, 0.02863949304446578, 0.02863949304446578]
- prediction_non_empty_rate per step: [1.0, 1.0, 1.0, 1.0]
- pred_pos_pixel_ratio per step: [1.0, 1.0, 1.0, 1.0]
- all_background per step: [False, False, False, False]
- all_foreground per step: [True, True, True, True]

## Train-mode preservation

- train_mode_restored_after_validation: True
- trainable_in_train_mode_before_step: True
- steps_checked / steps_passed: 5 / 5
- training_mode_violation_steps: []
- post_validation_checks: [{'after_val_step': 0, 'ok': True, 'model_training': True, 'message': ''}, {'after_val_step': 2, 'ok': True, 'model_training': True, 'message': ''}, {'after_val_step': 4, 'ok': True, 'model_training': True, 'message': ''}, {'after_val_step': 5, 'ok': True, 'model_training': True, 'message': ''}]

## Gradient and frozen-backbone checks

- Frozen backbone no-grad check: True
- Trainable non-backbone gradient present check: True
- Trainable non-backbone finite gradient check: True
- Optimizer excludes frozen backbone: True

## Degeneracy checks

- Any degenerate validation point: True
- Persistent collapse from start: True
- All-background validation points: 0
- All-foreground validation points: 4

## Artifact outputs

- Summary JSON: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train\larger_lr3e5_1200step_bnfix_dryrun\hybrid_controlled_short_train_summary.json`
- Summary YAML: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train\larger_lr3e5_1200step_bnfix_dryrun\hybrid_controlled_short_train_summary.yaml`
- Report: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train\larger_lr3e5_1200step_bnfix_dryrun\hybrid_controlled_short_train_report.md`
- Train steps CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train\larger_lr3e5_1200step_bnfix_dryrun\train_steps.csv`
- Validation steps CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train\larger_lr3e5_1200step_bnfix_dryrun\validation_steps.csv`
- Gradient stats CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train\larger_lr3e5_1200step_bnfix_dryrun\gradient_stats.csv`
- Selection log CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train\larger_lr3e5_1200step_bnfix_dryrun\selection_log.csv`
- Progress plot: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train\larger_lr3e5_1200step_bnfix_dryrun\progress.png`
- Visual file count: 0

## Failure cases

None.

## Warnings

None.

## Decision

- Status: `BLOCKED`

## Next recommended PR

- `Blocking fix PR`