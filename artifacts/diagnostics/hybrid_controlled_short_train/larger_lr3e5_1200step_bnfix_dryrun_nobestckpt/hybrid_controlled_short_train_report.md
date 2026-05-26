# Controlled Short Hybrid Training Report

## Repository findings

- Existing hybrid path reused: `src/models/hybrid.py::HybridFoundationUNet`.
- Existing backbone path reused: `src/models/backbone.py::FoundationXBackbone`.
- Existing PR-6 diagnostics were extended for longer bounded training and scheduled validation.
- The full single-split runner was intentionally not reused because it writes to `artifacts/runs/` and exceeds PR-7 scope.

## Setup

```
py scripts/controlled_train_hybrid_short.py --input_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\nnUNet_raw\Dataset101_Pneumothorax\imagesTr --labels_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\nnUNet_raw\Dataset101_Pneumothorax\labelsTr --foundation_checkpoint C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\checkpoints\foundation_x.pth --output_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train\larger_lr3e5_1200step_bnfix_dryrun_nobestckpt --img_size 256 --device cpu --num_train_cases 8 --num_val_cases 4 --max_steps 3 --batch_size 1 --lr 3e-05 --bce_loss_weight 1.0 --val_every 2 --no_visuals --save_diagnostic_checkpoint
```

- Device: `cpu`
- Train cases: 8 (pos=4, neg=4)
- Val cases: 4 (pos=2, neg=2)
- Max steps: 3
- Validation interval: 2
- Batch size: 1
- Learning rate: 3e-05
- BCE loss weight: 1.0
- Frozen backbone: True

## Dataset selection

- Train/val disjoint overlap count: 0
- Train case IDs count: 8
- Val case IDs count: 4

## Training results

- Steps completed: 3 / 3
- Train total loss series: [1.8055492639541626, 1.8059756755828857, 1.7500793933868408]
- Train predicted positive pixel ratio series: [0.7423095703125, 0.7564849853515625, 0.6896209716796875]
- Train non-empty prediction rate series: [1.0, 1.0, 1.0]
- Probability output shape sample: `[1, 1, 256, 256]`
- Raw logits shape sample: `[1, 1, 256, 256]`
- NaN/Inf totals: prob_nan=0, prob_inf=0, logits_nan=0, logits_inf=0

## Validation results

- Validation schedule steps: [0, 2, 3]
- Validation executed steps: [0, 2, 3]
- Validation mean total loss per step: [1.7720482349395752, 1.772978037595749, 1.77549347281456]
- Validation Dice mean per step: [0.008017974381800741, 0.008017974381800741, 0.008017974381800741]
- Validation probability min per step: [0.5207699537277222, 0.5210103392601013, 0.5216223001480103]
- Validation probability max per step: [0.525050163269043, 0.525810182094574, 0.5275192260742188]
- Validation probability mean per step: [0.5227140739125389, 0.5230961628060413, 0.5241650275575012]
- Validation probability std per step: [0.00047348399263450727, 0.0007135747937032188, 0.0009504225167806595]
- Validation raw logit min per step: [0.08312754333019257, 0.08409073948860168, 0.08654316514730453]
- Validation raw logit max per step: [0.10028461366891861, 0.10333240032196045, 0.110188327729702]
- Validation raw logit mean per step: [0.0909189551435361, 0.09245063283202626, 0.09673582812342829]
- Validation raw logit std per step: [0.0018978448679552633, 0.002860477339257025, 0.0038107364384700497]
- Validation probability foreground ratio @0.5 per step: [1.0, 1.0, 1.0]
- Validation non-empty rate per step: [1.0, 1.0, 1.0]

## Validation threshold sweep

### threshold = 0.05

- dice_mean per step: [0.008017974381800741, 0.008017974381800741, 0.008017974381800741]
- dice_pos_mean per step: [0.016035948763601482, 0.016035948763601482, 0.016035948763601482]
- prediction_non_empty_rate per step: [1.0, 1.0, 1.0]
- pred_pos_pixel_ratio per step: [1.0, 1.0, 1.0]
- all_background per step: [False, False, False]
- all_foreground per step: [True, True, True]

### threshold = 0.10

- dice_mean per step: [0.008017974381800741, 0.008017974381800741, 0.008017974381800741]
- dice_pos_mean per step: [0.016035948763601482, 0.016035948763601482, 0.016035948763601482]
- prediction_non_empty_rate per step: [1.0, 1.0, 1.0]
- pred_pos_pixel_ratio per step: [1.0, 1.0, 1.0]
- all_background per step: [False, False, False]
- all_foreground per step: [True, True, True]

### threshold = 0.20

- dice_mean per step: [0.008017974381800741, 0.008017974381800741, 0.008017974381800741]
- dice_pos_mean per step: [0.016035948763601482, 0.016035948763601482, 0.016035948763601482]
- prediction_non_empty_rate per step: [1.0, 1.0, 1.0]
- pred_pos_pixel_ratio per step: [1.0, 1.0, 1.0]
- all_background per step: [False, False, False]
- all_foreground per step: [True, True, True]

### threshold = 0.30

- dice_mean per step: [0.008017974381800741, 0.008017974381800741, 0.008017974381800741]
- dice_pos_mean per step: [0.016035948763601482, 0.016035948763601482, 0.016035948763601482]
- prediction_non_empty_rate per step: [1.0, 1.0, 1.0]
- pred_pos_pixel_ratio per step: [1.0, 1.0, 1.0]
- all_background per step: [False, False, False]
- all_foreground per step: [True, True, True]

### threshold = 0.40

- dice_mean per step: [0.008017974381800741, 0.008017974381800741, 0.008017974381800741]
- dice_pos_mean per step: [0.016035948763601482, 0.016035948763601482, 0.016035948763601482]
- prediction_non_empty_rate per step: [1.0, 1.0, 1.0]
- pred_pos_pixel_ratio per step: [1.0, 1.0, 1.0]
- all_background per step: [False, False, False]
- all_foreground per step: [True, True, True]

### threshold = 0.50

- dice_mean per step: [0.008017974381800741, 0.008017974381800741, 0.008017974381800741]
- dice_pos_mean per step: [0.016035948763601482, 0.016035948763601482, 0.016035948763601482]
- prediction_non_empty_rate per step: [1.0, 1.0, 1.0]
- pred_pos_pixel_ratio per step: [1.0, 1.0, 1.0]
- all_background per step: [False, False, False]
- all_foreground per step: [True, True, True]

## Train-mode preservation

- train_mode_restored_after_validation: True
- trainable_in_train_mode_before_step: True
- steps_checked / steps_passed: 3 / 3
- training_mode_violation_steps: []
- post_validation_checks: [{'after_val_step': 0, 'ok': True, 'model_training': True, 'message': ''}, {'after_val_step': 2, 'ok': True, 'model_training': True, 'message': ''}, {'after_val_step': 3, 'ok': True, 'model_training': True, 'message': ''}]

## Gradient and frozen-backbone checks

- Frozen backbone no-grad check: True
- Trainable non-backbone gradient present check: True
- Trainable non-backbone finite gradient check: True
- Optimizer excludes frozen backbone: True

## Degeneracy checks

- Any degenerate validation point: True
- Persistent collapse from start: True
- All-background validation points: 0
- All-foreground validation points: 3

## Artifact outputs

- Summary JSON: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train\larger_lr3e5_1200step_bnfix_dryrun_nobestckpt\hybrid_controlled_short_train_summary.json`
- Summary YAML: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train\larger_lr3e5_1200step_bnfix_dryrun_nobestckpt\hybrid_controlled_short_train_summary.yaml`
- Report: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train\larger_lr3e5_1200step_bnfix_dryrun_nobestckpt\hybrid_controlled_short_train_report.md`
- Train steps CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train\larger_lr3e5_1200step_bnfix_dryrun_nobestckpt\train_steps.csv`
- Validation steps CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train\larger_lr3e5_1200step_bnfix_dryrun_nobestckpt\validation_steps.csv`
- Gradient stats CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train\larger_lr3e5_1200step_bnfix_dryrun_nobestckpt\gradient_stats.csv`
- Selection log CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train\larger_lr3e5_1200step_bnfix_dryrun_nobestckpt\selection_log.csv`
- Progress plot: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train\larger_lr3e5_1200step_bnfix_dryrun_nobestckpt\progress.png`
- Visual file count: 0

## Failure cases

None.

## Warnings

None.

## Decision

- Status: `BLOCKED`

## Next recommended PR

- `Blocking fix PR`