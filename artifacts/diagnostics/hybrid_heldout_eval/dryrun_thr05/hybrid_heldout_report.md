# Hybrid Heldout Evaluation Report

- Hybrid checkpoint: `artifacts\diagnostics\hybrid_controlled_short_train\larger_lr3e5_1200step_bnfix_dryrun\diagnostic_checkpoint_best_dice_pos_mean_thr_050.pth`
- Foundation X: `checkpoints\foundation_x.pth`
- imagesTs: `nnUNet_raw\Dataset101_Pneumothorax\imagesTs`
- heldout_labelsTs: `nnUNet_raw\Dataset101_Pneumothorax\heldout_labelsTs`
- threshold: `0.5`
- img_size: `256`
- device: `cpu`
- generated: `2026-05-26T16:54:24.059217+00:00`

## Status: `PARTIAL_PASS` (exit code 0)

## Comparison vs pure nnU-Net heldout baseline

| metric | nnU-Net baseline | hybrid PR-8 best | Δ (hybrid − baseline) | beats baseline? |
|---|---:|---:|---:|---|
| case-level precision | 0.7719 | 0.0000 | -0.7719 | NO |
| case-level recall | 0.6919 | 0.0000 | -0.6919 | NO |
| case-level specificity | 0.9414 | 0.0000 | -0.9414 | NO |
| case-level F1 | 0.7297 | 0.0000 | -0.7297 | NO |
| mean Dice positive cases | 0.3722 | n/a | n/a | n/a |
| mean Dice detected positives only | 0.5380 | n/a | n/a | n/a |
| negative case FPR (lower is better) | 0.0586 | 1.0000 | +0.9414 | NO |

## Heldout aggregate

- total cases processed: 4
- cases with label: 4
- gt positive cases: 0
- gt negative cases: 4
- pred positive cases: 4
- TP / FN / FP / TN: 0 / 0 / 4 / 0
- mean Dice (all cases): 0.0
- median Dice (all cases): 0.0
- mean IoU (all cases): 0.0
- mean precision (all cases): 0.0
- mean recall (all cases): 0.0
- detected positive count: 0

## Tiny-positive subgroup

- (not computed; no positive cases)

## Checkpoint metadata

- state_dict missing_keys_count: 0
- state_dict unexpected_keys_count: 0
- PR-8 step (per ckpt payload): 0
- PR-8 selection_metric: dice_pos_mean_thr_050
- PR-8 metric_value: 0.02863949304446578

## Outputs

- summary_json: `artifacts\diagnostics\hybrid_heldout_eval\dryrun_thr05\hybrid_heldout_summary.json`
- summary_yaml: `artifacts\diagnostics\hybrid_heldout_eval\dryrun_thr05\hybrid_heldout_summary.yaml`
- report_md: `artifacts\diagnostics\hybrid_heldout_eval\dryrun_thr05\hybrid_heldout_report.md`
- case_csv: `artifacts\diagnostics\hybrid_heldout_eval\dryrun_thr05\hybrid_heldout_case_metrics.csv`
- pred_masks_dir: `artifacts\diagnostics\hybrid_heldout_eval\dryrun_thr05\pred_masks` (saved=True)
- visuals_dir: `artifacts\diagnostics\hybrid_heldout_eval\dryrun_thr05\visuals`
- visual_grids_written: 1

## Failures

None.
