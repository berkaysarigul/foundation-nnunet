# Foundation-nnU-Net Validation Checklist

Purpose: strict pass/fail validation rules for closing recovery tasks.

Use:
- Run the relevant checklist items before closing any task in `RECOVERY_TODO.md`.
- If validation logic changes, update this file and record the change in `DECISIONS.md` when it affects methodology.

## 1. Label decoding correctness

What to check:
- The accepted RLE decoder reproduces authoritative mask behavior for curated positive, negative, and multi-region samples.

How to check it:
- Decode curated examples using the repository decoder and compare against authoritative references.
- Verify mask shape, orientation, sparsity, region count, and exact or near-exact overlap where a golden mask exists.

Failure symptoms:
- Masks appear transposed, mirrored, implausibly tiny/large, striped, or inconsistent with the annotation source.
- Positive rows with suspicious token patterns still cannot be explained by the accepted contract.

What to do if it fails:
- Stop all model work.
- Reopen the decoder contract decision.
- Re-check indexing, memory order, merge logic, and annotation-source assumptions.

## 2. Image-mask overlay sanity

What to check:
- Preprocessed images and masks are anatomically aligned and visually plausible after resize and any dilation.

How to check it:
- Review overlays for a curated sample of positives and negatives spanning small, medium, and large masks.
- Inspect both original-mask and dilated-mask variants.

Failure symptoms:
- Systematic left-right flips, edge offsets, masks outside plausible pleural regions, or dilation applied in a way that destroys target meaning.

What to do if it fails:
- Audit orientation handling, interpolation choices, resize order, and dilation stage.
- Do not regenerate a trusted dataset version until overlays pass.

## 3. Split leakage and split policy

What to check:
- Train/val/test image IDs are disjoint and class ratios meet the accepted split policy.

How to check it:
- Perform explicit set-intersection checks and compute positive/negative ratios for each split.
- Verify split seed and policy against the dataset manifest.

Failure symptoms:
- Any overlap, missing IDs, duplicated IDs, or unexpected class-ratio drift.

What to do if it fails:
- Regenerate the split deterministically and update the split fingerprint.
- Re-run downstream dataset stats before training.

## 4. Mask variant integrity

What to check:
- Original and dilated masks are both present, binary, and clearly labeled as separate variants.

How to check it:
- Sample files from each variant and verify unique values, dimensions, naming, and manifest references.

Failure symptoms:
- Original masks overwritten by dilated masks, grayscale mask values, mismatched counts, or ambiguous variant naming.

What to do if it fails:
- Stop all experiments using the processed dataset.
- Repair preprocessing outputs and regenerate the dataset.

## 5. Metric correctness

What to check:
- Dice, IoU, precision, recall, F1, and any optional metrics match the accepted mathematical definitions and edge-case policy.
- The primary model-selection metric is `val_dice_pos_mean` and is implemented as positive-only per-image mean Dice.

How to check it:
- Use handcrafted prediction/target pairs covering empty-empty, empty-positive, positive-empty, and partial-overlap cases.
- Verify per-image and positive-only reductions explicitly.
- Verify that checkpoint-selection calculations exclude negative images from the primary metric and do not use batch-level micro aggregation.

Failure symptoms:
- Trainer and evaluator disagree, positive-only metrics change with batch composition, or empty-mask behavior is unstable/undefined.
- Checkpoint ranking uses `val_loss`, all-image Dice, or batch-micro Dice instead of `val_dice_pos_mean`.

What to do if it fails:
- Stop using current metrics for checkpoint selection.
- Unify metric logic in one backend and re-run parity tests.

## 6. Trainer / evaluator parity

What to check:
- Given the same saved predictions and targets, trainer-side validation metrics and offline evaluator metrics agree.

How to check it:
- Run both paths on the same sample batch or saved prediction set and compare outputs field-by-field.

Failure symptoms:
- Same predictions produce different Dice/IoU values or different image counts.

What to do if it fails:
- Treat all model-selection history as untrusted.
- Fix the shared metric path before continuing.

## 7. Stale artifact invalidation

What to check:
- No legacy artifact is being used as authoritative evidence.
- The repository-level location used for authoritative experiment runs is not `results/`.

How to check it:
- Confirm every cited metric table, plot, or prediction sample is traceable to a run with config, dataset version, checkpoint, and threshold metadata.
- Confirm authoritative runs are located under `artifacts/runs/`, not under `results/`.

Failure symptoms:
- An artifact cannot be linked to a run manifest, uses fake IDs, or contradicts current output schema.
- New authoritative outputs are still being placed under `results/`.

What to do if it fails:
- Mark it legacy immediately and remove it from comparison workflows.

## 8. Threshold tuning discipline

What to check:
- Threshold and post-processing selection uses validation data only and the chosen values are recorded.

How to check it:
- Review the tuning script/run metadata and ensure the selected threshold is frozen before test evaluation.

Failure symptoms:
- Threshold chosen on test data, threshold not recorded, or repeated manual tuning against test outputs.

What to do if it fails:
- Invalidate the reported test metrics and rerun selection on validation only.

## 9. Reproducibility and provenance

What to check:
- Every run records code/config/dataset/checkpoint/seed/threshold metadata and can be reconstructed later.

How to check it:
- Inspect run directories and confirm required metadata files are present and internally consistent.
- Confirm the relative metadata layout inside a run directory matches:
  - `<run_dir>/metadata/run_metadata.yaml`
  - `<run_dir>/metadata/config_snapshot.yaml`
- For an authoritative training run, confirm the minimum mandatory output set exists:
  - `<run_dir>/metrics/history.csv`
  - `<run_dir>/checkpoints/best_checkpoint_metadata.yaml`
  - `<run_dir>/selection/selection_state.yaml`
  - `<run_dir>/qualitative/validation_samples/`
- Confirm the minimum metadata record includes:
  - `run_id`
  - `started_at`
  - `model_type`
  - `config_path`
  - `config_hash`
  - `code_revision` or `code_fingerprint`
  - `code_fingerprint_scope`
  - `dataset_root`
  - `dataset_fingerprint`
  - `split_fingerprint`
  - `train_mask_variant`
  - `eval_mask_variant`
  - `initial_checkpoint_path`
  - `resume_checkpoint_path`
  - `input_size`
  - `seed`
  - `selection_metric`
  - `selected_threshold`
  - `selected_postprocess`

Failure symptoms:
- Missing config snapshot, unclear dataset version, unknown threshold, or ambiguous checkpoint origin.
- Missing code fingerprint fallback in a non-Git environment.
- Missing distinction between training and evaluation mask variants.
- Missing required training outputs even when metadata exists.

What to do if it fails:
- Do not promote the run into result summaries.
- Repair provenance logging before launching further experiments.

## 10. Gradient flow for frozen vs unfrozen backbone

What to check:
- Frozen mode yields zero backbone gradients; unfrozen mode yields expected nonzero backbone gradients.

How to check it:
- Run a controlled forward/backward pass in both modes and inspect gradient norms and optimizer parameter groups.

Failure symptoms:
- Backbone gradients stay zero when unfrozen, or update when meant to be frozen.

What to do if it fails:
- Reopen the `no_grad` and parameter-freezing logic before any hybrid training.

## 11. Hybrid scale alignment

What to check:
- Each Foundation X feature map is fused into the semantically matched stage of the segmentation model.

How to check it:
- Print or assert feature shapes for 256 and 512 inputs, then compare them to the documented fusion design.

Failure symptoms:
- Large corrective upsampling into shallow encoder stages, missing bottleneck use of deep features, or undocumented stage remapping.

What to do if it fails:
- Do not optimize the hybrid.
- Redesign the fusion map and bottleneck/context usage first.

## 12. Baseline gate

What to check:
- A strong pretrained supervised baseline exists before hybrid work resumes.

How to check it:
- Confirm a trusted pretrained-encoder baseline has run end-to-end with corrected metrics, threshold selection, and reproducible outputs.

Failure symptoms:
- Hybrid work starts while the baseline is still the weak random-init U-Net or while trust blockers remain open.

What to do if it fails:
- Pause hybrid work and return to Phase 3 baseline tasks.
