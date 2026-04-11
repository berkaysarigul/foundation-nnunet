# Foundation-nnU-Net Validation Checklist

Purpose: strict pass/fail validation rules for closing recovery tasks.

Use:
- Run the relevant checklist items before closing any task in `RECOVERY_TODO.md`.
- If validation logic changes, update this file and record the change in `DECISIONS.md` when it affects methodology.

## 1. Label decoding correctness

What to check:
- The accepted RLE decoder reproduces authoritative mask behavior for curated positive, negative, and multi-region samples.
- The annotation source used for label validation matches the raw file actually present in the workspace.
- For this workspace, corpus-level auto-resolution must select `cumulative_gap_pairs` for `data/raw/SIIM-ACR/train-rle.csv`.

How to check it:
- Decode curated examples using the repository decoder and compare against authoritative references.
- Verify mask shape, orientation, sparsity, region count, and exact or near-exact overlap where a golden mask exists.
- Confirm the recovery documents point to `data/raw/SIIM-ACR/train-rle.csv` when that is the file present locally, and do not rely on absent helper filenames as source-of-truth.
- Run `py -3 -m unittest tests.test_rle_contract -v` and confirm the golden harness passes for:
  - negative empty-mask fixtures
  - edge-case zero-gap fixtures
  - multi-region synthetic fixtures
  - curated local CSV cases
- Run `py -3 scripts/validate_siim_rle_contract.py` and verify:
  - `resolved_mode=cumulative_gap_pairs`
  - `valid_absolute_pairs=0`
  - `valid_cumulative_gap_pairs` matches the positive-row count
- Confirm that explicitly requesting `absolute_pairs` for the shipped local CSV fails fast instead of decoding silently.

Failure symptoms:
- Masks appear transposed, mirrored, implausibly tiny/large, striped, or inconsistent with the annotation source.
- Positive rows with suspicious token patterns still cannot be explained by the accepted contract.
- Recovery logic still relies on absent raw annotation/helper files such as `stage_2_train.csv` or `mask_functions.py`.
- Auto-resolution is ambiguous or resolves to `absolute_pairs` on the shipped local corpus.
- The preprocessing path still accepts an incompatible explicit mode without error.

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
- Run `py -3 scripts/audit_dicom_intensity.py --sample_size 128 --preview_dir <tmp_dir> --preview_limit 3` and verify:
  - metadata audit reports `MONOCHROME2`, `CR`, 8-bit single-channel images across the corpus
  - all rescale/window/VOI fields are absent for the local bundle
  - exported preview PNGs look anatomically plausible and not contrast-inverted
- After P0.7 regeneration, run `py -3 scripts/validate_processed_dataset.py --dataset_dir data/processed/pneumothorax_trusted_v1 --preview_dir <tmp_dir> --preview_limit 3` and verify the exported overlays are anatomically aligned for both `original_masks` and `dilated_masks`.

Failure symptoms:
- Systematic left-right flips, edge offsets, masks outside plausible pleural regions, or dilation applied in a way that destroys target meaning.

What to do if it fails:
- Audit orientation handling, interpolation choices, resize order, and dilation stage.
- Do not regenerate a trusted dataset version until overlays pass.

## 3. Split leakage and split policy

What to check:
- Train/val/test image IDs are disjoint and class ratios meet the accepted split policy.
- For publication-facing stratified splits, the class label used for stratification is the binary image-level label derived from `original_masks` foreground presence, not `dilated_masks`.

How to check it:
- Run `py -3 -m unittest tests.test_stratified_splits -v` and confirm the deterministic split helper is:
  - reproducible for the same seed
  - disjoint across train/val/test
  - sorted by image ID in each split
  - class-ratio preserving on a synthetic binary-label fixture
- Perform explicit set-intersection checks and compute positive/negative ratios for each split.
- Verify split seed and policy against the dataset manifest.
- For the publication-facing regenerated split, verify the policy matches D-023 exactly:
  - two-stage stratified `train_test_split`
  - `random_state=42`
  - first split `test_size=0.15`
  - second split `test_size=0.17647058823529413` on the remaining `train_val`
  - final split IDs stored in sorted order
- For a regenerated stratified split, verify each split's positive ratio stays within `1.0` absolute percentage point of the dataset-wide positive ratio unless integer rounding makes that impossible.
- For the current trusted dataset version, run `py -3 scripts/validate_processed_dataset.py --dataset_dir data/processed/pneumothorax_trusted_v1` and verify:
  - split union equals the processed image ID set
  - no split overlap is reported
  - the current seed-42 unstratified split counts are `7471 / 1602 / 1602`

Failure symptoms:
- Any overlap, missing IDs, duplicated IDs, or unexpected class-ratio drift.
- Stratification labels are computed from `dilated_masks` or another non-official target definition.

What to do if it fails:
- Regenerate the split deterministically and update the split fingerprint.
- Re-run downstream dataset stats before training.

## 4. Mask variant integrity

What to check:
- Original and dilated masks are both present, binary, and clearly labeled as separate variants.
- The processed dataset contract records default training and evaluation mask variants explicitly.

How to check it:
- Before full dataset regeneration, run `py -3 -m unittest tests.test_mask_variants -v` and confirm:
  - the default training variant is `dilated_masks`
  - the default evaluation variant is `original_masks`
  - the manifest records both variant directories and their scientific intent
- Sample files from each variant and verify unique values, dimensions, naming, and manifest references.
- After P0.7 dataset regeneration, inspect `mask_variants.json` and verify it matches the accepted policy.
- After P0.7 dataset regeneration, run `py -3 scripts/validate_processed_dataset.py --dataset_dir data/processed/pneumothorax_trusted_v1` and verify:
  - `images = original_masks = dilated_masks = 10675`
  - `positive_images = 2379`
  - both mask variants remain binary and manifest-consistent

Failure symptoms:
- Original masks overwritten by dilated masks, grayscale mask values, mismatched counts, or ambiguous variant naming.
- Training/evaluation defaults are implicit, contradictory, or missing from the processed dataset contract.

What to do if it fails:
- Stop all experiments using the processed dataset.
- Repair preprocessing outputs and regenerate the dataset.

## 5. Metric correctness

What to check:
- Dice, IoU, precision, recall, F1, and any optional metrics match the accepted mathematical definitions and edge-case policy.
- The primary model-selection metric is `val_dice_pos_mean` and is implemented as positive-only per-image mean Dice.

How to check it:
- Use handcrafted prediction/target pairs covering empty-empty, empty-positive, positive-empty, and partial-overlap cases.
- Run `py -3 -m unittest tests.test_metrics_reduction -v` and confirm:
  - Dice supports `micro`, `mean`, `positive_mean`, and `none`
  - `mean`, `positive_mean`, and `micro` produce distinct expected values on a mixed handcrafted batch
  - empty-empty overlap metrics evaluate to `1.0`
  - one-empty-one-positive overlap metrics evaluate to `0.0`
  - `positive_mean` returns `NaN` when there are no positive target images
- Run `py -3 -m unittest tests.test_evaluate_metrics_backend -v` and confirm evaluator-side per-image records are produced through `reduction="none"` rather than implicit batch-micro reduction.
- Run `py -3 -m unittest tests.test_trainer_validation_aggregation -v` and confirm trainer-side all-image Dice/IoU means are reconstructed from per-image sums and counts, not from averaging per-batch micro metrics.
- Run `py -3 -m unittest tests.test_trainer_validation_aggregation -v` and confirm trainer-side `val_dice_pos_mean` is reconstructed from positive-image Dice sums and positive image counts, not from averaging per-batch micro Dice over positive subsets.
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
- Run `py -3 -m unittest tests.test_trainer_evaluator_parity -v` and confirm:
  - the same saved prediction fixture is loaded once and consumed by both paths
  - trainer-side all-image Dice/IoU means match evaluator-side per-image record means
  - trainer-side positive-only Dice mean matches the evaluator-side mean over positive records only
- If debugging outside the test harness, run both paths on the same sample batch or saved prediction set and compare outputs field-by-field.

Failure symptoms:
- Same predictions produce different Dice/IoU values or different image counts.

What to do if it fails:
- Treat all model-selection history as untrusted.
- Fix the shared metric path before continuing.

## 7. Stale artifact invalidation

What to check:
- No legacy artifact is being used as authoritative evidence.
- The repository-level location used for authoritative experiment runs is not `results/`.
- Notebook-generated outputs are not being used as evidence unless they are traceable to exact config, checkpoint, and dataset version or fingerprint.

How to check it:
- Confirm every cited metric table, plot, or prediction sample is traceable to a run with config, dataset version, checkpoint, and threshold metadata.
- Confirm authoritative runs are located under `artifacts/runs/`, not under `results/`.
- Confirm any notebook-derived figure, table, metric, or sample cited as evidence has explicit traceability to config, checkpoint, and dataset version/fingerprint.

Failure symptoms:
- An artifact cannot be linked to a run manifest, uses fake IDs, or contradicts current output schema.
- New authoritative outputs are still being placed under `results/`.
- Notebook outputs are cited in decisions or reports without the required provenance trail.

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
- Run `py -3 -m unittest tests.test_trainer_config_surface -v` and confirm:
  - `loss.type=dice_focal` instantiates the trusted `DiceFocalLoss`
  - `training.optimizer` selects `AdamW` or `Adam` exactly as requested
  - `training.scheduler` selects `ReduceLROnPlateau` or `none` exactly as requested
  - unsupported trainer config values fail fast instead of silently falling back
  - legacy `checkpoints/last_*.pth` resume files without `training_components` metadata are rejected
- Inspect `data/processed/pneumothorax_trusted_v1/dataset_manifest.json` and confirm it records:
  - `dataset_version`
  - `dataset_fingerprint`
  - `raw_source.annotation_csv_sha256`
  - `generation.resolved_rle_mode`
  - `generation.seed`
  - `mask_statistics`
  - `split_summary`
  - `fingerprints.splits`
- Run `py -3 scripts/validate_processed_dataset.py --dataset_dir data/processed/pneumothorax_trusted_v1` and confirm manifest statistics and fingerprints match the on-disk files.
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
