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
- The current `hausdorff` helper is not part of authoritative paper-path reporting unless a later explicit decision reintroduces it with a correct tested definition.

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
- Legacy documentation that conflicts with recovered methodology is explicitly marked as non-authoritative instead of being left to silently compete with repo memory.

How to check it:
- Confirm every cited metric table, plot, or prediction sample is traceable to a run with config, dataset version, checkpoint, and threshold metadata.
- Confirm authoritative runs are located under `artifacts/runs/`, not under `results/`.
- Confirm any notebook-derived figure, table, metric, or sample cited as evidence has explicit traceability to config, checkpoint, and dataset version/fingerprint.
- Confirm stale guides such as `docs/foundation_nnunet_dev_guide.md` carry an explicit legacy/non-authoritative warning and redirect readers to the recovery-memory files.
- Confirm stale top-level guides such as `CLAUDE.md` carry the same explicit legacy/non-authoritative warning and do not present stale workflow commands as current source-of-truth.
- Confirm stale operational notebooks such as `notebooks/train_colab.ipynb` and `notebooks/train_local.ipynb` carry an explicit legacy/non-authoritative warning before any stale workflow cells.

Failure symptoms:
- An artifact cannot be linked to a run manifest, uses fake IDs, or contradicts current output schema.
- New authoritative outputs are still being placed under `results/`.
- Notebook outputs are cited in decisions or reports without the required provenance trail.
- Legacy docs still read like active source-of-truth and preserve unsafe assumptions about raw annotations, `results/`, or hybrid posture without a warning banner.
- `CLAUDE.md` still reads like the active runbook and preserves stale hybrid-first or `results/`-first operational guidance without a warning banner.
- `notebooks/train_colab.ipynb` still opens as if it were the active authoritative Colab workflow and exposes stale `results/` or hybrid-first cells without a warning banner.
- `notebooks/train_local.ipynb` still opens as if it were the active authoritative local training workflow and exposes stale `results/`, old processed-dataset assumptions, or hybrid-first cells without a warning banner.

What to do if it fails:
- Mark it legacy immediately and remove it from comparison workflows.

## 8. Threshold tuning discipline

What to check:
- Threshold and post-processing selection uses validation data only and the chosen values are recorded.

How to check it:
- Run `py -3 -m unittest tests.test_threshold_selection -v` and confirm:
  - threshold selection only accepts `split="val"`
  - the accepted immediate selection surface is enforced:
    - `selection.metric=val_dice_pos_mean`
    - `selection.threshold_candidates` includes `0.5` and stays inside `(0, 1)`
    - `selection.postprocess=none`
  - selection prefers the threshold that maximizes positive-only validation Dice rather than all-image mean Dice
  - exact-score ties resolve deterministically back toward `0.5`
- Confirm the persisted threshold artifact path matches `<run_dir>/selection/selection_state.yaml`.
- Confirm test evaluation refuses to run without `selection_state.yaml`.
- Confirm test evaluation refuses to reuse `selection_state.yaml` when `model_type`, `checkpoint_path`, `dataset_root`, `eval_mask_variant`, or `input_size` do not match the current evaluation context.
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
- Inspect `selection_state.yaml` and confirm it records:
  - `selection_split`
  - `selection_metric`
  - `selection_state_path`
  - `selected_threshold`
  - `selected_postprocess`
  - `threshold_candidates`
  - `threshold_summary`
  - `model_type`
  - `checkpoint_path`
  - `dataset_root`
  - `train_mask_variant`
  - `eval_mask_variant`
  - `input_size`
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
- Confirm D-036 canonical CSV schema compliance:
  - run `py -3 -m unittest tests.test_run_artifacts -v` and verify authoritative `history.csv` emits the ordered columns:
    - `epoch`
    - `train_loss`
    - `val_loss`
    - `val_dice_mean`
    - `val_dice_pos_mean`
    - `val_iou_mean`
  - confirm legacy in-memory trainer aliases (`val_dice`, `val_dice_pos`, `val_iou`) are upgrade-only and do not appear in newly emitted authoritative `history.csv`
  - run `py -3 -m unittest tests.test_evaluation_run_outputs -v` and verify authoritative `reports/test_metrics.csv` emits the ordered required prefix:
    - `image_id`
    - `split`
    - `model_type`
    - `checkpoint_path`
    - `eval_mask_variant`
    - `selection_metric`
    - `selected_threshold`
    - `selected_postprocess`
    - `positive`
    - `dice`
    - `iou`
    - `precision`
    - `recall`
    - `f1`
  - confirm per-image evaluation outputs preserve exact dataset `image_id` values rather than synthetic row IDs
  - confirm authoritative evaluation outputs that enumerate individual images also carry explicit `subset_tag` values (`positive` / `negative`) instead of relying only on the boolean `positive` flag
  - confirm validation/test qualitative manifests preserve the same `image_id` and `subset_tag` fields for each sampled image entry
  - confirm threshold-reusing evaluation outputs preserve metadata completeness:
    - per-image `reports/test_metrics.csv` rows include `selection_state_path`, `train_mask_variant`, `eval_mask_variant`, `selected_threshold`, and `selected_postprocess`
    - `reports/test_summary.yaml` includes the same reused-threshold and mask-variant context
    - validation/test qualitative manifests include the same reused-threshold and mask-variant context
  - run `py -3 -m unittest tests.test_threshold_selection -v` and confirm test-time selection-state reuse rejects mismatched `train_mask_variant` in addition to other context mismatches
  - confirm authoritative report artifacts do not emit `hausdorff` while D-039 is in force:
    - `reports/test_metrics.csv` has no `hausdorff` column
    - `reports/test_summary.yaml` subset summaries have no `hausdorff` field
    - qualitative manifests do not embed `hausdorff` in per-sample metric payloads

Failure symptoms:
- Missing config snapshot, unclear dataset version, unknown threshold, or ambiguous checkpoint origin.
- Missing code fingerprint fallback in a non-Git environment.
- Missing distinction between training and evaluation mask variants.
- Missing required training outputs even when metadata exists.
- Newly emitted authoritative CSVs still use ambiguous legacy history names or drift away from the D-036 ordered required columns.
- Saved-threshold outputs require cross-reading multiple files to recover threshold provenance or training/evaluation mask-variant roles.
- Test-time selection-state reuse accepts a mismatched `train_mask_variant` without error.
- Authoritative outputs still emit `hausdorff` even though the current helper is not an accepted paper-path metric.

What to do if it fails:
- Do not promote the run into result summaries.
- Repair provenance logging before launching further experiments.

## 10. Gradient flow for frozen vs unfrozen backbone

What to check:
- Frozen mode yields zero backbone gradients; unfrozen mode yields expected nonzero backbone gradients.
- Optimizer parameter groups and backbone mode policy do not silently contradict the intended frozen/unfrozen semantics.

How to check it:
- Run a controlled forward/backward pass in both modes and inspect gradient norms and optimizer parameter groups.
- Before trusting the gradient results, confirm current code-path alignment:
  - `build_optimizer()` filters only `param.requires_grad=True` parameters
  - the trainer does not unconditionally force `model.foundation_x.backbone.eval()` in unfrozen mode
  - no unconditional `torch.no_grad()` remains around the Foundation X path when testing unfrozen behavior
- Run `py -3 -m unittest tests.test_hybrid_backbone_mode_policy -v` and confirm:
  - frozen mode keeps the backbone in `eval()`
  - unfrozen mode is not silently forced back to `eval()` by the trainer helper
- Run `py -3 -m unittest tests.test_hybrid_gradient_flow -v` and confirm:
  - `FoundationXBackbone.forward()` suppresses gradients only in frozen mode
  - the hybrid path yields no Foundation X gradients when frozen
  - the hybrid path yields nonzero Foundation X gradients when unfrozen

Failure symptoms:
- Backbone gradients stay zero when unfrozen, or update when meant to be frozen.
- Backbone parameters appear in optimizer groups but still cannot receive gradients because trainer/mode policy or hidden `no_grad()` overrides unfrozen behavior.
- Trainer-side mode-policy regressions silently re-freeze unfrozen backbones before gradient checks even run.
- Frozen and unfrozen hybrid runs become indistinguishable at the Foundation X gradient level because a hidden `no_grad()` wrapper still survives somewhere in the path.

What to do if it fails:
- Reopen the `no_grad` and parameter-freezing logic before any hybrid training.

## 11. Hybrid scale alignment

What to check:
- Each Foundation X feature map is fused into the semantically matched stage of the segmentation model.

How to check it:
- Print or assert feature shapes for 256 and 512 inputs, then compare them to the documented fusion design.
- Confirm the current-state inventory matches D-054 before proposing a redesign:
  - Foundation X emits `H/4,H/8,H/16,H/32`
  - the current hybrid maps those stages to U-Net `H,H/2,H/4,H/8`
  - every current fusion therefore requires a `4x` upsample into a shallower stage
  - the deepest Foundation X feature is not consumed at a natural `H/32` context slot
- Confirm the proposed corrected mapping matches D-055:
  - `fx[0]->e3`
  - `fx[1]->e4`
  - `fx[2]->H/16 bottleneck/context`
  - `fx[3]->dedicated H/32 context slot`
  - no corrected redesign fuses Foundation X directly into `e1` or `e2`
- Confirm the deeper-context requirement matches D-056:
  - baseline U-Net bottoms out at `H/16`
  - there is no natural `H/32` slot in the current architecture
  - a corrected four-stage hybrid redesign therefore adds an explicit `H/32` context head
  - resize-only reuse of `e4` or the current bottleneck for `fx[3]` is off-protocol
- Confirm the deepest-feature usage rule matches D-057:
  - `fx[3]` enters only through the dedicated `H/32` context head
  - `fx[3]` is processed at native `H/32` before any transition
  - `fx[3]` makes exactly one learned `H/32 -> H/16` transition
  - reconnect happens only through the `fx[2]`-aligned `H/16` context branch
  - direct reuse of `fx[3]` in `e4`, decoder skips, or shallow encoder fusion is forbidden

Failure symptoms:
- Large corrective upsampling into shallow encoder stages, missing bottleneck use of deep features, or undocumented stage remapping.

What to do if it fails:
- Do not optimize the hybrid.
- Redesign the fusion map and bottleneck/context usage first.
- Run `py -3 -m unittest tests.test_hybrid_scale_contract -v` to verify the executable D-058 contract helper still accepts valid `256` and `512` layouts while rejecting scale drift.

## 12. Baseline gate

What to check:
- A strong pretrained supervised baseline exists before hybrid work resumes.
- The first corrected comparison between the plain U-Net and the pretrained `ResNet34` encoder baseline respects the fixed fairness protocol rather than changing multiple knobs at once.
- The first corrected pretrained baseline run carries the fixed baseline-gate evidence package, not just a checkpoint and console output.

How to check it:
- Confirm a trusted pretrained-encoder baseline has run end-to-end with corrected metrics, threshold selection, and reproducible outputs.
- Confirm the initial corrected comparison keeps the shared protocol fixed across both arms:
  - same trusted dataset root and split fingerprint
  - same `input_size=512`
  - same `train_mask_variant=dilated_masks` and `eval_mask_variant=original_masks`
  - same training augmentation path and weighted train sampler policy
  - same `loss.type=dice_focal`
  - same `training.optimizer=AdamW`, `training.learning_rate=1e-4`, and `training.weight_decay=0.01`
  - same `training.scheduler=ReduceLROnPlateau`
  - same `training.batch_size=8`, `training.epochs=150`, `training.early_stopping_patience=30`, and `seed=42`
  - same checkpoint ranking by `val_dice_pos_mean`
  - same validation-only threshold selection path with `selection.postprocess=none`
- Confirm the initial pretrained baseline is fine-tuned end-to-end and that any grayscale adaptation stays inside the model path rather than introducing a separate RGB dataset pipeline.
- Confirm the first comparison does not add ROI/crop preprocessing, test-time augmentation, hybrid/Foundation X components, or model-specific hyperparameter retuning for only one arm.
- Confirm the baseline-gate run directory includes:
  - `<run_dir>/selection/selection_state.yaml`
  - `<run_dir>/reports/test_metrics.csv`
  - `<run_dir>/reports/test_summary.yaml`
  - `<run_dir>/qualitative/validation_samples/`
  - `<run_dir>/qualitative/test_samples/`
- Confirm `reports/test_metrics.csv` is produced from the held-out `test` split using the same best checkpoint and reused `selection_state.yaml` context that the run declares as authoritative.
- Confirm the test report package is sufficient to recover real image IDs, split identity, eval mask variant, selected threshold/postprocess context, and corrected per-image test metrics, even if the final cross-file schema harmonization remains a later P1.2 task.
- Confirm each qualitative package includes a manifest listing the selected image IDs; reject notebook screenshots or ad hoc copied images as substitutes for the authoritative qualitative package.

Failure symptoms:
- Hybrid work starts while the baseline is still the weak random-init U-Net or while trust blockers remain open.
- The pretrained baseline run uses a different data root, split, optimizer, scheduler, loss, threshold policy, or other non-architectural change relative to the corrected plain U-Net comparison arm.
- The first pretrained run quietly adds crop strategy, post-processing, TTA, staged freezing, or a separate RGB dataset path.
- The initial pretrained baseline run is missing `reports/test_metrics.csv`, `reports/test_summary.yaml`, or either qualitative package.
- The qualitative directories contain cherry-picked images with no manifest, or the test report cannot be tied back to the authoritative checkpoint and saved threshold-selection state.

What to do if it fails:
- Pause hybrid work and return to Phase 3 baseline tasks.
- Treat the comparison as protocol-breaking and do not use it as the publication anchor or hybrid decision baseline.

## 13. ROI / crop gate

What to check:
- ROI/crop work is driven by the explicit `P1.7` gate in D-030 rather than ad hoc dissatisfaction with a run.
- If the current trusted full-image baseline reports held-out `test` positive-only Dice mean below `0.60`, a controlled crop/ROI comparison is treated as mandatory before hybrid work resumes or the full-image baseline is accepted as the paper-path anchor.

How to check it:
- Confirm the current trusted full-image baseline evidence package records a held-out `test` positive-only Dice mean and that the value is read from authoritative repo memory, not from legacy artifacts.
- Confirm D-030 remains the source-of-truth gate:
  - `test` positive-only Dice mean `< 0.60` => crop/ROI comparison is mandatory
  - `test` positive-only Dice mean `>= 0.60` => crop/ROI work may remain deferred unless another later decision changes scope
- If the gate is triggered, confirm the subsequent crop/ROI comparison keeps the same trusted dataset root, split, corrected metric path, and non-crop training/evaluation protocol unless a later explicit decision records a justified change.

Failure symptoms:
- Crop/ROI work starts or is skipped without checking the D-030 gate first.
- The gate is triggered but the next comparison silently changes optimizer, threshold policy, dataset root, or other non-crop variables at the same time.
- The crop/ROI decision is justified from legacy artifacts or chat memory instead of the authoritative baseline evidence package.

What to do if it fails:
- Re-anchor the decision to the trusted baseline evidence package.
- Reopen the `P1.7` gate decision before running or citing crop/ROI experiments.

## 14. Immediate ROI / crop policy

What to check:
- The first `P1.7` crop comparison follows the fixed D-031 policy instead of inventing a new crop regime during implementation.
- Train-time ROI use is leakage-safe and evaluation remains directly comparable to the trusted full-image baseline.

How to check it:
- Confirm the implemented immediate crop arm matches D-031 exactly:
  - crop only on the `train` split
  - positive `train` images use mask-guided square ROI crops derived from the current training mask variant
  - negative `train` images use random square crops from the same `512 x 512` image space
  - crop size is `384 x 384`
  - every crop is resized back to `512 x 512` before entering the model stack
  - `val` and `test` remain full-image with no crop path
- Run `py -3 -m unittest tests.test_train_roi_crop_policy -v` and confirm:
  - positive train crops enlarge the sparse ROI after resize-back-to-`512`
  - negative train crops preserve empty masks and output shape `(1, 512, 512)`
  - `data.train_crop` is rejected outside `split='train'`
  - `data.train_crop` fails fast unless `data.input_size=512`
- Run `py -3 -m unittest tests.test_authoritative_pretrained_roi_crop_config -v` and confirm `configs/pretrained_resnet34_roi_crop_authoritative.yaml` keeps the shared authoritative pretrained protocol fixed while setting only:
  - `data.train_crop.mode=roi_train_only`
  - `data.train_crop.crop_size=384`
- Run `py -3 -m unittest tests.test_authoritative_pretrained_runner -v` and confirm the authoritative pretrained runner accepts the dedicated crop config without widening the shared protocol surface.
- Confirm no ground-truth mask, selected threshold, or test-derived signal influences crop placement on `val` or `test`.
- Confirm the crop comparison keeps the same trusted dataset root, split, optimizer, scheduler, threshold-selection policy, and evaluation artifact path as the current trusted full-image baseline.

Failure symptoms:
- Validation/test use label-guided crops or any other ROI shortcut.
- The crop arm silently changes tensor size, dataset root, optimizer, threshold policy, or other non-crop variables at the same time.
- The first crop implementation introduces an additional ROI detector, sliding-window inference path, or test-time crop ensemble beyond the D-031 scope.
- The dedicated crop config drifts away from `384 x 384` train-only ROI crops while still being treated as the immediate `P1.7` compare arm.

What to do if it fails:
- Reject the crop result as non-authoritative for `P1.7`.
- Re-implement the comparison so only the approved D-031 train-time crop policy changes.

## 15. Hybrid keep / drop evidence

What to check:
- No future hybrid candidate is treated as keep-worthy unless it clears both the D-033 performance bar and the D-034 evidence contract.
- Hybrid reopening evidence must be auditable from artifacts, not reconstructed from notebook screenshots or copied console logs.
- Any frozen/unfrozen hybrid claim follows the D-050 semantics contract instead of relying on hidden unconditional `torch.no_grad()` behavior.

How to check it:
- Confirm the future hybrid run uses the same trusted evaluation regime required by D-033:
  - trusted dataset root
  - corrected metric path
  - validation-only threshold selection
  - authoritative run-artifact structure
- Confirm the run directory includes the minimum D-034 artifact family:
  - `<run_dir>/metadata/run_metadata.yaml`
  - `<run_dir>/metadata/config_snapshot.yaml`
  - `<run_dir>/metrics/history.csv`
  - `<run_dir>/checkpoints/best_checkpoint.pth`
  - `<run_dir>/checkpoints/best_checkpoint_metadata.yaml`
  - `<run_dir>/selection/selection_state.yaml`
  - `<run_dir>/reports/test_metrics.csv`
  - `<run_dir>/reports/test_summary.yaml`
  - `<run_dir>/qualitative/validation_samples/`
  - `<run_dir>/qualitative/test_samples/`
- Confirm there is an explicit baseline comparison record tying the hybrid candidate back to the trusted full-image pretrained baseline and reporting:
  - the baseline reference score `0.4951`
  - the hybrid candidate held-out `test` positive-only Dice
  - the absolute delta versus baseline
  - whether the candidate cleared the D-033 keep threshold `>= 0.5151`
- Confirm the engineering-integrity proof set exists and is reviewable:
  - frozen/unfrozen backbone gradient behavior is validated
  - fusion-stage shapes are asserted/documented at the active input size
  - branch-normalization policy is explicit in config and run metadata
- Confirm D-050 semantics are respected before accepting any frozen/unfrozen claim:
  - frozen mode means no gradient path through Foundation X and explicit frozen parameters
  - unfrozen mode means no unconditional `torch.no_grad()` remains in `src/models/hybrid.py` or `src/models/backbone.py`
- Reject notebook screenshots, copied logs, or a single manually reported scalar as sufficient evidence on their own.

Failure symptoms:
- A hybrid run reports a promising score but lacks the authoritative artifact bundle.
- The comparison is made against the wrong reference arm, such as the failed crop run instead of the trusted full-image baseline.
- Gradient-flow, fusion-alignment, or normalization evidence is missing, implicit, or only described informally in chat/notebooks.
- A future hybrid candidate numerically exceeds `0.5151` but cannot be audited from artifacts alone.
- A future hybrid note claims unfrozen behavior while `src/models/hybrid.py` or `src/models/backbone.py` still hardcode unconditional `torch.no_grad()` around the Foundation X path.

What to do if it fails:
- Keep the hybrid deferred.
- Do not promote the run into critical-path work, paper tables, or architectural conclusions.
- Reopen the missing evidence items before spending further time on hybrid optimization.

## 16. Foundation X framing boundary

What to check:
- Any Foundation X or hybrid result reported under the current checkpoint provenance stays inside the D-035 claim boundary.
- No document, notebook, table, or summary presents Foundation X as clean external pretraining or target-unseen generalization on SIIM.
- Under D-040, Foundation X is deferred from the main paper path unless a future hybrid candidate clears D-033 and D-034.
- Under D-041, wording that upgrades Foundation X into generic foundation-model superiority or clean-transfer evidence is explicitly forbidden.
- Under D-042, any allowed Foundation X discussion must compare back to the trusted full-image `pretrained_resnet34_unet` baseline rather than to weaker or legacy anchors.

How to check it:
- Review the relevant result summary, manuscript text, notebook narrative, or decision note against D-006, D-035, D-040, D-041, and D-042.
- Confirm the allowed framing remains one of:
  - leakage-aware in-domain transfer
  - secondary ablation against the trusted full-image pretrained baseline
  - engineering analysis under explicit SIIM-exposure caveats
- Confirm current-state paper placement also respects D-040:
  - Foundation X is absent from the abstract
  - Foundation X is absent from headline results tables and the default main-paper storyline
  - if Foundation X is mentioned before D-033/D-034 are cleared, it appears only as future-work / limitations context or a clearly labeled appendix-side leakage-aware ablation
- Confirm the following claim classes are absent:
  - clean external-pretraining generalization on SIIM
  - target-unseen transfer into SIIM
  - broader foundation-model knowledge isolated from SIIM exposure
  - Foundation X replacing the trusted full-image pretrained baseline as the default headline anchor solely because it scored higher on the current split
- Confirm wording-level inflation is also absent:
  - Foundation X is not described as the `better model`, `stronger model`, or `superior model` by default under the current recovered state
  - any mention of benefit does not hide the SIIM-exposure caveat or the leakage-aware secondary framing
  - any mention of pretraining advantage does not describe the setup as clean external transfer
- Confirm baseline-comparison discipline also holds:
  - the trusted full-image `pretrained_resnet34_unet` baseline is named as the comparison anchor
  - the baseline reference score `0.4951`, the candidate score, and the absolute delta are all stated together
  - the write-up states whether the candidate did or did not clear the D-033 keep threshold `>= 0.5151`
  - the crop run (`0.4625`), plain U-Net, legacy `results/`, and notebook-only metrics are not used as the primary narrative anchor
- If a future hybrid candidate is discussed after clearing D-033 and D-034, confirm the write-up still labels it as leakage-aware secondary evidence unless a later explicit decision introduces a verified non-SIIM-exposed checkpoint.

Failure symptoms:
- Foundation X is described as generic external transfer or clean pretraining on SIIM.
- A hybrid result is promoted to the main paper claim without explicit SIIM-exposure caveats.
- Foundation X appears in the abstract, headline tables, or default narrative even though no hybrid candidate has cleared D-033 and D-034.
- Foundation X is described as the default superior model or its claimed benefit is written without explicit leakage-aware caveats.
- Foundation X is compared primarily against the crop arm, plain U-Net, legacy artifacts, or notebook-only metrics instead of the trusted full-image baseline.
- A Foundation X discussion omits the baseline score `0.4951`, the candidate score, the absolute delta, or the D-033 threshold status.
- The trusted full-image pretrained baseline is displaced as the headline anchor without a separate methodology decision that changes the D-035 boundary.

What to do if it fails:
- Downgrade the wording to the D-035 leakage-aware framing.
- Remove Foundation X from the headline paper path and demote it back to D-040-compatible placement.
- Remove the unsupported claim from paper/reporting artifacts.
- Reopen methodology review before citing the result as evidence.

## 17. Publication-grade evaluation direction

What to check:
- The publication-grade evaluation plan follows D-043 and uses repeated stratified train/val/test splits rather than treating plain 5-fold CV as the default path.
- The chosen direction preserves the already trusted evaluation discipline:
  - train on `train`
  - threshold selection on `val` only
  - held-out reporting on `test`

How to check it:
- Review the relevant decision note, reporting plan, notebook narrative, or orchestration proposal against D-043.
- Confirm the chosen direction is explicitly described as repeated stratified train/val/test splits.
- Confirm any repeated-split proposal preserves:
  - a distinct validation split for threshold selection
  - a distinct held-out test split for final reporting
  - the same trusted dataset/evaluator/selection-state discipline already used by the authoritative baseline path
- Confirm plain 5-fold CV is not presented as the current default publication plan unless a later explicit decision replaces D-043 with a nested-validation design.

Failure symptoms:
- A proposal calls plain 5-fold CV the default publication path under the current stack.
- Validation-only threshold selection disappears or is implicitly merged into the final reporting fold.
- The evaluation plan changes direction without a new explicit decision.

What to do if it fails:
- Re-anchor the plan to D-043.
- Restore the explicit train/val/test roles.
- Reopen methodology review before implementing orchestration changes.

## 18. Split-bootstrap and paired comparison discipline

What to check:
- Publication-grade uncertainty estimates follow D-044 and use the repeated split as the statistical unit.
- Model-vs-model comparisons on the repeated-split path are paired by identical split instances rather than treated as unpaired runs.

How to check it:
- Review the relevant reporting plan, notebook narrative, or orchestration proposal against D-044.
- Confirm single-model uncertainty is described as:
  - mean held-out `test` metric across repeated splits
  - two-sided 95% percentile bootstrap CI over split-level values
- Confirm model-comparison uncertainty is described as:
  - mean paired split-level delta
  - two-sided 95% percentile bootstrap CI over paired deltas
- Confirm the default paired comparison target remains held-out `test` positive-only Dice mean.
- Confirm the pairing rule requires both models to use the exact same repeated split instances.
- Confirm image-level bootstrap is not presented as the default publication CI path.

Failure symptoms:
- Confidence intervals are described as image-level bootstrap by default.
- Model comparisons are treated as unpaired even though repeated split pairing is available.
- The repeated-split plan omits the paired delta and instead compares only separate model means.
- The primary paired comparison target drifts away from held-out `test` positive-only Dice without a new decision.

What to do if it fails:
- Re-anchor the reporting plan to D-044.
- Restore split-level bootstrap and paired-delta language.
- Reopen methodology review before implementing result aggregation.

## 19. Final repeated-split evidence package

What to check:
- Final publication-grade repeated-split reporting follows D-045 and ships the full minimum evidence package rather than only a final averaged table.

How to check it:
- Review the relevant reporting plan, orchestration proposal, or final artifact layout against D-045.
- Confirm the final package includes:
  - a split manifest covering every repeated split instance and its exact train/val/test IDs
  - one authoritative run-artifact package per model per split instance
  - one machine-readable split-level aggregation table
  - one machine-readable paired-delta table for each named model-vs-model comparison
  - one final summary artifact with model means, split-bootstrap 95% CIs, paired-delta means, paired-delta 95% CIs, and contributing split counts
- Confirm the repeated-split package still points back to the same trusted dataset / evaluation regime rather than mixing legacy or notebook-only outputs.

Failure symptoms:
- Only the final average metrics are retained.
- Split-level rows cannot be traced back to authoritative per-split run artifacts.
- Paired comparisons are summarized without a machine-readable paired-delta table.
- The final summary omits confidence intervals or the number of contributing split instances.

What to do if it fails:
- Re-anchor the report bundle to D-045.
- Regenerate the missing machine-readable artifacts before treating the repeated-split result as publication-grade evidence.
