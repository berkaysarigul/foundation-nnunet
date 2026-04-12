# Foundation-nnU-Net Agent Context

Current phase:
- Phase 3 baseline follow-up

Current blocker:
- The repository now has a trusted regenerated dataset, corrected per-image validation metrics, demonstrated trainer/evaluator parity, a refreshed publication-facing stratified split, an accepted immediate trainer config surface, a complete validation-only threshold-selection path, a chosen pretrained baseline family, a fixed fair comparison protocol, a fixed baseline-gate output package, a concrete pretrained model path in code, trainer-side authoritative run artifact emission, a validated evaluation-side artifact path under the same authoritative run directory, a dedicated authoritative pretrained-run config, a Colab-friendly single entrypoint that chains `train -> select -> test` under one authoritative run directory, a safe `select_test` runner stage for continuing from an existing `best_checkpoint.pth` without reopening training, one completed authoritative pretrained baseline run on GPU/Colab, a fixed `P1.7` crop/ROI gate, a fixed immediate crop-comparison policy, and now an implemented D-031 train-only ROI crop path in code. The next blocker is executing that single approved crop-comparison arm on GPU under the same corrected protocol.

Highest-priority open tasks:
1. Execute the fixed `P1.7` crop arm from D-031 on a GPU-capable environment via `configs/pretrained_resnet34_roi_crop_authoritative.yaml`, keeping validation/test full-image so the result stays comparable to the current trusted full-image baseline.
2. Keep all future model comparisons tied to the trusted dataset and corrected metric path.
3. Keep hybrid work paused until the ROI/crop decision and baseline evidence are stable.
4. Keep any future reruns on GPU-capable environments, because this local desktop runtime still reports `torch 2.11.0+cpu` with `cuda_available=False`.

What is already trusted:
- The high-level repo structure and module boundaries.
- The baseline U-Net implementation is structurally standard.
- The current audit findings are the source of truth for recovery planning.
- The current `results/` directory is explicitly quarantined as legacy/non-authoritative and should not be used as evidence.
- The workspace is now initialized as a Git repository and connected to `origin` at `https://github.com/berkaysarigul/foundation-nnunet.git`.
- The minimum provenance contract now requires a Git revision when available, otherwise a deterministic code fingerprint.
- The relative metadata layout inside a future authoritative run directory is now defined in repo memory.
- The minimum mandatory output set for an authoritative training run is now defined in repo memory.
- The authoritative primary model-selection metric is now defined as positive-only per-image validation Dice.
- Legacy artifacts will remain in place with explicit warnings until a separate authoritative output location is defined.
- The repository-level authoritative output location is now defined as `artifacts/runs/`.
- Notebook-generated outputs are non-authoritative by default unless they are traceable to exact config, checkpoint, and dataset version/fingerprint.
- The authoritative local raw annotation source in this workspace is `data/raw/SIIM-ACR/train-rle.csv`; stale references to `stage_2_train.csv` and `mask_functions.py` are not source-of-truth here.
- The accepted local SIIM decoder contract is now `cumulative_gap_pairs` with Fortran-order mask layout and `-1` for empty masks.
- `scripts/validate_siim_rle_contract.py` reproduces the corpus evidence that `train-rle.csv` resolves unambiguously to `cumulative_gap_pairs` and that `absolute_pairs` is incompatible with the shipped positives.
- `src/data/preprocess.py` now validates the corpus RLE mode before preprocessing and supports explicit compatibility modes only when they match the corpus.
- `tests/test_rle_contract.py` and `tests/fixtures/siim_rle_golden_cases.json` now provide a repeatable golden decode harness covering negative, edge-case, multi-region, and curated local CSV examples.
- The canonical regression command for RLE trust is now `py -3 -m unittest tests.test_rle_contract -v`.
- The processed dataset contract now uses separate `original_masks/` and `dilated_masks/` directories plus `mask_variants.json`.
- The default mask policy is now explicit in code and config: train on `dilated_masks`, validate/test/report on `original_masks` unless a run records a different variant deliberately.
- `tests/test_mask_variants.py` is now the canonical contract-level smoke test for mask-variant defaults and manifest semantics.
- The local SIIM DICOM bundle has now been audited end-to-end: all 10,712 training DICOMs are `CR`, `MONOCHROME2`, single-channel, unsigned 8-bit images with no modality LUT, rescale slope/intercept, VOI LUT, or window-center/width tags.
- The accepted DICOM intensity policy is now: preserve native 8-bit intensities when already in display range, apply modality/VOI transforms only if present, invert `MONOCHROME1` if encountered, and use Windows long-path-safe reads for local preprocessing.
- `scripts/audit_dicom_intensity.py` is now the canonical metadata audit entrypoint and `tests/test_dicom_intensity_policy.py` is the canonical unit-test harness for the intensity policy.
- The canonical trusted processed dataset root is now `data/processed/pneumothorax_trusted_v1`.
- `data/processed/pneumothorax_trusted_v1/dataset_manifest.json` is now the authoritative dataset manifest, with dataset fingerprint `c47230301897c0b474bef236a09e6151b74911d75e739575ab91043bc6cc7b6d` and split fingerprint `40d562818cf8f128b0c14bfaccb8d7dae2a49b9380aac48832d70d70cb0dc695`.
- The trusted regenerated dataset currently contains 10,675 images, 2,379 positive studies, and 8,296 negative studies under the accepted local raw bundle.
- `scripts/validate_processed_dataset.py` is now the canonical end-to-end validation entrypoint for the trusted processed dataset contract.
- `configs/config.yaml` now points `data.processed_dir` to `data/processed/pneumothorax_trusted_v1`.
- `src/training/metrics.py` now exposes explicit overlap-metric reductions: `micro`, `mean`, `positive_mean`, and `none`.
- The accepted overlap-metric empty-mask policy is now explicit and regression-tested: empty-empty => `1.0`, one-empty-one-positive => `0.0`, and `positive_mean` returns `NaN` when no positive targets exist.
- `tests/test_metrics_reduction.py` is now the canonical regression harness for metric reductions and empty-mask edge cases.
- `src/evaluation/evaluate.py` now routes per-image overlap metrics through the shared backend with explicit `reduction="none"`.
- `tests/test_evaluate_metrics_backend.py` is now the canonical evaluator-side regression harness for per-image metric wiring.
- `src/training/trainer.py` now aggregates all-image validation Dice/IoU from per-image scores instead of averaging implicit batch-micro metrics over loader steps.
- `src/training/trainer.py` now aggregates `val_dice_pos_mean` from positive-image Dice sums and positive image counts rather than batch-level micro Dice over positive subsets.
- `tests/test_trainer_validation_aggregation.py` is now the canonical regression harness for trainer-side all-image validation aggregation.
- `scheduler.step(val_dice_pos_mean)` and best-checkpoint ranking now operate on the corrected positive-image mean Dice path.
- `tests/test_trainer_evaluator_parity.py` now proves that trainer-side aggregated Dice/IoU/positive-Dice match evaluator-side per-image records on the same saved prediction fixture.
- The accepted publication-facing stratification target is now the binary image-level positive/negative label derived from `original_masks` foreground presence, with target split proportions `70 / 15 / 15` and a desired per-split positive-ratio deviation of at most `1.0` absolute percentage point from the dataset-wide ratio.
- The current trusted split does not meet that publication-facing target closely enough to preserve: train `22.6610%`, val `21.2235%`, test `21.5980%` versus dataset-wide `22.2857%`, so a regenerated stratified split is now the accepted direction.
- The final deterministic publication-facing split policy is now fixed: two-stage stratified `train_test_split` with seed `42`, `15%` test, then `17.647058823529413%` validation from the remaining `train_val`, using image-level labels from `original_masks`, and sorted final split IDs.
- `src/data/preprocess.py::create_splits` now implements that deterministic two-stage stratified policy in code.
- `scripts/regenerate_trusted_split.py` is now the canonical helper for refreshing `splits.json` and `dataset_manifest.json` under the fixed stratified policy.
- The refreshed trusted split now matches the policy target closely: train `22.2862%`, val `22.2846%`, test `22.2846%` versus dataset-wide `22.2857%`.
- The previous config/trainer mismatch inventory has now been resolved for the accepted immediate surface.
- `src/training/trainer.py` now resolves an accepted immediate config surface instead of silently hardcoding training components.
- The accepted immediate trainer config surface is now:
  - `loss.type`: `dice_focal`
  - `training.optimizer`: `AdamW` or `Adam`
  - `training.scheduler`: `ReduceLROnPlateau` or `none`
- Unsupported trainer config values now fail fast at startup instead of silently falling back.
- Resume checkpoints for authoritative runs must now carry canonical `training_components` metadata; legacy `checkpoints/last_*.pth` files without that metadata are rejected.
- `tests/test_trainer_config_surface.py` is now the canonical regression harness for the accepted immediate trainer config surface.
- `src/evaluation/evaluate.py` now exposes a validation-only threshold-selection helper that sweeps an accepted immediate threshold grid over the corrected per-image metric backend.
- The accepted immediate threshold-selection surface is now:
  - `selection.metric`: `val_dice_pos_mean`
  - `selection.threshold_candidates`: `0.05` to `0.95` inclusive in `0.05` steps
  - `selection.postprocess`: `none`
- Threshold selection now fails fast if asked to tune on any split other than `val`.
- Validation threshold selection is now persisted to `<run_dir>/selection/selection_state.yaml`.
- Test evaluation now requires `selection_state.yaml` input and validates that its `model_type`, `checkpoint_path`, `dataset_root`, `eval_mask_variant`, and `input_size` match the current evaluation context before using the selected threshold.
- `tests/test_threshold_selection.py` is now the canonical regression harness for validation-only threshold selection, selection-state persistence, and selection-state reuse.
- The selected primary pretrained baseline family for immediate implementation is now `ImageNet-pretrained ResNet34 encoder U-Net`, implemented within the existing repo stack rather than by introducing a new segmentation framework dependency.
- This baseline family is currently the publication anchor because it is the lowest-risk strong supervised upgrade over the current plain U-Net while staying compatible with the existing `torchvision`/`timm` dependency surface and standard encoder-decoder segmentation practice.
- The immediate fair-comparison protocol for the first strong supervised baseline is now fixed:
  - compare the corrected plain U-Net against the future `ImageNet-pretrained ResNet34 encoder U-Net`
  - keep the trusted data/eval path fixed: `data/processed/pneumothorax_trusted_v1`, the current trusted split fingerprint, `input_size=512`, `train_mask_variant=dilated_masks`, and `eval_mask_variant=original_masks`
  - keep the current augmentation/sampling/training surface fixed: `get_train_transforms()`, the existing weighted train sampler policy, `loss.type=dice_focal`, `training.optimizer=AdamW`, `training.learning_rate=1e-4`, `training.weight_decay=0.01`, `training.scheduler=ReduceLROnPlateau`, `training.batch_size=8`, `training.epochs=150`, `training.early_stopping_patience=30`, and `seed=42`
  - keep the corrected selection/reporting path fixed: checkpoint ranking by `val_dice_pos_mean`, validation-only threshold selection under D-025/D-026, and `selection.postprocess=none`
  - allow only the architecture/initialization change for the first comparison: no ROI/crop, no test-time augmentation, no model-specific hyperparameter sweep, no staged freezing/unfreezing, no hybrid features, and no alternate dataset variant; any grayscale adaptation required by the pretrained encoder must stay inside the model path rather than introducing a separate RGB dataset pipeline
- The required authoritative output package for the first pretrained baseline gate is now fixed:
  - D-010 minimum training outputs still apply under the run directory
  - the tuned validation threshold artifact remains the canonical `<run_dir>/selection/selection_state.yaml`
  - the held-out test report must include a machine-readable per-image file at `<run_dir>/reports/test_metrics.csv` plus an aggregated summary at `<run_dir>/reports/test_summary.yaml`
  - qualitative evidence must include both `<run_dir>/qualitative/validation_samples/` and `<run_dir>/qualitative/test_samples/`
  - each qualitative directory must carry a manifest describing the selected image IDs so the evidence package is auditable rather than ad hoc
  - exact trainer/evaluator output column naming remains deferred to P1.2, but the baseline-gate package must already be sufficient to recover real image IDs, split identity, eval mask variant, threshold/postprocess context, and corrected per-image metrics
- `src/models/resnet34_unet.py` now provides the concrete `PretrainedResNet34UNet` implementation for the selected pretrained baseline family.
- The pretrained baseline path now keeps grayscale adaptation inside the model by replacing the ResNet34 stem conv with a 1-channel version initialized from the RGB pretrained filters, rather than introducing a separate RGB dataset pipeline.
- `src/training/trainer.py::build_model` and `src/evaluation/evaluate.py::build_model` now accept `model_type=pretrained_resnet34_unet`.
- `tests/test_pretrained_resnet34_unet.py` is now the canonical targeted regression harness for the pretrained baseline model path and factory wiring.
- `src/training/run_artifacts.py` is now the canonical helper module for authoritative training-side run directories, provenance metadata, config snapshots, history output, code fingerprint fallback, and best-checkpoint metadata.
- `src/training/trainer.py` now creates or reuses an authoritative run directory under `artifacts/runs/` via `--run_dir` and writes training-side artifacts there instead of leaking new authoritative outputs into `results/` or top-level `checkpoints/`.
- Trainer-side authoritative outputs now include:
  - `<run_dir>/metadata/run_metadata.yaml`
  - `<run_dir>/metadata/config_snapshot.yaml`
  - `<run_dir>/metrics/history.csv`
  - `<run_dir>/checkpoints/best_checkpoint.pth`
  - `<run_dir>/checkpoints/last_checkpoint.pth`
  - `<run_dir>/checkpoints/best_checkpoint_metadata.yaml`
- The trainer now initializes best-checkpoint selection from `-inf` so the first epoch always materializes a best checkpoint artifact and matching metadata instead of risking an empty best-checkpoint path when the first corrected metric equals `0.0`.
- `tests/test_run_artifacts.py` is now the canonical targeted regression harness for authoritative trainer-side run artifact helpers and provenance payloads.
- `src/evaluation/evaluate.py` now derives evaluation-side authoritative outputs from `<run_dir>/selection/selection_state.yaml` so that validation threshold state, held-out test reports, and validation/test qualitative packages land under the same trainer-created `artifacts/runs/<run_id>/` tree instead of `results/`.
- Evaluation-side authoritative outputs now target:
  - `<run_dir>/selection/selection_state.yaml`
  - `<run_dir>/reports/test_metrics.csv`
  - `<run_dir>/reports/test_summary.yaml`
  - `<run_dir>/qualitative/validation_samples/`
  - `<run_dir>/qualitative/test_samples/`
- Validation and test qualitative packages now write a deterministic manifest plus per-sample image, target-mask, prediction-mask, and overlay PNG files for up to four positives and four negatives per split in split order.
- Evaluation now syncs `metadata/run_metadata.yaml` with the selected threshold and selected post-processing state once `selection_state.yaml` is written or reused.
- `tests/test_evaluation_run_outputs.py` is now the canonical evaluator-side regression harness for authoritative run-directory output emission, and it passes alongside `tests.test_threshold_selection`, `tests.test_run_artifacts`, and `tests.test_evaluate_metrics_backend` under `C:\Users\beko5\AppData\Local\Programs\Python\Python310\python.exe`.
- `configs/pretrained_resnet34_authoritative.yaml` is now the dedicated config for the first authoritative `pretrained_resnet34_unet` baseline run and locks the fixed D-028 protocol while changing only the architecture/initialization path relative to the corrected plain U-Net baseline.
- `tests/test_authoritative_pretrained_config.py` is now the canonical regression harness for that dedicated pretrained baseline config, and it passes alongside `tests.test_trainer_config_surface` under `C:\Users\beko5\AppData\Local\Programs\Python\Python310\python.exe`.
- `scripts/run_authoritative_pretrained_baseline.py` is now the Colab-friendly single entrypoint for the first authoritative pretrained baseline and supports `stage=all`, `train`, `select`, `test`, and `select_test` under one authoritative `run_dir`.
- The authoritative pretrained runner fails fast on off-protocol config values instead of silently allowing a widened comparison surface.
- `tests/test_authoritative_pretrained_runner.py` is now the canonical regression harness for the authoritative pretrained runner and proves that `stage=all` reuses one authoritative `run_dir` across `train -> select -> test`, `stage=select_test` reuses an existing best checkpoint without invoking training, and non-`all` stages require an explicit `--run_dir`.
- The first authoritative pretrained baseline run has now completed on GPU/Colab under `/content/drive/MyDrive/foundation_nnunet_runs/resnet34_authoritative_v1`: training was manually stopped after epoch 20 once validation collapsed to empty-mask predictions, the best checkpoint remained epoch 9 with `val_dice_pos_mean=0.5024`, `--stage select_test` reused that checkpoint, `selection_state.yaml` selected threshold `0.95`, and the held-out `test_summary.yaml` reported `1602` test images (`357` positive / `1245` negative) with positive-only Dice mean `0.4951`.
- D-030 now fixes the `P1.7` crop/ROI gate: if the best currently trusted full-image supervised baseline reports held-out `test` positive-only Dice mean below `0.60`, crop/ROI work becomes mandatory on the critical path. The current trusted full-image baseline result (`0.4951`) triggers that gate.
- D-031 now fixes the immediate crop/ROI comparison policy: use train-only mask-guided `384 x 384` ROI crops for positive train images, matched random `384 x 384` crops for negative train images, resize every crop back to `512 x 512` before the model stack, and keep validation/test evaluation full-image with no label-guided evaluation crop path.
- `src/data/dataset.py` now exposes the fixed D-031 train-only crop path behind `data.train_crop`, using mask-guided `384 x 384` crops for positive `train` images, random `384 x 384` crops for negative `train` images, and resize-back-to-`512` before augmentation/model input.
- `src/training/trainer.py` now wires `data.train_crop` into the train dataset only and logs the active crop mode while leaving `val` untouched.
- `configs/pretrained_resnet34_roi_crop_authoritative.yaml` is now the dedicated authoritative config for the immediate D-031 crop comparison arm.
- `tests/test_train_roi_crop_policy.py` and `tests/test_authoritative_pretrained_roi_crop_config.py` are now the canonical targeted regressions for the D-031 crop implementation and crop-run config, and `tests/test_authoritative_pretrained_runner.py` now proves the authoritative runner accepts that dedicated crop config.

What is still untrusted:
- The existing processed dataset under `data/processed/pneumothorax/`, because it predates the corrected RLE contract and mask-variant separation.
- Historical metrics and plots under `results/`, which remain legacy-only artifacts.
- Any future run that bypasses the trusted dataset root or corrected metric path.
- Any trainer config outside the accepted immediate surface until a later decision expands it.
- Any post-processing mode beyond `none` until a later explicit decision expands the search space.
- Any initial baseline comparison that changes more than the architecture/initialization relative to the fixed protocol above.
- Any ROI/crop result until the now-implemented fixed D-031 crop arm is actually run and compared against the trusted full-image baseline under the same corrected protocol.
- Any future pretrained baseline result that lacks the fixed baseline-gate output package under its authoritative run directory.
- Any claim involving Foundation X as clean external pretraining on SIIM.
- The scientific value of the current hybrid design.

Current strategic direction:
- Fix trust issues first, then build a strong pretrained CNN baseline, then decide whether the hybrid is worth redesigning.

Next 3 actions:
1. Execute the fixed D-031 crop comparison on a GPU-capable environment with `configs/pretrained_resnet34_roi_crop_authoritative.yaml`.
2. Evaluate that crop run through the same corrected authoritative artifact path and compare it against the trusted full-image baseline.
3. Keep hybrid work paused until the crop/ROI comparison resolves whether full-image training should be retained or replaced for the paper-path baseline.
