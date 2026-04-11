# Foundation-nnU-Net Agent Context

Current phase:
- Phase 3 baseline preparation

Current blocker:
- The repository now has a trusted regenerated dataset, corrected per-image validation metrics, demonstrated trainer/evaluator parity, a refreshed publication-facing stratified split, an accepted immediate trainer config surface, a complete validation-only threshold-selection path, and a chosen pretrained baseline family. The next blocker is defining the fair training protocol for that baseline and then implementing it without widening scope.

Highest-priority open tasks:
1. Define the fair training protocol for the chosen pretrained baseline relative to the corrected current U-Net.
2. Keep all future model comparisons tied to the trusted dataset and corrected metric path.
3. Implement the chosen pretrained baseline only after the protocol is fixed in repo memory.
4. Keep hybrid work paused until a strong supervised baseline exists.
5. Delay output-schema cleanup until the pretrained baseline path exists.

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

What is still untrusted:
- The existing processed dataset under `data/processed/pneumothorax/`, because it predates the corrected RLE contract and mask-variant separation.
- Historical metrics and plots under `results/`, which remain legacy-only artifacts.
- Any future run that bypasses the trusted dataset root or corrected metric path.
- Any trainer config outside the accepted immediate surface until a later decision expands it.
- Any post-processing mode beyond `none` until a later explicit decision expands the search space.
- Any pretrained baseline result until the selected `ResNet34` encoder path is implemented and run end-to-end under the trusted protocol.
- Any claim involving Foundation X as clean external pretraining on SIIM.
- The scientific value of the current hybrid design.

Current strategic direction:
- Fix trust issues first, then build a strong pretrained CNN baseline, then decide whether the hybrid is worth redesigning.

Next 3 actions:
1. Define the fair training protocol for the selected `ImageNet-pretrained ResNet34 encoder U-Net` relative to the corrected current U-Net.
2. Specify the required authoritative outputs for that baseline run: tuned validation threshold, test report, and qualitative examples.
3. Implement the selected pretrained baseline only after the protocol and output package are fixed in repo memory.
