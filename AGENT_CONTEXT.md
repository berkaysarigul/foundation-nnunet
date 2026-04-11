# Foundation-nnU-Net Agent Context

Current phase:
- Phase 2 evaluation correctness

Current blocker:
- The repository now has a trusted regenerated dataset, a tested shared metric reduction backend, and evaluator-side per-image wiring, but trainer-side validation aggregation is still mathematically wrong.

Highest-priority open tasks:
1. Rewrite validation metrics to operate per image on the trusted dataset path.
2. Fix positive-only Dice counting so checkpoint selection matches `val_dice_pos_mean`.
3. Keep all current model comparisons non-authoritative until the corrected metric path is implemented.
4. Preserve strict separation between training mask variants and official reporting mask variants in all future runs.
5. Decide whether the split should now be regenerated under stratification for publication use.

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
- `data/processed/pneumothorax_trusted_v1/dataset_manifest.json` is now the authoritative dataset manifest, with dataset fingerprint `dc198ef70c4e5420e36b22dfb5f6d3ae28a9872bf64ee2e94a193ee711e06c12` and split fingerprint `3fe21a9e65cf883e5303aaad327238a69745f93bfa084b08f364ed058ad4bfc9`.
- The trusted regenerated dataset currently contains 10,675 images, 2,379 positive studies, and 8,296 negative studies under the accepted local raw bundle.
- `scripts/validate_processed_dataset.py` is now the canonical end-to-end validation entrypoint for the trusted processed dataset contract.
- `configs/config.yaml` now points `data.processed_dir` to `data/processed/pneumothorax_trusted_v1`.
- `src/training/metrics.py` now exposes explicit overlap-metric reductions: `micro`, `mean`, `positive_mean`, and `none`.
- The accepted overlap-metric empty-mask policy is now explicit and regression-tested: empty-empty => `1.0`, one-empty-one-positive => `0.0`, and `positive_mean` returns `NaN` when no positive targets exist.
- `tests/test_metrics_reduction.py` is now the canonical regression harness for metric reductions and empty-mask edge cases.
- `src/evaluation/evaluate.py` now routes per-image overlap metrics through the shared backend with explicit `reduction="none"`.
- `tests/test_evaluate_metrics_backend.py` is now the canonical evaluator-side regression harness for per-image metric wiring.

What is still untrusted:
- The existing processed dataset under `data/processed/pneumothorax/`, because it predates the corrected RLE contract and mask-variant separation.
- Historical metrics and plots under `results/`, which remain legacy-only artifacts.
- Validation/model-selection numbers from the current trainer, because the code path has not yet been updated to honor the defined authoritative metric.
- Trainer/evaluator parity, because evaluator now uses the shared backend explicitly but trainer-side validation aggregation is still using the old batch-level accumulation logic.
- Any claim involving Foundation X as clean external pretraining on SIIM.
- The scientific value of the current hybrid design.

Current strategic direction:
- Fix trust issues first, then build a strong pretrained CNN baseline, then decide whether the hybrid is worth redesigning.

Next 3 actions:
1. Rewrite validation metrics once the trusted dataset path exists.
2. Fix positive-only validation counting on top of the corrected metric path.
3. Decide whether to regenerate the split under stratification now that the trusted dataset contract is fixed.
