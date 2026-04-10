# Foundation-nnU-Net Agent Context

Current phase:
- Phase 1 data correctness

Current blocker:
- The repository still lacks a trusted processed dataset and trusted validation metrics, even though the local RLE contract is now resolved.

Highest-priority open tasks:
1. Add golden decode checks that fail if the accepted SIIM contract regresses.
2. Preserve original and dilated masks separately before any trusted dataset regeneration.
3. Audit and lock the DICOM intensity policy.
4. Regenerate the processed dataset with versioned outputs once masks and DICOM policy are settled.
5. Rewrite validation metrics to operate per image and fix positive-only Dice counting once the trusted dataset path is ready.

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

What is still untrusted:
- The existing processed dataset under `data/processed/pneumothorax/`, because it predates the corrected RLE contract and still collapses mask variants.
- Historical metrics and plots under `results/`, which remain legacy-only artifacts.
- Validation/model-selection numbers from the current trainer, because the code path has not yet been updated to honor the defined authoritative metric.
- Any claim involving Foundation X as clean external pretraining on SIIM.
- The scientific value of the current hybrid design.
- The final DICOM intensity policy, which has not yet been audited on representative raw samples.

Current strategic direction:
- Fix trust issues first, then build a strong pretrained CNN baseline, then decide whether the hybrid is worth redesigning.

Next 3 actions:
1. Add golden decode checks on curated positive, negative, and multi-row samples.
2. Define and implement separate original-mask versus dilated-mask outputs.
3. Lock the DICOM intensity policy before regenerating a trusted dataset.
