# Foundation-nnU-Net Decisions

Purpose: persistent record of architectural, methodological, and workflow decisions that affect experiments and claims.

## 2026-04-10 / D-001

Decision:
- Treat all current historical artifacts in `results/` as legacy and non-authoritative until regenerated from the corrected pipeline.

Reason:
- Existing CSVs and plots do not reliably match present code behavior or evaluator output format.

Alternatives considered:
- Continue using legacy artifacts as rough guidance.
- Delete all legacy artifacts immediately.

Impact on experiments / methodology:
- No current metric table, plot, or image under `results/` should be cited as evidence.
- Future authoritative runs must carry provenance metadata.

## 2026-04-10 / D-002

Decision:
- Label correctness and metric correctness are hard blockers for all new experiments.

Reason:
- The audit identified unresolved RLE trust issues and biased validation/model-selection metrics. These defects change scientific conclusions, not just engineering quality.

Alternatives considered:
- Start baseline work in parallel before labels and metrics are fully validated.

Impact on experiments / methodology:
- No architecture comparison or hyperparameter tuning should be treated as meaningful until these blockers are cleared.

## 2026-04-10 / D-003

Decision:
- Preserve original and dilated mask variants separately in the processed dataset.

Reason:
- Training on dilated masks may be useful for sparse targets, but evaluation against official SIIM targets requires access to original masks. Overwriting originals with dilated masks destroys scientific traceability.

Alternatives considered:
- Keep only dilated masks.
- Keep only original masks and abandon dilation experiments.

Impact on experiments / methodology:
- All runs must record which mask variant was used for training, validation, and reporting.
- Final claims must distinguish official-mask performance from any dilated-target training setup.

## 2026-04-11 / D-017

Decision:
- The processed dataset layout now stores mask variants separately:
  - `<processed_dir>/original_masks/`
  - `<processed_dir>/dilated_masks/`
  - `<processed_dir>/mask_variants.json`
- The default run policy is:
  - training mask variant: `dilated_masks`
  - validation/test/final reporting mask variant: `original_masks`
- Dilation is treated as a separate mask variant inside the processed dataset, not as an overwrite of the original labels and not as a wholly separate dataset root.

Reason:
- P0.5 required the original labels to remain available while still supporting sparse-target training experiments.
- Storing both variants under the same processed dataset root keeps image IDs and splits aligned while making variant provenance explicit.
- Defaulting training to the dilated target preserves the current sparse-target research direction without allowing official reporting to silently drift away from the original annotations.

Alternatives considered:
- Keep only one `masks/` directory and switch its meaning between experiments.
- Make dilation a completely separate processed dataset root.
- Use original masks for every stage by default and make dilation purely ad hoc.

Impact on experiments / methodology:
- Any authoritative run must record `train_mask_variant` and `eval_mask_variant`.
- Final official SIIM reporting must use `original_masks` unless a future explicit decision changes that claim boundary.
- Training with `dilated_masks` is allowed as an optimization strategy, but it does not change the evaluation target for official reporting.
- The existing processed dataset remains untrusted until P0.7 regenerates it with the new layout.

## 2026-04-11 / D-018

Decision:
- The accepted local DICOM intensity policy for `data/raw/SIIM-ACR/dicom-images-train` is:
  - read DICOMs through a Windows long-path-safe helper
  - preserve native pixel values when the image is already single-channel and within display range `[0, 255]`
  - apply modality LUT/rescale only if corresponding DICOM metadata exists
  - invert `MONOCHROME1` images if encountered
  - apply VOI LUT/windowing only if corresponding metadata exists
  - fall back to linear min-max scaling into uint8 only when the post-transform pixels are outside `[0, 255]`

Reason:
- Full-corpus metadata audit over all 10,712 local training DICOMs showed:
  - `PhotometricInterpretation=MONOCHROME2` for every file
  - `Modality=CR` for every file
  - `BitsStored=8`, `BitsAllocated=8`, `PixelRepresentation=0`, `SamplesPerPixel=1` for every file
  - `RescaleSlope`, `RescaleIntercept`, `WindowCenter`, `WindowWidth`, and `VOILUTFunction` are absent for every file
- Sample pixel audit showed many images already have maxima below 255, so the previous per-image `pixels / max * 255` step would artificially stretch contrast on a large fraction of the dataset.
- Preview PNGs generated under the new policy looked anatomically plausible and not contrast-inverted.

Alternatives considered:
- Keep the old per-image max normalization.
- Always apply min-max scaling regardless of metadata.
- Hardcode a SIIM-specific 8-bit assumption without future-safe modality/VOI handling.

Impact on experiments / methodology:
- Future preprocessing on this local bundle should preserve native 8-bit CR intensities instead of rescaling every image by its per-image maximum.
- The local preprocessing path is now robust to long Windows file paths, which is necessary for successful dataset regeneration in this workspace.
- If a future bundle introduces `MONOCHROME1`, rescale, or VOI metadata, the same helper path can apply those transforms explicitly instead of silently ignoring them.
- The existing processed dataset remains untrusted until it is regenerated in P0.7 under this policy.

## 2026-04-11 / D-019

Decision:
- The canonical trusted processed dataset version for the current recovery path is `pneumothorax_trusted_v1`.
- Its canonical local root is `data/processed/pneumothorax_trusted_v1`.
- A processed dataset version is authoritative only if it includes:
  - `images/`
  - `original_masks/`
  - `dilated_masks/`
  - `mask_variants.json`
  - `splits.json`
  - `dataset_manifest.json`
- `dataset_manifest.json` is the authoritative source for the dataset fingerprint, split fingerprint, generation parameters, and summary statistics. The directory name alone is not sufficient provenance.

Reason:
- P0.7 regenerated the processed dataset after the accepted RLE contract, mask-variant contract, and DICOM intensity policy were all fixed.
- Using a new versioned root avoids ambiguity with the older untrusted `data/processed/pneumothorax/` directory.
- A dataset-level manifest is required so future runs can record exact dataset provenance rather than assuming a processed directory name implies correctness.

Alternatives considered:
- Overwrite the old unversioned processed dataset in place.
- Keep using an unversioned `data/processed/pneumothorax/` root and rely on chat or notebook notes for provenance.
- Delay dataset-level manifesting until the training pipeline is repaired.

Impact on experiments / methodology:
- Future trusted runs must point to `data/processed/pneumothorax_trusted_v1` unless a later explicit dataset version supersedes it.
- The old processed dataset root remains legacy/untrusted and must not be used for authoritative experiments.
- Dataset fingerprints and split fingerprints must be read from `dataset_manifest.json`, not inferred informally.
- Metric-repair work in Phase 2 should now treat `pneumothorax_trusted_v1` as the fixed data substrate.

## 2026-04-11 / D-020

Decision:
- The canonical overlap-metric reduction modes are:
  - `micro`: aggregate TP/FP/FN across the full evaluated tensor set before computing the metric
  - `mean`: compute the metric per image, then average across all images
  - `positive_mean`: compute the metric per image, then average only across images whose target mask contains foreground
  - `none`: return the per-image metric tensor without reduction
- The explicit empty-mask policy for Dice, IoU, Precision, Recall, and F1 is:
  - pred empty and target empty -> `1.0`
  - pred empty and target positive -> `0.0`
  - pred positive and target empty -> `0.0`
  - `positive_mean` over a subset with zero positive target images -> `NaN`
- `micro` metrics may remain available as auxiliary diagnostics, but authoritative checkpoint selection and official reporting should use per-image reductions, not batch-micro aggregation.

Reason:
- P0.8 required the overlap-metric math to stop depending on implicit batch aggregation.
- The audit already showed that batch-micro Dice in the trainer was biasing model selection.
- Empty-mask edge cases must be explicit because SIIM contains many negative studies, and undefined behavior would silently distort all-image metrics.

Alternatives considered:
- Keep the old implicit micro reduction as the only supported mode.
- Use smoothing constants to avoid explicit empty-mask policy.
- Return `NaN` for every empty-mask overlap case.

Impact on experiments / methodology:
- Phase 2 alignment work must wire trainer and evaluator to these named reductions instead of relying on implicit batch semantics.
- `val_dice_pos_mean` now has an explicit metric backend contract: Dice with `positive_mean`.
- Any legacy metric history computed before this reduction contract was implemented remains non-authoritative.

## 2026-04-10 / D-004

Decision:
- No hybrid optimization work should start before a trusted strong supervised baseline exists.

Reason:
- The current hybrid is both semantically misaligned and methodologically constrained by Foundation X pretraining on SIIM. A strong supervised baseline is required to judge whether the hybrid is worth keeping.

Alternatives considered:
- Salvage the current hybrid immediately.
- Abandon the hybrid permanently now.

Impact on experiments / methodology:
- Recovery effort focuses first on trust recovery and baseline strength.
- Hybrid work must pass an explicit keep/drop gate later.

## 2026-04-10 / D-005

Decision:
- The default strategic path is: trust recovery -> strong pretrained CNN baseline -> hybrid keep/drop decision -> paper-grade methodology.

Reason:
- This ordering maximizes scientific validity and minimizes wasted work on a currently broken hybrid.

Alternatives considered:
- Prioritize novelty before baseline strength.
- Focus only on engineering cleanup without a staged research plan.

Impact on experiments / methodology:
- Task prioritization should always favor trust and baseline milestones over hybrid novelty.

## 2026-04-10 / D-006

Decision:
- Foundation X under the current setup cannot be presented as clean external-pretraining generalization on SIIM.

Reason:
- The audit identified SIIM-ACR exposure inside the Foundation X pretraining corpus, which creates a leakage-sensitive claim boundary.

Alternatives considered:
- Present Foundation X as generic external transfer anyway.
- Remove Foundation X entirely from the project immediately.

Impact on experiments / methodology:
- Any future Foundation X results must be framed as in-domain transfer, ablation, or deferred work unless a non-SIIM-pretrained checkpoint is used.

## 2026-04-10 / D-007

Decision:
- Provenance requirements must not assume Git metadata exists in the working directory; a code fingerprint fallback is mandatory.

Reason:
- The current workspace is not a detected Git repository, but experiment traceability is still required.

Alternatives considered:
- Defer provenance until the project is inside Git.
- Require manual version labels only.

Impact on experiments / methodology:
- Authoritative runs must record either a Git revision or a deterministic code fingerprint over the relevant source/config files.

## 2026-04-10 / D-008

Decision:
- Every authoritative training or evaluation run must persist a minimum metadata record with the following fields:
  - `run_id`
  - `started_at`
  - `model_type`
  - `config_path`
  - `config_hash`
  - `code_revision` if available, otherwise `code_fingerprint`
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

Reason:
- Recovery depends on being able to reconstruct exactly which code, data, target definition, and inference settings produced a result. This is especially important because the workspace is not a detected Git repository.

Alternatives considered:
- Store only a config snapshot and seed.
- Require Git before defining provenance.
- Track provenance manually in notebooks or chat.

Impact on experiments / methodology:
- No future run should be treated as authoritative unless this minimum metadata record exists.
- Training/evaluation provenance must distinguish mask variants and threshold/post-processing choices, not just model weights.
- The implementation may evolve later, but these fields are the non-negotiable minimum contract.

## 2026-04-10 / D-009

Decision:
- Until a repository-level run root is chosen, the relative metadata layout inside any authoritative run directory must be:
  - `<run_dir>/metadata/run_metadata.yaml`
  - `<run_dir>/metadata/config_snapshot.yaml`

Reason:
- Recovery needs a stable on-disk contract for provenance before trainer/evaluator implementation begins.
- Defining the relative layout now avoids ambiguity without prematurely deciding the global output root or the full list of non-metadata artifacts.

Alternatives considered:
- Store all metadata in a single top-level file.
- Delay file naming until trainer implementation starts.
- Define the full run artifact tree now, including metrics, checkpoints, and visuals.

Impact on experiments / methodology:
- Any future authoritative run must have a `metadata/` directory under its run directory.
- `run_metadata.yaml` is the canonical provenance record defined in D-008.
- `config_snapshot.yaml` is the exact resolved config used by that run.
- The global run root location and the rest of the artifact tree remain open and will be decided separately.

## 2026-04-10 / D-010

Decision:
- In addition to the metadata files defined in D-009, every authoritative training run must persist the following minimum output set:
  - `<run_dir>/metrics/history.csv`
  - `<run_dir>/checkpoints/best_checkpoint_metadata.yaml`
  - `<run_dir>/selection/selection_state.yaml`
  - `<run_dir>/qualitative/validation_samples/`

Reason:
- Provenance is not sufficient by itself; a training run must also retain the minimum evidence needed to reconstruct learning dynamics, the chosen checkpoint context, the selected threshold/post-processing state, and qualitative sanity checks.
- This resolves the minimum required outputs without prematurely fixing metric columns or the full evaluation artifact schema.

Alternatives considered:
- Require only metadata and leave all other outputs optional.
- Define the full training and evaluation artifact tree in one step.
- Defer output requirements until trainer implementation.

Impact on experiments / methodology:
- No training run should be treated as authoritative unless these minimum outputs exist alongside the metadata contract.
- `history.csv` is required, but its exact column schema can still be specified later.
- `best_checkpoint_metadata.yaml` is required even before final checkpoint naming policy is fully implemented.
- `selection_state.yaml` is the required record for selected threshold and post-processing state.
- Qualitative validation samples are mandatory evidence, not optional extras.

## 2026-04-11 / D-011

Decision:
- The authoritative primary metric for model selection is `val_dice_pos_mean`, defined as:
  - mean Dice over positive validation images only
  - computed per image, then averaged
  - evaluated on the run's declared evaluation mask variant
  - evaluated using the run's recorded threshold and post-processing state

Reason:
- Overall or micro-aggregated Dice is too easily biased by the large number of negative images in SIIM pneumothorax segmentation.
- The audit already identified batch-level aggregation bias and incorrect positive-only counting in the current trainer.
- A positive-only per-image Dice metric best matches the actual research objective: segment pneumothorax correctly on positive studies.

Alternatives considered:
- Validation loss
- All-image mean Dice
- Batch-level or dataset-level micro Dice
- IoU as the primary checkpoint-selection metric

Impact on experiments / methodology:
- Once Phase 2 metric corrections are implemented, checkpoint ranking and early stopping must use `val_dice_pos_mean`.
- Any current trainer result that was selected using another metric remains non-authoritative.
- The `selection_metric` field in run metadata must be `val_dice_pos_mean` for authoritative runs unless a later explicit decision supersedes it.

## 2026-04-11 / D-012

Decision:
- Legacy artifacts currently under `results/` will remain in place for now and will not be moved or renamed during recovery Phase 0, provided they continue to carry explicit legacy/non-authoritative warnings.

Reason:
- Moving or renaming them before an authoritative output location is defined would add churn without improving scientific trust.
- The real risk is mistaken use as evidence, and that risk is already mitigated by explicit legacy warnings plus recovery-memory policy.

Alternatives considered:
- Move all legacy artifacts into a separate archive directory immediately.
- Rename every legacy file with a `legacy_` prefix.

Impact on experiments / methodology:
- Existing artifacts remain available for historical/debug context only.
- No current file under `results/` becomes authoritative by staying in place.
- A later task will define the separate authoritative output location; until then, `results/` should still be treated as legacy-only.

## 2026-04-11 / D-013

Decision:
- The repository-level canonical location for authoritative experiment runs is `artifacts/runs/`.

Reason:
- Recovery needs a clear separation between legacy artifacts in `results/` and future authoritative outputs.
- A dedicated run root reduces ambiguity without forcing code changes yet.

Alternatives considered:
- Continue using `results/` for both legacy and authoritative outputs.
- Use a different root such as `outputs/` or `experiments/`.
- Delay the location decision until trainer implementation.

Impact on experiments / methodology:
- Future authoritative runs must not be written into `results/`.
- Each authoritative run should create its own run directory under `artifacts/runs/`.
- The relative metadata and required output contracts defined earlier now apply under this repository-level run root.

## 2026-04-11 / D-014

Decision:
- Notebook-generated outputs are non-authoritative by default and may be treated as evidence only when they are traceable to the exact config, checkpoint, and dataset version or fingerprint used to produce them.

Reason:
- Notebooks are convenient for exploration, but they make it too easy to generate untracked figures, tables, or metrics that look credible without being reproducible.
- Recovery requires a bright line between exploratory outputs and evidence suitable for model selection or publication.

Alternatives considered:
- Allow notebook outputs as evidence if the user manually vouches for them.
- Ban notebook use entirely.
- Defer notebook policy until later documentation cleanup.

Impact on experiments / methodology:
- Ad hoc notebook outputs must not be used for model selection, architecture comparison, or paper reporting unless full traceability exists.
- The existence of a notebook file does not make its outputs authoritative.
- This closes the remaining Phase 0 notebook-evidence ambiguity under `P0.1`.

## 2026-04-11 / D-015

Decision:
- For this workspace, the authoritative local SIIM annotation source is `data/raw/SIIM-ACR/train-rle.csv`.
- Stale documentation references to `stage_2_train.csv` and `mask_functions.py` are not source-of-truth for this workspace because those files are not present in the bundled raw data.

Reason:
- The actual raw bundle available in the repository contains `train-rle.csv` and does not contain the originally documented `stage_2_train.csv` or `mask_functions.py`.
- Recovery must ground label validation on the files that actually exist in the workspace, not on stale development-guide assumptions.

Alternatives considered:
- Continue treating the development guide as authoritative even when its referenced raw files are absent.
- Delay the annotation-source decision until the full decoder contract is resolved.

Impact on experiments / methodology:
- All decoder validation work must be grounded in `data/raw/SIIM-ACR/train-rle.csv`.
- The missing `mask_functions.py` means the decoder contract is still unresolved; this decision identifies the authoritative local source but does not by itself validate decode semantics.
- Future documentation cleanup should remove or correct stale references to `stage_2_train.csv` and `mask_functions.py`.

## 2026-04-11 / D-016

Decision:
- For the local `data/raw/SIIM-ACR/train-rle.csv` bundle, the accepted decoder contract is `cumulative_gap_pairs`, not standard `absolute_pairs`.
- Contract details:
  - `-1` denotes an empty mask.
  - The first positive token is the absolute 1-based start of the first run.
  - Each subsequent positive token is the zero-count gap after the previous run.
  - Length tokens remain run lengths.
  - Flat decoding uses Fortran order before reshaping to `(height, width)`.
- The preprocessing entrypoint may accept explicit `absolute_pairs` for compatibility, but the default workflow must validate the corpus and resolve a single compatible mode before decoding.

Reason:
- Corpus-level evidence from the shipped `train-rle.csv` shows:
  - `positive_rows=3286`
  - `negative_rows=8296`
  - `valid_absolute_pairs=0`
  - `valid_cumulative_gap_pairs=3286`
- Curated positive samples decode into monotonic non-overlapping runs only under the cumulative-gap interpretation.
- Negative rows are consistently encoded as `-1`.

Alternatives considered:
- Keep treating the shipped CSV as standard absolute start/length pairs.
- Hardcode a single local decoder mode with no validation step.
- Delay the decoder decision until after mask regeneration.

Impact on experiments / methodology:
- Any masks generated by the old absolute-pair decoder remain untrusted and must be regenerated in P0.7.
- `src/data/preprocess.py` now resolves `--rle_mode auto` against the corpus before decoding.
- Explicitly requesting `absolute_pairs` on the shipped local CSV should fail fast instead of silently producing wrong masks.
- P0.4 will add golden regression checks for this accepted contract.

## 2026-04-11 / D-021

Decision:
- The stratification target for publication-facing train/val/test splits is the binary image-level label:
  - `positive` if the image's accepted `original_masks` target contains any foreground pixel
  - `negative` otherwise
- Stratification must be defined against the official/original target, not `dilated_masks`.
- Target split proportions remain `70 / 15 / 15`.
- A regenerated stratified split should preserve the global positive ratio in each split as closely as integer constraints allow, with a validation target of at most `1.0` absolute percentage point deviation from the dataset-wide positive ratio for each split.

Reason:
- P1.1 requires a concrete stratification objective before deciding whether to preserve the current image IDs or regenerate the split.
- Using `original_masks` avoids letting the training-only dilation policy influence publication-facing split semantics.
- The current unstratified split already has only modest drift, so the target should be "match the global image-level class ratio closely" rather than inventing a more complex balancing rule.

Alternatives considered:
- Stratify using `dilated_masks`.
- Stratify on pixel-level foreground fraction instead of image-level positive/negative.
- Accept the existing unstratified split as-is for publication use.

Impact on experiments / methodology:
- Any future stratified split regeneration must derive labels from `original_masks` or the equivalent accepted raw-label definition.
- Validation of the stratified split should check class-ratio deviation against the dataset-wide positive ratio, not just overlap-free IDs.
- This decision defines the target class balance only; it does not yet decide whether the current split IDs are preserved or regenerated.

## 2026-04-11 / D-022

Decision:
- The current trusted split IDs under `data/processed/pneumothorax_trusted_v1/splits.json` will **not** be preserved for publication-facing experiments.
- The split must be regenerated under the accepted stratified policy.

Reason:
- Measured image-level positive ratios for the current trusted split are:
  - train: `22.6610%` (`1693 / 7471`)
  - val: `21.2235%` (`340 / 1602`)
  - test: `21.5980%` (`346 / 1602`)
- The dataset-wide positive ratio is `22.2857%` (`2379 / 10675`).
- Under D-021, the target is to keep each split within `1.0` absolute percentage point of the dataset-wide ratio when feasible.
- The current validation split deviates by about `1.0622` percentage points, so the current split narrowly misses the accepted publication-facing target.
- No authoritative baseline results depend on keeping the existing split IDs, so regeneration has lower methodological risk than preserving a slightly off-policy split for convenience.

Alternatives considered:
- Preserve the current split IDs because the drift is small.
- Preserve only the training IDs and regenerate validation/test.

Impact on experiments / methodology:
- The current split remains usable for legacy/debug context inside the trusted dataset version, but it should not be the publication-facing split once P1.1 is fully implemented.
- The next split task should implement deterministic regeneration under the accepted stratified policy and then update the split fingerprint.
- Strong baseline work should prefer the regenerated stratified split once it exists.

## 2026-04-11 / D-023

Decision:
- The publication-facing stratified split policy is:
  - deterministic two-stage `train_test_split` with `random_state=42`
  - stage 1: split all image IDs into `train_val` and `test` with `test_size=0.15`
  - stage 2: split `train_val` into `train` and `val` with `test_size=0.17647058823529413`
  - both stages use `stratify=` on the binary image-level labels defined in D-021
  - each final split is stored in sorted image-ID order for stable manifests and diffs
- The accepted publication-facing split seed remains `42`.

Reason:
- P1.1 needed the exact deterministic split mechanism pinned down before regenerating `splits.json`.
- Keeping seed `42` preserves continuity with the existing recovery dataset while changing only the split policy itself.
- A two-stage stratified split reproduces the intended `70 / 15 / 15` proportions with standard tooling and keeps the logic easy to audit.

Alternatives considered:
- Change the seed while introducing stratification.
- Use one custom splitter implementation instead of two standard `train_test_split` calls.
- Use unsorted output order and rely on downstream code not to care.

Impact on experiments / methodology:
- The next split-regeneration implementation must follow this exact two-stage seeded policy.
- Any future split fingerprint change under `pneumothorax_trusted_v1` should be attributable to this policy update rather than an arbitrary seed change.
- Validation should confirm both the seed and the staged stratified procedure match this decision.

## 2026-04-11 / D-024

Decision:
- The accepted immediate trainer config surface for authoritative baseline work is:
  - `loss.type`: `dice_focal` only
  - `training.optimizer`: `AdamW` or `Adam`
  - `training.scheduler`: `ReduceLROnPlateau` or `none`
- Unsupported values must fail fast during trainer startup instead of silently falling back to hardcoded defaults.
- Resume checkpoints used for authoritative runs must carry canonical `training_components` metadata, and legacy `checkpoints/last_*.pth` files without that metadata must be rejected.

Reason:
- P1.5 required a small, trusted, immediately usable configuration surface before ablation sweeps or baseline training can be considered auditable.
- `DiceFocalLoss` is the only currently implemented and trusted loss in the repository, so expanding loss support now would widen scope without corresponding validation.
- `Adam` is the minimal additional optimizer needed for near-term controlled ablations relative to `AdamW` without introducing new hyperparameter fields.
- `ReduceLROnPlateau` remains aligned with the corrected positive-only checkpoint-selection metric, while `none` is the only additional scheduler mode needed immediately for controlled comparisons.
- Legacy resume checkpoints without explicit component metadata would allow old hardcoded behavior to leak into new config-driven runs.

Alternatives considered:
- Keep silently hardcoding `DiceFocalLoss`, `AdamW`, and `ReduceLROnPlateau`.
- Expand the surface immediately to additional losses, `SGD`, or cosine/one-cycle schedulers before those options are separately validated.
- Allow legacy `last_*.pth` checkpoints to resume without component metadata and trust the operator to notice mismatches manually.

Impact on experiments / methodology:
- Near-term trainer-side ablations may vary optimizer and scheduler only within this accepted surface; anything beyond it remains out of bounds until a later explicit decision expands support.
- `loss.type` is now part of the authoritative config contract even though the immediate accepted surface contains only one trusted option.
- Changing `training.optimizer` or `training.scheduler` inside the accepted surface now changes trainer behavior explicitly and audibly instead of being ignored.
- Pre-fix resume checkpoints without `training_components` metadata are non-authoritative for continued config-driven runs and should be discarded rather than resumed.

## 2026-04-11 / D-025

Decision:
- The accepted immediate validation-only threshold-selection policy is:
  - selection split: `val` only
  - optimized metric: `val_dice_pos_mean`
  - threshold candidate grid: `0.05` to `0.95` inclusive in `0.05` steps
  - post-processing search space: `none` only for now
- Threshold selection must fail fast if invoked on `train` or `test`.
- If multiple thresholds tie on the optimized metric, the deterministic tie-break is:
  - prefer the threshold closest to the legacy default `0.5`
  - if still tied, prefer the smaller threshold

Reason:
- P1.4 needed a concrete, reproducible threshold-selection policy before strong-baseline runs can be compared fairly.
- D-011 already fixed the authoritative selection metric as positive-only per-image Dice, so threshold tuning should optimize the same corrected metric rather than invent a second objective.
- A coarse-but-auditable `0.05` grid is sufficient for the immediate blocker and keeps the search space explicit in config and tests.
- Post-processing remains scientifically open, so the immediate safe policy is to pin it to `none` instead of silently mixing in untracked contour or min-area heuristics.
- Failing fast on non-validation splits closes the most obvious test-leakage path while storage/reuse wiring is still pending.
- Preferring `0.5` on exact ties minimizes gratuitous drift from the historical inference threshold.

Alternatives considered:
- Keep using a fixed hardcoded threshold of `0.5` with no validation selection.
- Tune thresholds on all-image mean Dice or IoU instead of `val_dice_pos_mean`.
- Include contour filtering or minimum-area pruning immediately before their provenance path is defined.
- Allow threshold search on `test` and rely on manual discipline to avoid leakage.

Impact on experiments / methodology:
- Near-term threshold sweeps are now constrained to a documented validation-only policy instead of ad hoc manual probing.
- Any selected threshold outside the declared candidate grid is non-authoritative unless a later decision expands the search space.
- Any tuned run using post-processing other than `none` remains out of bounds until that search surface is explicitly approved and recorded.
- Authoritative storage and test-time reuse of the selected threshold are handled separately in D-026.

## 2026-04-11 / D-026

Decision:
- The authoritative persisted threshold-selection artifact is `<run_dir>/selection/selection_state.yaml`.
- `selection_state.yaml` must record at least:
  - `selection_split`
  - `selection_metric`
  - `selected_threshold`
  - `selected_postprocess`
  - `threshold_candidates`
  - `threshold_summary`
  - `model_type`
  - `checkpoint_path`
  - `dataset_root`
  - `eval_mask_variant`
  - `input_size`
- Authoritative test evaluation must require `selection_state.yaml` input instead of silently defaulting to `0.5`.
- Before reusing a saved threshold on test, evaluation must fail fast unless the saved selection state matches the current:
  - `model_type`
  - `checkpoint_path`
  - `dataset_root`
  - `eval_mask_variant`
  - `input_size`

Reason:
- P1.4 required the chosen threshold to be persisted and replayed on test without ambiguity.
- A saved threshold is only scientifically meaningful if it is tied to the exact model checkpoint and evaluation context that produced it on validation.
- Requiring the canonical `selection/selection_state.yaml` path aligns implementation with the already accepted run-artifact contract in D-010.
- Failing fast on mismatched reuse closes a provenance hole where the wrong threshold file could otherwise be applied to a different checkpoint or dataset silently.

Alternatives considered:
- Keep the selected threshold only in memory or chat history.
- Store only the scalar threshold without the surrounding evaluation context.
- Allow test evaluation to fall back to `0.5` when no saved selection state is supplied.
- Allow arbitrary filenames and directories for the persisted threshold artifact.

Impact on experiments / methodology:
- Test metrics are no longer authoritative unless they are produced with an explicit, matching `selection_state.yaml`.
- Validation threshold selection is now an auditable artifact rather than an informal parameter choice.
- Any selection-state file whose context does not match the current test evaluation must be treated as invalid evidence rather than "close enough."
- P1.4 is now complete for the immediate `postprocess=none` path; broader post-processing search remains a later decision.

## 2026-04-11 / D-027

Decision:
- The single primary pretrained baseline family for immediate implementation is `ImageNet-pretrained ResNet34 encoder U-Net`.
- The immediate implementation path should stay inside the current repository stack by using the already-present encoder/runtime dependencies (`torchvision` and/or `timm`) rather than introducing a new segmentation-framework dependency just to obtain the baseline.

Reason:
- P1.6 required one strong supervised baseline family to be fixed before protocol design or implementation could continue.
- `ResNet34` is a standard, publication-legible pretrained encoder upgrade over the current plain U-Net without changing the overall encoder-decoder framing of the project.
- Compared with deeper or more novelty-heavy alternatives, `ResNet34` offers a favorable balance of:
  - lower integration risk
  - lower compute/memory pressure at the current `512` input size
  - straightforward multiscale feature hierarchy for a U-Net-style decoder
  - clean methodological separation from the Foundation X hybrid path
- The repository already depends on `torchvision` and `timm`, and `timm` is already used in `src/models/backbone.py`, so this family fits the current stack better than adding an extra external segmentation package as a new hard dependency.

Alternatives considered:
- `ImageNet-pretrained ResNet50 encoder U-Net`: stronger but higher compute cost and wider scope for the first corrected baseline.
- `ImageNet-pretrained EfficientNet encoder U-Net`: reasonable, but a less straightforward immediate drop-in relative to the current U-Net-style encoder stages.
- `ImageNet-pretrained DenseNet encoder U-Net`: chest-X-ray-relevant on the classification side, but less natural as the first low-risk segmentation anchor here.
- Introducing `segmentation_models_pytorch` immediately: would speed implementation, but adds a new dependency surface when the current recovery path is trying to keep baseline changes narrow and auditable.

Impact on experiments / methodology:
- The paper-path supervised anchor is now fixed to a pretrained `ResNet34` encoder family unless a later explicit decision supersedes it.
- The next baseline task should define a fair training protocol around this chosen family rather than reopening encoder selection.
- Hybrid keep/drop work should compare against this baseline family, not against the current random-init plain U-Net.
- This decision selects the family only; exact implementation details such as grayscale adaptation, decoder wiring, and configuration surface remain separate follow-up tasks.

## 2026-04-11 / D-028

Decision:
- The immediate fair-comparison protocol for the first strong supervised baseline is fixed as a one-variable comparison between:
  - the corrected current plain U-Net baseline
  - the future `ImageNet-pretrained ResNet34 encoder U-Net`
- For that first authoritative comparison, both runs must keep the following data and evaluation substrate identical:
  - trusted dataset root: `data/processed/pneumothorax_trusted_v1`
  - the current trusted split manifest/fingerprint under that dataset version
  - `data.input_size=512`
  - `data.train_mask_variant=dilated_masks`
  - `data.eval_mask_variant=original_masks`
  - the existing training augmentation path from `src.data.augmentations.get_train_transforms()`
  - the existing weighted train sampler policy used by `src/training/trainer.py`
- For that first authoritative comparison, both runs must keep the following optimization and model-selection settings identical:
  - `loss.type=dice_focal`
  - `training.optimizer=AdamW`
  - `training.learning_rate=0.0001`
  - `training.weight_decay=0.01`
  - `training.scheduler=ReduceLROnPlateau`
  - `training.batch_size=8`
  - `training.epochs=150`
  - `training.early_stopping_patience=30`
  - `seed=42`
  - checkpoint ranking by `val_dice_pos_mean`
  - validation-only threshold selection and reuse under D-025 and D-026
  - `selection.postprocess=none`
- The first authoritative comparison may change only the segmentation architecture and initialization path. It must not introduce, for only one arm:
  - ROI/crop preprocessing
  - alternate dataset roots or mask semantics
  - test-time augmentation
  - post-processing beyond `none`
  - model-specific optimizer/loss/scheduler retuning
  - staged encoder freezing/unfreezing
  - any hybrid/Foundation X component
- The pretrained baseline should be fine-tuned end-to-end from the start in the immediate protocol; if grayscale adaptation is required for the pretrained encoder, that adaptation must happen inside the model path rather than by creating a second RGB dataset pipeline.

Reason:
- P1.6 required the fair training protocol to be fixed in repo memory before implementation begins.
- The repository now has a trusted dataset, corrected per-image metrics, corrected checkpoint-selection logic, and a defined validation-only threshold path, so the main remaining fairness risk is changing multiple knobs at once when the pretrained baseline is introduced.
- Fixing the first comparison as a one-variable architectural change keeps the baseline result publication-legible and prevents scope creep into crop strategy, post-processing, or model-specific retuning before the pretrained path even exists.

Alternatives considered:
- Tune optimizer, scheduler, augmentation, or batch policy separately for the pretrained model immediately.
- Freeze the pretrained encoder initially, or use a staged unfreeze schedule in the first comparison.
- Create a separate RGB-preprocessed dataset branch for the pretrained encoder.
- Fold ROI/crop or post-processing changes into the first pretrained baseline run.

Impact on experiments / methodology:
- The next implementation task should add the `ResNet34` encoder U-Net path without reopening the rest of the corrected baseline protocol.
- The first corrected comparison between the plain U-Net and pretrained baseline is now explicitly a single-variable architecture/initialization comparison.
- Any initial pretrained-baseline result that changes additional knobs relative to this fixed protocol is non-authoritative for the paper-path baseline gate and should not be used for hybrid keep/drop decisions.

## 2026-04-11 / D-029

Decision:
- The first authoritative `ImageNet-pretrained ResNet34 encoder U-Net` baseline run is baseline-gate eligible only if, in addition to the minimum training outputs already required by D-010, its run directory also contains the following evidence package:
  - tuned validation-threshold artifact at `<run_dir>/selection/selection_state.yaml`
  - machine-readable held-out test report at `<run_dir>/reports/test_metrics.csv`
  - aggregated held-out test summary at `<run_dir>/reports/test_summary.yaml`
  - qualitative validation package at `<run_dir>/qualitative/validation_samples/`
  - qualitative test package at `<run_dir>/qualitative/test_samples/`
- The baseline-gate test report package must be generated from the same best checkpoint and reused `selection_state.yaml` context that the run declares as authoritative.
- The exact trainer/evaluator output schema and final column naming remain a separate P1.2 task, but the baseline-gate test report package must already be sufficient to recover:
  - real image IDs
  - split identity
  - eval mask variant
  - selected threshold and selected post-processing context
  - corrected per-image test metrics
- Each qualitative package must include a manifest recording which image IDs were selected into the package so the qualitative evidence is auditable and not ad hoc. The qualitative package is required for both validation and test splits; screenshots copied out of notebooks or chat are not substitutes.

Reason:
- P1.6 required the output package for the first strong pretrained baseline to be fixed before implementation starts.
- D-010 already defined a minimum training-run output floor, but it did not yet pin the held-out test evidence package needed for the actual baseline gate.
- The repository already has a canonical validation-threshold artifact path under D-026, so the missing piece was defining the authoritative test report and qualitative evidence package that must accompany the first pretrained baseline claim.
- Requiring manifests for qualitative samples closes a cherry-picking hole while still leaving the exact sampling strategy open for implementation.

Alternatives considered:
- Treat the D-010 minimum training outputs as sufficient and leave test reporting/qualitative evidence informal.
- Require only a scalar test summary with no machine-readable per-image report.
- Accept notebook screenshots or ad hoc exported images as qualitative evidence.
- Fully lock the final CSV column schema now, even though P1.2 is the dedicated output-schema task.

Impact on experiments / methodology:
- The next implementation task must write the first pretrained baseline's evidence under `artifacts/runs/<run_id>/` as a complete package, not just a checkpoint plus console logs.
- A pretrained baseline result without `reports/test_metrics.csv`, `reports/test_summary.yaml`, and auditable validation/test qualitative packages is non-authoritative for the baseline gate even if the training loop itself ran successfully.
- P1.2 remains responsible for harmonizing exact trainer/evaluator output schema and naming, but it should refine this package rather than reopen whether the package itself is required.

## 2026-04-12 / D-030

Decision:
- The `P1.7` ROI/crop gate is now defined against the first authoritative full-image pretrained baseline result, using the held-out `test` positive-only Dice mean from the baseline-gate evidence package as the decision metric for whether crop/ROI work becomes mandatory on the critical path.
- Crop/ROI work becomes mandatory if the best currently trusted full-image supervised baseline reports `test` positive-only Dice mean below `0.60`.
- Under the current authoritative baseline result, the gate is triggered: the trusted full-image pretrained baseline reported positive-only Dice mean `0.4951`, so a justified crop/ROI comparison is now required before hybrid work or publication-path baseline framing can continue.

Reason:
- `P1.7` exists because image-level balancing does not by itself solve the extreme pixel sparsity in SIIM pneumothorax segmentation.
- The trusted dataset manifest now confirms that foreground occupies only about `0.3049%` of all pixels overall (`foreground_fraction_all_pixels=0.0030487728788925277`) and that even positive images have a median of only `2125` foreground pixels at `512 x 512`, so a full-image baseline that remains below a clear positive-case overlap bar should not be treated as evidence that crop/ROI is unnecessary.
- Using the held-out `test` positive-only Dice mean keeps the gate tied to the same corrected, publication-facing metric family already adopted for trustworthy segmentation assessment, while avoiding all-image averages that can be biased by negatives.
- The threshold is fixed at `0.60` so the project does not treat a roughly half-overlap positive-case baseline as strong enough to rule out crop/ROI under this sparsity regime.

Alternatives considered:
- Leave the crop/ROI decision informal and decide ad hoc after each run.
- Gate on validation metrics instead of the held-out baseline evidence package.
- Gate on all-image Dice or IoU instead of positive-only Dice.
- Set a looser threshold near the current result and defer crop/ROI despite the observed sparsity pressure.

Impact on experiments / methodology:
- `P1.7` is now active on the critical path: the threshold-definition subtask is complete, and the next task is a controlled crop/ROI comparison against the current trusted full-image baseline.
- Hybrid work remains paused because the current trusted full-image baseline did not clear the crop/ROI gate.
- Any crop/ROI comparison must keep the same trusted dataset root, split, corrected metric path, and non-crop protocol settings unless a later explicit decision changes that scope.

## 2026-04-12 / D-031

Decision:
- The immediate `P1.7` crop/ROI comparison arm is fixed as a train-only ROI-crop policy layered onto the current trusted full-image pretrained baseline protocol; validation, threshold selection, and held-out test evaluation remain full-image.
- The accepted immediate crop policy is:
  - operate on the existing `512 x 512` trusted processed images; do not create a second processed dataset root
  - for positive `train` images, derive a square ROI crop from the current training mask variant, using the mask bounding box as the anchor, expanding/clamping it to a `384 x 384` window with limited center jitter
  - for negative `train` images, sample a random `384 x 384` crop from the same `512 x 512` image space
  - resize every crop back to `512 x 512` before it enters the existing model stack so the architecture, batch surface, and evaluation tensor size stay otherwise unchanged
  - keep `val` and `test` uncropped full images in the immediate comparison arm
- The accepted immediate leakage constraints are:
  - ground-truth masks may guide crop placement on the `train` split only
  - `val` and `test` crop placement must not depend on ground-truth masks, test-time threshold state, or any separate ROI detector in this immediate comparison
  - any future image-only ROI detector, lung crop heuristic, sliding-window inference, or test-time crop ensemble is out of scope for the immediate `P1.7` comparison and would require a later explicit decision

Reason:
- The current blocker is to test whether moderate train-time zoom into pneumothorax regions helps under the observed sparsity regime without reopening multiple variables at once.
- Keeping validation and test evaluation full-image preserves direct comparability with the trusted baseline result and avoids label leakage on evaluation splits.
- A `384 x 384` crop from the `512 x 512` processed image provides a conservative zoom-in rather than an aggressive patch regime, which keeps more thoracic context while still enlarging sparse targets.
- Resizing crops back to `512 x 512` keeps the existing model path, batch surface, checkpoint/evaluator contracts, and corrected artifact pipeline intact for the first crop comparison.
- Using the current training mask variant to anchor positive train crops avoids introducing a second target-definition branch inside the crop comparison arm.

Alternatives considered:
- Apply mask-guided crops on validation or test, which would leak target information.
- Introduce a second cropped processed dataset root before proving the crop idea is worthwhile.
- Use much smaller patch sizes such as `256 x 256`, which would widen the comparison into a more aggressive patch-training experiment.
- Add a learned/image-only ROI detector or sliding-window inference immediately, which would introduce extra moving parts beyond the immediate crop question.

Impact on experiments / methodology:
- The next `P1.7` implementation task is now decision-complete: add the fixed train-only `384 x 384` ROI-crop arm, run it on GPU, and compare it against the trusted full-image baseline under the same corrected protocol.
- The immediate crop comparison must differ from the trusted full-image baseline only by this train-time crop policy; optimizer, scheduler, threshold-selection policy, mask-variant contract, and evaluation path stay fixed.
- Any crop result that changes evaluation to label-guided ROI selection or adds a second ROI mechanism is non-authoritative for the immediate `P1.7` decision.

## 2026-04-15 / D-032

Decision:
- The immediate D-031 train-only ROI/crop comparison arm does not replace the trusted full-image pretrained baseline as the current paper-path supervised anchor.
- For the current recovery path, the trusted full-image pretrained baseline remains the primary supervised reference because the first authoritative D-031 crop run underperformed it on the held-out `test` positive-only Dice mean:
  - full-image pretrained baseline: `0.4951`
  - immediate D-031 crop comparison: `0.4625`
- Immediate ROI/crop work is therefore resolved for the current critical path: the tested D-031 arm did not justify replacing the full-image baseline, and any further ROI/crop exploration would require a new explicit decision rather than continuing as the default next step.

Reason:
- `P1.7` required evidence either that a justified crop policy should be retained or that full-image training should remain the main path with evidence.
- The authoritative GPU/Colab crop run under `/content/drive/MyDrive/foundation_nnunet_runs/resnet34_roi_crop_authoritative_v1` completed the required comparison under the same trusted dataset, corrected metric path, and authoritative artifact protocol as the full-image baseline.
- Although the crop arm was competitive enough to be worth checking, it did not beat the held-out full-image result and also showed the same late empty-mask collapse pattern seen in the earlier run family, so there is no evidence-based basis to promote it over the current full-image anchor.

Alternatives considered:
- Keep ROI/crop active on the critical path despite the first authoritative crop run underperforming.
- Treat the crop result as inconclusive and continue iterating ROI/crop policy before deciding whether to move on.
- Replace the full-image baseline with the crop run anyway because it was “close enough.”

Impact on experiments / methodology:
- `P1.7` is now complete: the current paper-path supervised anchor remains the trusted full-image pretrained baseline.
- The next critical-path decision moves to `P1.8`, namely whether the current hybrid is worth further investment relative to that trusted full-image baseline.
- Any future ROI/crop work is now off the default critical path unless a later explicit decision reopens it with a new justification and comparison scope.

## 2026-04-15 / D-033

Decision:
- The default `P1.8` decision state for the current hybrid is now `defer`, not `keep`.
- The hybrid may return to the active critical path only if a future authoritative hybrid run clears both sides of a keep/drop gate relative to the trusted full-image pretrained baseline:
  - engineering-integrity side: the active hybrid path must first clear the already-open hybrid repair tasks needed to make its behavior interpretable (`P1.9` gradient-flow semantics, `P1.10` fusion-scale alignment, and `P1.11` branch-normalization policy), or an explicitly recorded equivalent proof set
  - performance side: under the same trusted dataset root, corrected metric path, validation-only threshold-selection discipline, and authoritative run-artifact package used by the current baseline, the hybrid must beat the trusted full-image pretrained baseline on held-out `test` positive-only Dice mean by at least `+0.02` absolute
- Given the current trusted full-image baseline result `0.4951`, the immediate keep threshold for a future hybrid candidate is therefore `>= 0.5151` held-out `test` positive-only Dice mean.
- If a future hybrid candidate fails either side of this gate, the hybrid remains deferred from the main paper path and should not outrank the remaining cleanup tasks (`P1.2`, `P1.3`, `P1.12`).

Reason:
- The repository now has a trusted supervised anchor and an already-tested crop branch, so hybrid work no longer needs to proceed on hope alone.
- The current hybrid is not merely another architecture variant; it carries extra methodological risk because Foundation X pretraining is SIIM-exposed and the code path still has unresolved technical issues called out in `P1.9` through `P1.11`.
- On the current single-split evidence base, a tie or tiny gain over the baseline is not enough to justify reopening the highest-risk branch of the project; the hybrid needs a clear win, not parity.
- Anchoring the gate to the trusted full-image baseline prevents the weaker immediate crop arm from becoming the comparison target by accident.

Alternatives considered:
- Keep the hybrid on the critical path even before a concrete outperformance bar exists.
- Require only a strict `> 0.4951` improvement with no margin.
- Drop the hybrid permanently right now before a formal gate is recorded.

Impact on experiments / methodology:
- The first `P1.8` subtask is now decision-complete: the hybrid is deferred-by-default until it clears a recorded engineering and performance gate.
- `P1.9`, `P1.10`, and `P1.11` remain conditional follow-up tasks rather than automatic next steps.
- The remaining `P1.8` work is now narrower: record the exact evidence checklist and the Foundation X paper-framing constraints needed before any future hybrid reopening.

## 2026-04-15 / D-034

Decision:
- Continued hybrid work is justified only if a future hybrid candidate ships a minimum reopening evidence package, not just a scalar metric or notebook screenshot.
- The minimum D-034 evidence package for any future hybrid reopening is:
  - one authoritative hybrid run directory under the same trusted evaluation regime used by the current full-image baseline, carrying the same minimum artifact family already required for baseline-gate work:
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
  - an explicit comparison record back to the trusted full-image pretrained baseline run, including the exact held-out `test` positive-only Dice delta relative to `0.4951` and whether the candidate cleared the D-033 keep threshold `>= 0.5151`
  - an explicit engineering-integrity proof set showing:
    - backbone gradient behavior is correct for the intended frozen/unfrozen mode
    - fusion-stage shapes are asserted and documented at the active input size
    - hybrid branch-normalization policy is recorded in config and run metadata rather than left implicit
- Console logs, copied notebook cells, or a single manually reported metric do not count as sufficient evidence for reopening hybrid work.
- A future hybrid candidate that numerically beats `0.5151` without the rest of this evidence package still does not clear D-033.

Reason:
- The hybrid branch combines higher engineering risk with a tighter methodological claim boundary than the trusted supervised baseline, so its proof burden must be higher than "one promising score."
- The recovery path has already shown that unaudited metrics, copied logs, and non-authoritative artifacts are not enough to support project-level decisions.
- Reusing the baseline-gate artifact family keeps any future hybrid comparison auditable and directly comparable to the already trusted baseline evidence.

Alternatives considered:
- Allow any future held-out metric win to reopen hybrid work immediately.
- Require only engineering proofs and leave held-out comparison artifacts informal.
- Require only a side-by-side metric table and skip implementation-integrity evidence.

Impact on experiments / methodology:
- The second `P1.8` subtask is now decision-complete: hybrid reopening requires an auditable evidence package, not just a promising run.
- Any future hybrid candidate must prove both interpretability of the implementation and baseline-relative value before it can compete for critical-path time.
- The remaining `P1.8` work is now the Foundation X paper-framing boundary, not whether scalar evidence alone is enough.

## 2026-04-15 / D-035

Decision:
- Under the current checkpoint provenance, any future Foundation X hybrid result may be framed only as leakage-aware in-domain transfer or ablation work, not as clean external pretraining or target-unseen generalization on SIIM.
- Unless a later run uses a verified non-SIIM-exposed Foundation X checkpoint, the allowed paper/path framing is limited to:
  - secondary ablation against the trusted full-image pretrained baseline
  - in-domain transfer analysis using a SIIM-exposed initialization source
  - engineering evidence about whether Foundation X features add value despite the exposure caveat
- Under the current setup, the following claim classes are forbidden:
  - describing Foundation X as clean external pretraining for SIIM
  - presenting a Foundation X hybrid result as evidence of target-unseen transfer or cross-dataset generalization into SIIM
  - implying that any hybrid gain isolates broader foundation-model knowledge rather than a SIIM-exposed initialization advantage
  - replacing the trusted full-image pretrained baseline as the default headline paper anchor solely because a SIIM-exposed Foundation X hybrid performs better on the current split
- Even if a future hybrid candidate clears D-033 and D-034, it remains a leakage-aware secondary comparison under the current checkpoint provenance unless a later explicit methodology decision broadens that claim boundary.

Reason:
- D-006 already established that the Foundation X pretraining corpus includes SIIM exposure, which means the main scientific risk is claim inflation rather than raw metric reporting alone.
- D-033 and D-034 now define when hybrid work is technically worth reopening; this separate decision is needed so a future reopening does not silently expand the paper claim boundary beyond what the data provenance supports.
- Keeping the trusted full-image supervised baseline as the headline anchor preserves a clean comparison point even if a SIIM-exposed Foundation X hybrid later becomes competitive.

Alternatives considered:
- Treat any future hybrid win as sufficient to promote Foundation X to the main paper contribution despite the exposure caveat.
- Defer all Foundation X framing decisions entirely until the final paper-writing stage.
- Remove Foundation X from all future discussion immediately, even as a leakage-aware ablation.

Impact on experiments / methodology:
- `P1.8` is now decision-complete: the hybrid is deferred by default, requires D-033 and D-034 to reopen, and remains leakage-aware secondary evidence under the current checkpoint provenance.
- `P1.12` is now narrower: it should formalize the final leak-aware methodology around an already-fixed claim boundary, not reopen whether clean external-transfer claims are allowed.
- Any future document, notebook, or result summary that describes Foundation X outside this framing must be treated as methodologically non-authoritative.

## 2026-04-15 / D-036

Decision:
- The first exact authoritative CSV schema contract under `P1.2` is now fixed for trainer history and evaluator per-image reports.
- The mandatory canonical `history.csv` columns, in order, are:
  - `epoch`
  - `train_loss`
  - `val_loss`
  - `val_dice_mean`
  - `val_dice_pos_mean`
  - `val_iou_mean`
- Legacy in-memory trainer keys remain upgrade-only aliases for resume compatibility:
  - `val_dice` -> `val_dice_mean`
  - `val_dice_pos` -> `val_dice_pos_mean`
  - `val_iou` -> `val_iou_mean`
- The mandatory canonical `reports/test_metrics.csv` columns, in order, are:
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
  - `hausdorff`
  - `precision`
  - `recall`
  - `f1`
- Future optional columns may be appended after these canonical columns, but authoritative CSV writers must preserve this mandatory ordered prefix.

Reason:
- `P1.2` exists because the current trainer history and evaluator reports are easy to misread when column names drift away from the corrected metric definitions.
- The trainer had already moved to per-image mean aggregation, but its saved history still used ambiguous names like `val_dice` and `val_iou`, which no longer made the reduction semantics obvious.
- D-029 intentionally deferred exact schema locking until a dedicated `P1.2` pass; that dedicated pass now needs a stable ordered contract so future reports are easier to audit across runs and code paths.

Alternatives considered:
- Leave the current ambiguous history naming in place and document the meaning informally.
- Fully defer exact CSV naming until every later `P1.2` subtask is also complete.
- Lock an even larger schema now, including subset-tag additions that remain a separate open `P1.2` task.

Impact on experiments / methodology:
- Future authoritative `history.csv` outputs must use explicit `_mean` suffixes for corrected validation reductions rather than the older ambiguous metric names.
- Future authoritative `reports/test_metrics.csv` outputs now have a fixed ordered required prefix, making cross-run parsing and review easier without blocking later additive fields.
- Existing older checkpoints may still resume through the legacy in-memory aliases, but new authoritative CSVs should emit only the D-036 canonical names.
- `P1.2` remains open for the remaining output-schema subtasks, especially explicit subset-tag handling and final metadata completeness checks.

## 2026-04-15 / D-037

Decision:
- Evaluation outputs that enumerate individual evaluated images must now carry both:
  - the exact dataset `image_id` used for that row or qualitative sample
  - an explicit `subset_tag` describing positive/negative subset membership
- The accepted immediate `subset_tag` vocabulary is:
  - `positive`
  - `negative`
- Under the current authoritative evaluator path:
  - `reports/test_metrics.csv` must preserve the exact dataset `image_id` for each per-image row and append `subset_tag` after the D-036 required ordered prefix
  - validation/test qualitative manifests must preserve the same exact `image_id` values and include the same `subset_tag` field for each sampled image entry
- The existing boolean `positive` field remains valid as a machine-friendly flag, but it is no longer the only subset-membership signal in authoritative evaluation outputs.

Reason:
- `P1.2` remained open after D-036 because the evaluator still expressed subset membership only indirectly through a boolean `positive` column.
- For human audit and downstream aggregation, explicit subset labels are easier to scan and less error-prone than reconstructing subset identity from booleans or summary tables.
- The authoritative report package already depends on exact image traceability, so the per-image CSV rows and qualitative manifests should make that traceability explicit rather than implicit.

Alternatives considered:
- Keep only the boolean `positive` field and rely on downstream readers to derive subset names.
- Add subset tags only to summary YAML, not to per-image or qualitative outputs.
- Rename the existing `positive` column instead of adding an explicit subset-tag field.

Impact on experiments / methodology:
- Authoritative evaluation outputs are now easier to audit because every per-image record and sampled qualitative entry explicitly says both "which image" and "which subset."
- Existing D-036 canonical required columns stay stable; `subset_tag` is an additive evaluation field layered on top rather than a rewrite of the ordered required prefix.
- `P1.2` remains open only for the final metadata-completeness subtask after this image-ID/subset-tag surface is fixed.

## 2026-04-15 / D-038

Decision:
- The final `P1.2` output-metadata completeness contract is now fixed for authoritative evaluation artifacts that depend on saved threshold reuse.
- The persisted `selection/selection_state.yaml` must now include:
  - `selection_state_path`
  - `train_mask_variant`
  - `eval_mask_variant`
  - the selected threshold/postprocess fields already required earlier
- Any authoritative evaluation artifact derived from a reused `selection_state.yaml` must now expose, in its own metadata surface:
  - the reused `selection_state_path`
  - `train_mask_variant`
  - `eval_mask_variant`
  - `selected_threshold`
  - `selected_postprocess`
- Under the current stack, this completeness rule applies at least to:
  - per-image `reports/test_metrics.csv` rows as additive metadata columns
  - `reports/test_summary.yaml`
  - validation/test qualitative `manifest.yaml` files
- Selection-state reuse during test evaluation must now validate `train_mask_variant` in addition to the already-recorded checkpoint, dataset root, evaluation mask variant, and input size.

Reason:
- `P1.2` remained open after D-036 and D-037 because threshold reuse and mask-variant context were still split across files in a way that made downstream auditing harder than necessary.
- The repository already treated `selection_state.yaml` as the canonical saved-threshold artifact, so downstream outputs should explicitly point back to that artifact rather than only restating a scalar threshold value.
- Both training-target and evaluation-target mask variants matter scientifically in this repo, so evaluation outputs are incomplete if they expose only `eval_mask_variant` and omit `train_mask_variant`.

Alternatives considered:
- Treat run-level metadata as sufficient and keep report/manifest metadata thinner.
- Record only scalar threshold values in downstream outputs without an explicit pointer back to `selection_state.yaml`.
- Carry only `eval_mask_variant` and assume `train_mask_variant` can always be recovered later from another file.

Impact on experiments / methodology:
- `P1.2` is now decision-complete: authoritative trainer/evaluator outputs have an explicit schema contract, explicit image/subset traceability, and explicit threshold/mask-variant metadata completeness.
- Future report review can trace any held-out metric row or qualitative sample back to the exact reused threshold artifact and both mask-variant roles without opening a second file first.
- The next critical-path blocker now moves to `P1.3` rather than remaining inside schema cleanup.

## 2026-04-15 / D-039

Decision:
- The current `hausdorff` path is removed from authoritative paper-path reporting instead of being repaired in-place during this recovery turn.
- Under the current recovery path, authoritative outputs must not emit or rely on `hausdorff` in:
  - `reports/test_metrics.csv`
  - `reports/test_summary.yaml`
  - validation/test qualitative manifests
  - console summary tables used for authoritative run review
- The existing `src/training/metrics.py::hausdorff_distance` helper remains non-authoritative debug code only until a future explicit decision replaces it with a correctly specified and regression-tested metric such as true HD95.
- D-036's authoritative evaluation CSV prefix is therefore now interpreted without `hausdorff`; the required reported metrics are:
  - `dice`
  - `iou`
  - `precision`
  - `recall`
  - `f1`

Reason:
- The current helper docstring claims `95th-percentile Hausdorff distance`, but the implementation actually takes the maximum directed distance on each side and averages valid batch elements, so the label is scientifically wrong.
- Recovery priority here is trustworthy reporting, not speculative metric redesign.
- Keeping a mislabeled distance metric in authoritative outputs would silently undermine the corrected reporting path we just finished under `P1.2`.

Alternatives considered:
- Re-implement HD95 immediately in the same turn.
- Keep the current value but rename it in-place to a non-HD95 variant.
- Leave `hausdorff` in outputs with only a documentation warning.

Impact on experiments / methodology:
- `P1.3` is decision-complete for the current recovery path: no mislabeled Hausdorff metric remains in authoritative reported outputs.
- Historical `hausdorff` values in old outputs remain non-authoritative and should not be cited.
- If a future paper version truly needs a distance metric, it must re-enter through a new explicit decision plus a dedicated correctness test harness rather than by reviving the current helper silently.

## 2026-04-16 / D-040

Decision:
- Under the current recovered project state, Foundation X is deferred from the main paper path rather than promoted as an active result family.
- Until a future hybrid candidate clears both D-033 and D-034, Foundation X may appear only in one of two scoped roles:
  - a short future-work / limitations note, or
  - a clearly labeled appendix-side leakage-aware ablation section that does not change the headline paper narrative
- Under the current state, Foundation X must not appear as:
  - an abstract-level contribution
  - a main results-table anchor
  - the default experimental storyline for the paper body
- The trusted full-image `pretrained_resnet34_unet` baseline remains the main paper-path supervised anchor unless a later explicit decision changes that role after new evidence.

Reason:
- The current repository has no authoritative hybrid evidence package that clears D-033 and D-034, while the hybrid engineering blockers in `P1.9` through `P1.11` remain open.
- D-035 already fixed the leakage-aware claim boundary; this decision narrows the remaining `P1.12` methodology question from "what can be claimed" to "where Foundation X is allowed to appear in the paper at all."
- Keeping Foundation X out of the headline narrative avoids paper-level dependence on an unresolved, SIIM-exposed, currently deferred branch.

Alternatives considered:
- Make Foundation X a required main-paper in-domain transfer section even before a qualifying hybrid evidence package exists.
- Treat Foundation X as a mandatory ablation section in the main body despite the open engineering blockers.
- Remove Foundation X from all paper discussion immediately, including appendix or future-work framing.

Impact on experiments / methodology:
- The first `P1.12` subtask is now decision-complete: Foundation X is deferred from the main paper path under the current recovered state.
- The remaining `P1.12` work is now narrower: operationalize the forbidden-claim list and the comparison rules back to the trusted full-image baseline.
- Any notebook, draft, or summary that places Foundation X in the abstract, headline tables, or main storyline before D-033 and D-034 are cleared is methodologically non-authoritative.

## 2026-04-16 / D-041

Decision:
- The current Foundation X forbidden-claim list is now operational rather than only conceptual.
- Under the current setup, any manuscript text, notebook summary, table caption, slide, or repo note is methodologically non-authoritative if it does any of the following for Foundation X or the hybrid:
  - describes it as `external pretraining`, `clean pretraining`, `out-of-domain pretraining`, or equivalent wording that hides SIIM exposure
  - describes it as `generalization`, `target-unseen transfer`, `cross-dataset transfer into SIIM`, or equivalent wording that implies SIIM was unseen during pretraining
  - attributes any gain primarily to generic foundation-model knowledge without simultaneously acknowledging SIIM exposure and the leakage-aware framing
  - presents a Foundation X result as the default superior model, the main project contribution, or the headline comparison arm under the current recovered state
  - states or implies that a Foundation X result invalidates the trusted full-image `pretrained_resnet34_unet` baseline as the paper anchor without a later explicit decision that changes D-040
- Under the current setup, any allowed Foundation X mention must remain qualified as leakage-aware secondary evidence, future work, limitations context, or appendix-side ablation rather than clean-transfer evidence.

Reason:
- D-035 already fixed the scientific claim boundary, but the repo still needed an operational review rule for catching inflated wording in docs, notebooks, captions, and summaries.
- The main remaining methodology risk is not hidden in code anymore; it is wording drift that quietly upgrades a leakage-aware side path into a stronger scientific claim than the evidence supports.
- Making the forbidden list operational reduces ambiguity during future reporting cleanup under `P2.2` and later manuscript work.

Alternatives considered:
- Keep D-035 as a high-level framing rule and leave wording review informal.
- Ban all Foundation X mentions entirely instead of allowing leakage-aware secondary mentions.
- Wait until manuscript drafting to define the exact forbidden-claim surface.

Impact on experiments / methodology:
- The second `P1.12` subtask is now decision-complete: the current setup has an explicit forbidden-claim list for Foundation X.
- Validation/review can now fail a document for wording inflation even if no new experiment was run.
- The remaining `P1.12` task is now only the baseline-comparison rule set for any future leakage-aware Foundation X discussion.

## 2026-04-16 / D-042

Decision:
- Any future leakage-aware Foundation X or hybrid discussion must use the trusted full-image `pretrained_resnet34_unet` baseline as its mandatory comparison anchor.
- Under the current setup, a Foundation X result is methodologically non-authoritative unless the comparison record explicitly includes all of the following:
  - the trusted baseline identity: full-image `pretrained_resnet34_unet`
  - the trusted baseline held-out reference score: positive-only Dice mean `0.4951`
  - the candidate Foundation X / hybrid held-out `test` positive-only Dice under the same trusted evaluation regime
  - the absolute delta versus `0.4951`
  - whether the candidate did or did not clear the D-033 keep threshold `>= 0.5151`
- Under the current setup, Foundation X comparisons must not use any of the following as the primary narrative anchor:
  - the failed immediate crop arm (`0.4625`)
  - the plain U-Net as a stand-alone headline comparator
  - legacy `results/` artifacts
  - notebook-only metrics without the D-034 evidence package
- Any allowed Foundation X table, appendix note, or future-work mention must present the baseline-relative comparison as secondary context, not as a replacement for the baseline anchor.

Reason:
- Once D-040 and D-041 fixed paper placement and forbidden wording, the remaining ambiguity was how a future leakage-aware Foundation X result should be compared at all.
- The project now has exactly one trusted supervised anchor with a complete authoritative evidence package: the full-image `pretrained_resnet34_unet` baseline at `0.4951`.
- Forcing every future Foundation X mention back to that anchor prevents narrative drift toward weaker or non-authoritative comparators.

Alternatives considered:
- Allow Foundation X appendix discussion without an explicit baseline-relative delta.
- Allow the crop arm or plain U-Net to serve as the main comparator in some contexts.
- Postpone comparison rules until a future hybrid candidate actually exists.

Impact on experiments / methodology:
- `P1.12` is now decision-complete: the leak-aware Foundation X methodology has an explicit paper role, explicit forbidden claims, and explicit baseline-comparison rules.
- Any future Foundation X discussion that omits the baseline score `0.4951`, the candidate score, the absolute delta, or the D-033 threshold status is methodologically incomplete.
- The next critical-path blocker now moves to `P2.1`, not additional Foundation X framing work.

## 2026-04-16 / D-043

Decision:
- The publication-grade evaluation upgrade path will use repeated stratified train/val/test splits rather than single-pass 5-fold cross-validation.
- The reason this is the primary direction is methodological continuity with the already trusted authoritative pipeline:
  - train on `train`
  - select threshold on `val` only
  - report held-out metrics on `test`
- Under the current recovered stack, each repeated split will preserve that same three-way discipline instead of collapsing evaluation into a simpler fold-only loop.
- Plain 5-fold CV is not the chosen primary path because, under the current threshold-selection and artifact contracts, it would either:
  - force a nested validation design that the repo has not yet specified, or
  - weaken the clean `val`-only threshold-selection rule by blurring validation and test roles.

Reason:
- The repo already has one trusted end-to-end evaluation path, and it is explicitly built around a three-way split with saved validation-only threshold selection and held-out test reporting.
- Repeated stratified splits reuse that trusted protocol with minimal methodological drift, while still reducing single-split fragility.
- Choosing a direction that matches the existing artifact/evaluator contracts lowers implementation risk and keeps future results easier to compare back to the currently trusted baseline evidence package.

Alternatives considered:
- Promote standard 5-fold CV immediately as the default publication path.
- Keep single-split evaluation as the final publication plan.
- Defer the direction choice until after implementing confidence intervals.

Impact on experiments / methodology:
- The first `P2.1` subtask is now decision-complete: the repo's publication-grade evaluation direction is repeated stratified splits, not plain 5-fold CV.
- Future `P2.1` work now narrows to:
  - how many repeated splits to run
  - how to compute confidence intervals
  - how to define paired comparison rules and final evidence packaging
- Any future proposal to use 5-fold CV as the main paper path now requires a separate explicit methodology decision that also resolves how validation-only threshold selection will be preserved.

## 2026-04-16 / D-044

Decision:
- Publication-grade confidence intervals and model-vs-model comparisons will treat the repeated split as the statistical unit, not the individual image.
- Under the repeated stratified split plan, the reporting strategy is:
  - for a single model: report the arithmetic mean of the held-out `test` metric across repeated splits plus a two-sided 95% percentile bootstrap confidence interval over split-level values
  - for a model comparison: report the arithmetic mean of the paired split-level deltas plus a two-sided 95% percentile bootstrap confidence interval over those paired deltas
- The mandatory pairing rule is:
  - two models are paired only when they are trained/evaluated on the exact same repeated split instances under the same trusted evaluation regime
  - each paired comparison unit therefore consists of one split index / seed instance shared by both models
- The primary inferential target remains the held-out `test` positive-only Dice mean, so the default paired comparison statistic is:
  - `candidate_test_dice_pos_mean - reference_test_dice_pos_mean`
- Image-level bootstrap is not the primary publication path because it would understate the uncertainty introduced by split choice after the project has already acknowledged single-split fragility.
- Unpaired comparisons across different split instances are methodologically weaker and should not be the default headline comparison once repeated splits exist.

Reason:
- `P2.1` exists because uncertainty from split choice is now a known methodological concern; the CI strategy therefore needs to reflect split-level variability rather than pretending the split is fixed.
- The repo already treats the evaluation run as a split-specific authoritative artifact package, so the clean statistical unit is the repeated split result, not an image pooled across heterogeneous split realizations.
- Pairing by identical split instances preserves the strongest possible architecture comparison while controlling for the split-level variance that would otherwise dominate noisy medical-segmentation comparisons.

Alternatives considered:
- Use image-level bootstrap as the default CI path.
- Use unpaired model comparisons across different split realizations.
- Defer all CI/pairing decisions until after the repeated-split runner exists.

Impact on experiments / methodology:
- The second `P2.1` subtask is now decision-complete: publication-grade uncertainty will be split-bootstrap based and model comparisons will be paired by identical split instances.
- Future `P2.1` work now narrows to the remaining operational items:
  - how many repeated split instances to run
  - what minimum artifact/evidence package each split instance must contribute to the final report bundle
- Any future report that cites repeated-split confidence intervals or model deltas without split-level pairing or without split-bootstrap semantics is methodologically incomplete.

## 2026-04-16 / D-045

Decision:
- The minimum evidence package for final repeated-split reporting is now fixed.
- A publication-grade final report is methodologically incomplete unless it includes all of the following:
  - a split manifest for the repeated-split study that records every split instance identifier / seed and the exact train/val/test IDs used for that instance
  - one authoritative run-artifact package per model per split instance, each preserving the already trusted single-run evidence family (`metadata`, config snapshot, selection state, checkpoints, `test_metrics.csv`, `test_summary.yaml`, qualitative manifests)
  - one machine-readable split-level aggregation table that records, for each model and each split instance, at minimum:
    - split instance identifier
    - dataset/split fingerprint context
    - selected threshold
    - held-out `test` positive-only Dice mean
    - any additional reported held-out `test` metrics kept on the paper path
  - one machine-readable paired-comparison table that records, for each shared split instance and each named model-vs-model comparison:
    - reference model
    - candidate model
    - split instance identifier
    - split-level delta on held-out `test` positive-only Dice mean
  - one final summary artifact that reports:
    - repeated-split model means
    - split-bootstrap 95% confidence intervals
    - paired-delta means
    - paired-delta 95% confidence intervals
    - the exact number of completed split instances contributing to each statistic
- Notebook screenshots, copied console logs, or only a final averaged score are not sufficient as the final repeated-split evidence package.

Reason:
- D-043 and D-044 already fixed the evaluation direction and the statistical unit, but the repo still needed a minimum artifact contract for what a publication-grade repeated-split result bundle must actually contain.
- The project has already recovered from stale-artifact ambiguity once, so the repeated-split upgrade cannot rely on ad hoc tables or manually copied averages.
- Requiring both split-level and paired-delta machine-readable tables ensures that future summary statistics can be audited back to the exact authoritative split instances.

Alternatives considered:
- Require only final summary means and confidence intervals.
- Let each repeated-split study define its own artifact shape informally.
- Defer the evidence-package definition until after the orchestration code exists.

Impact on experiments / methodology:
- `P2.1` is now decision-complete: the repo has a fixed publication-grade evaluation direction, a fixed uncertainty/comparison rule, and a fixed minimum evidence package for final repeated-split reporting.
- Any future repeated-split result that lacks split manifests, per-split authoritative run packages, split-level aggregation tables, paired-delta tables, or the final summary artifact is methodologically incomplete.
- The next critical-path blocker now moves to `P2.2`, not additional repeated-split methodology design.

## 2026-04-16 / D-046

Decision:
- `docs/foundation_nnunet_dev_guide.md` is now explicitly classified as legacy design context, not as a source-of-truth implementation or methodology guide.
- If that file conflicts with the recovered repository state, the authoritative precedence order is:
  - `RECOVERY_TODO.md`
  - `AGENT_CONTEXT.md`
  - `DECISIONS.md`
  - `VALIDATION_CHECKLIST.md`
  - current code and tests
  - only then the legacy dev guide as historical context

Reason:
- The guide still contains stale assumptions about raw annotation files, processed dataset layout, `results/` usage, and Foundation X / hybrid posture.
- Keeping the historical guide is acceptable for context, but it must not silently outrank the recovery memory that now governs the project.

Alternatives considered:
- Rewrite the entire guide in one step.
- Delete the guide completely.
- Leave the guide untouched and rely on readers to discover the contradiction themselves.

Impact on experiments / methodology:
- The first `P2.2` atomic cleanup now has a clear rule: the old dev guide is preserved, but explicitly downgraded to legacy-only context.
- Future docs cleanup can proceed file-by-file without letting this guide silently reintroduce stale assumptions.
- Remaining `P2.2` work is now narrower: clean the still-active docs/notebooks that continue to present stale operational guidance.

## 2026-04-16 / D-047

Decision:
- `CLAUDE.md` is now explicitly classified as legacy operational context, not as an authoritative runbook for the recovered project state.
- If `CLAUDE.md` conflicts with the recovered methodology, the precedence order is the same as D-046:
  - `RECOVERY_TODO.md`
  - `AGENT_CONTEXT.md`
  - `DECISIONS.md`
  - `VALIDATION_CHECKLIST.md`
  - current code and tests
  - only then `CLAUDE.md` as historical context

Reason:
- `CLAUDE.md` still presents a hybrid-first project description, legacy `results/` usage, older SIIM `stage_2` assumptions, and stale checkpoint/evaluation commands as if they were the active workflow.
- Leaving it unmarked would let an actively named top-level guide silently compete with the recovered repo memory.

Alternatives considered:
- Rewrite the entire file in one pass.
- Delete `CLAUDE.md` completely.
- Leave it untouched and rely on the recovery files alone.

Impact on experiments / methodology:
- The second `P2.2` atomic cleanup now has a clear precedence rule for `CLAUDE.md`.
- Readers are redirected away from stale operational guidance before it can affect new experiments or documentation.
- Remaining `P2.2` work now narrows further to the legacy training notebooks and any other active-looking stale docs.

## 2026-04-16 / D-048

Decision:
- `notebooks/train_colab.ipynb` is now explicitly classified as legacy / non-authoritative operational context, not as an authoritative Colab runbook for the recovered project state.
- If this notebook conflicts with the recovered methodology, the same precedence order applies:
  - `RECOVERY_TODO.md`
  - `AGENT_CONTEXT.md`
  - `DECISIONS.md`
  - `VALIDATION_CHECKLIST.md`
  - current code and tests
  - only then the legacy notebook as historical context

Reason:
- The notebook still contains stale workflow cells that copy outputs into legacy `results/`, preserve hybrid-first operational paths, and reflect older checkpoint/reporting assumptions.
- Keeping the notebook is fine for historical context, but it must not silently present itself as the active Colab workflow after the recovery decisions.

Alternatives considered:
- Rewrite the full notebook into the new authoritative workflow in one pass.
- Delete the notebook entirely.
- Leave it untouched and rely only on the notebooks README.

Impact on experiments / methodology:
- The next `P2.2` cleanup step now has a clear rule: `train_colab.ipynb` is preserved, but explicitly downgraded to legacy-only context.
- Readers now hit a warning before seeing stale `results/` and hybrid-first cells.
- Remaining `P2.2` work narrows to the other legacy training notebook and any still-active stale notebook content.

## 2026-04-19 / D-049

Decision:
- `notebooks/train_local.ipynb` is now explicitly classified as legacy / non-authoritative operational context, not as an authoritative local training runbook for the recovered project state.
- If this notebook conflicts with the recovered methodology, the same precedence order applies:
  - `RECOVERY_TODO.md`
  - `AGENT_CONTEXT.md`
  - `DECISIONS.md`
  - `VALIDATION_CHECKLIST.md`
  - current code and tests
  - only then the legacy notebook as historical context

Reason:
- The notebook still contains stale workflow cells that point to the old processed dataset root, preserve legacy `results/` outputs, and expose hybrid-first checkpoint and evaluation assumptions.
- Keeping the notebook is acceptable for historical context, but it must not present itself as the active local training workflow after the recovery decisions.

Alternatives considered:
- Rewrite the full notebook into the recovered authoritative workflow in one pass.
- Delete the notebook entirely.
- Leave it untouched and rely only on repo-memory files plus the notebooks README.

Impact on experiments / methodology:
- `P2.2` can now close because the remaining legacy local training notebook is explicitly downgraded to legacy-only context.
- Readers now hit a warning before seeing stale `results/`, old processed-dataset assumptions, and hybrid-first cells.
- The next critical-path blocker moves back to `P1.9`, not more documentation cleanup.

## 2026-04-19 / D-050

Decision:
- `foundation_x.frozen` is now the intended single source of truth for hybrid backbone gradient semantics.
- Intended frozen-mode semantics:
  - Foundation X backbone parameters must have `requires_grad=False`.
  - Backbone forward must run without gradient tracking.
  - Backbone must stay in eval-only mode during both training and evaluation.
- Intended unfrozen-mode semantics:
  - Foundation X backbone parameters must have `requires_grad=True`.
  - Backbone forward must run with gradient tracking enabled.
  - No unconditional `torch.no_grad()` wrapper may remain in either `HybridFoundationUNet.forward()` or `FoundationXBackbone.forward()`.
  - Backbone module mode policy must be explicit and testable instead of silently forcing frozen semantics through hidden `eval()` or `no_grad()` behavior.

Reason:
- The current code reads `foundation_x.frozen` from config but still hardcodes frozen behavior in two forward paths:
  - `src/models/hybrid.py::HybridFoundationUNet.forward()` wraps Foundation X extraction in unconditional `torch.no_grad()`
  - `src/models/backbone.py::FoundationXBackbone.forward()` wraps backbone execution in unconditional `torch.no_grad()`
- That means `frozen=false` cannot currently produce the gradients that the config surface implies.

Alternatives considered:
- Keep hybrid permanently frozen and deprecate `foundation_x.frozen=false`.
- Allow partial unfreezing without a single explicit semantics contract.
- Treat the current hidden frozen behavior as acceptable because hybrid is deferred.

Impact on experiments / methodology:
- `P1.9` now has an explicit first-step contract: future implementation must make frozen and unfrozen behavior observable, not implicit.
- The next `P1.9` blocker narrows to optimizer-parameter filtering and backbone mode policy under this contract.
- Any future hybrid reopening evidence under D-034 must satisfy D-050 before gradient-flow claims are accepted.

## 2026-04-20 / D-051

Decision:
- The current hybrid optimizer/filtering and mode-policy inventory is now fixed in repo memory.
- Inventory conclusions:
  - `src/training/trainer.py::build_optimizer()` already filters parameters through `param.requires_grad`, so frozen Foundation X parameters are excluded from optimizer groups when the backbone is truly frozen.
  - `src/models/backbone.py::__init__()` only sets `requires_grad=False` when `frozen=True`; therefore unfrozen mode would leave backbone parameters trainable by default.
  - `src/models/hybrid.py::HybridFoundationUNet.train()` and `src/models/backbone.py::FoundationXBackbone.train()` only force backbone `eval()` when the corresponding frozen flag is true; those local train-mode hooks are compatible with D-050.
  - `src/training/trainer.py` still contains a stronger conflicting policy: every training epoch unconditionally calls `model.foundation_x.backbone.eval()` whenever the model has a Foundation X branch, even if `foundation_x.frozen=false`.
  - Because both forward paths still wrap Foundation X execution in unconditional `torch.no_grad()`, current optimizer inclusion of unfrozen backbone parameters does not produce usable gradients yet.

Reason:
- `P1.9` needed an exact code-path inventory before behavior-changing edits.
- The inventory shows optimizer parameter filtering is not the primary blocker; hidden frozen semantics in forward/mode policy are.

Alternatives considered:
- Treat optimizer filtering itself as broken and postpone the inventory until after code edits.
- Collapse parameter filtering and mode-policy analysis into one larger implementation step without first recording the current-state conclusions.

Impact on experiments / methodology:
- The second `P1.9` subtask can now close: current optimizer filtering and mode-policy behavior have been explicitly mapped against D-050.
- The next `P1.9` blocker narrows to code alignment: remove the trainer-side unconditional backbone `eval()` override and then remove the unconditional `torch.no_grad()` wrappers so gradient-flow validation becomes meaningful.
- Future reviews should not describe optimizer parameter groups as the main hybrid blocker unless D-051 is explicitly contradicted by later code changes.

## 2026-04-20 / D-052

Decision:
- `src/training/trainer.py` must not force the Foundation X backbone into `eval()` during training unless the active hybrid instance is explicitly frozen.
- Trainer-side backbone mode policy is now:
  - if the model has no Foundation X branch, do nothing
  - if the model has a Foundation X branch and the backbone is frozen, keep the backbone in `eval()`
  - if the model has a Foundation X branch and the backbone is unfrozen, do not override its training mode here

Reason:
- D-051 showed that optimizer parameter filtering was already structurally aligned, but `trainer.py` still overrode unfrozen behavior by calling `model.foundation_x.backbone.eval()` on every epoch.
- That unconditional override made it impossible for future unfrozen-mode gradient checks to mean what the config surface claimed.

Alternatives considered:
- Leave the unconditional trainer override in place and rely only on later gradient tests to reveal the contradiction.
- Remove all trainer-side backbone mode handling entirely.

Impact on experiments / methodology:
- The trainer no longer silently re-freezes unfrozen hybrid backbones at the mode-policy level.
- `tests/test_hybrid_backbone_mode_policy.py` is now the targeted regression harness for this trainer-side contract.
- The next remaining `P1.9` blocker narrows to the unconditional `torch.no_grad()` wrappers, after which explicit gradient-flow validation can become meaningful.

## 2026-04-20 / D-053

Decision:
- The remaining unconditional `torch.no_grad()` wrappers in the Foundation X path are removed.
- Gradient policy is now split cleanly:
  - `src/models/backbone.py::FoundationXBackbone.forward()` uses `torch.set_grad_enabled(not self.frozen)`
  - `src/models/hybrid.py::HybridFoundationUNet.forward()` delegates gradient behavior to the backbone instead of wrapping Foundation X extraction in its own `no_grad()` block

Reason:
- After D-052, the last code-level blocker preventing meaningful frozen/unfrozen gradient checks was the pair of unconditional `torch.no_grad()` wrappers in the Foundation X path.
- Keeping those wrappers would continue to nullify `foundation_x.frozen=false` even after trainer-side mode policy was fixed.

Alternatives considered:
- Leave the wrappers in place and treat all hybrid work as permanently frozen-only.
- Remove only one wrapper and rely on the other module to keep behavior consistent.

Impact on experiments / methodology:
- Frozen vs unfrozen gradient behavior is now controlled by the explicit `foundation_x.frozen` contract rather than hidden wrappers.
- `tests/test_hybrid_gradient_flow.py` now provides targeted regression coverage that frozen mode suppresses Foundation X gradients and unfrozen mode allows them.
- `P1.9` can now close, and the next critical-path blocker moves to `P1.10` fusion alignment.

## 2026-04-20 / D-054

Decision:
- The current hybrid fusion-map inventory is now fixed in repo memory before any redesign work.
- Current active mapping in `src/models/hybrid.py` is:
  - `fx[0]` (`128` ch, `H/4 × W/4`) -> `e1` (`64` ch, `H × W`) via `fusion1`
  - `fx[1]` (`256` ch, `H/8 × W/8`) -> `e2` (`128` ch, `H/2 × W/2`) via `fusion2`
  - `fx[2]` (`512` ch, `H/16 × W/16`) -> `e3` (`256` ch, `H/4 × W/4`) via `fusion3`
  - `fx[3]` (`1024` ch, `H/32 × W/32`) -> `e4` (`512` ch, `H/8 × W/8`) via `fusion4`
- Because `FusionBlock.forward()` upsamples Foundation X features to the U-Net feature size when they differ, every current fusion stage performs a `4x` spatial upsample into a shallower U-Net stage.
- The deepest Foundation X feature (`fx[3]`, `H/32`) is not consumed at a natural `H/32` context stage; it is first upsampled to `H/8` for `fused4`, then pooled back down to `H/16` before the bottleneck.

Reason:
- `P1.10` needed an exact current-state inventory before choosing a corrected target mapping.
- The code already made the mismatch visible, but it was not yet written down as a single authoritative reference for later redesign decisions.

Alternatives considered:
- Jump directly into a new target mapping without first freezing the current broken mapping in repo memory.
- Treat the existing comments in `src/models/hybrid.py`, `src/models/backbone.py`, and `src/models/unet.py` as sufficient documentation.

Impact on experiments / methodology:
- `P1.10` now has a fixed baseline inventory: the current hybrid does not align Foundation X stages to semantically matched U-Net stages.
- The next `P1.10` blocker narrows to choosing the corrected target mapping and deciding whether an explicit `H/32` context path is required for the deepest Foundation X feature.
- Future discussions should refer to D-054 when describing why the current hybrid fusion is considered structurally misaligned.

## 2026-04-20 / D-055

Decision:
- The corrected target stage mapping for both `256` and `512` inputs is now fixed at the level of relative spatial scale.
- Target mapping:
  - Foundation X `fx[0]` (`H/4`) aligns with U-Net `e3` (`H/4`)
  - Foundation X `fx[1]` (`H/8`) aligns with U-Net `e4` (`H/8`)
  - Foundation X `fx[2]` (`H/16`) aligns with the U-Net bottleneck / context stage (`H/16`)
  - Foundation X `fx[3]` (`H/32`) is reserved for a dedicated deeper context slot, not for direct fusion into any current `e1/e2/e3/e4` encoder stage
- This mapping is scale-invariant across the accepted input sizes:
  - for `256`: `64,32,16,8` map to `64,32,16,8`
  - for `512`: `128,64,32,16` map to `128,64,32,16`
- Therefore no corrected mapping may fuse Foundation X features into `e1` (`H`) or `e2` (`H/2`) under the current hybrid redesign path.

Reason:
- D-054 showed the current hybrid fuses every Foundation X stage into a shallower U-Net encoder stage and relies on `4x` upsampling at every fusion site.
- The U-Net encoder only exposes natural semantic-scale matches beginning at `e3` (`H/4`), not at `e1` or `e2`.

Alternatives considered:
- Keep the current `fx[0]->e1`, `fx[1]->e2`, `fx[2]->e3`, `fx[3]->e4` ladder and accept large upsampling into shallow stages.
- Force the deepest Foundation X feature into `e4` or the existing bottleneck despite scale mismatch.
- Collapse Foundation X to fewer stages and ignore the spatial hierarchy entirely.

Impact on experiments / methodology:
- `P1.10` now has a corrected target mapping anchor for both supported input sizes without yet committing to the exact architectural implementation.
- The next `P1.10` blocker narrows to architecture: decide whether the model needs an explicit deeper `H/32` context head and how `fx[3]` should enter that path.
- Future redesign work should treat `e1/e2` fusion as off-protocol unless a later explicit decision reopens that choice.

## 2026-04-20 / D-056

Decision:
- If the hybrid path is kept, the redesign now requires an explicit deeper `H/32` context head.
- The current U-Net path only exposes encoder skips at `H,H/2,H/4,H/8` plus a bottleneck at `H/16`; it does not contain a natural `H/32` slot.
- Therefore the deepest Foundation X stage `fx[3]` (`H/32`) may not be forced into `e4` (`H/8`) or directly into the existing bottleneck (`H/16`) by corrective resize alone under the current redesign path.
- The dedicated `H/32` slot reserved by D-055 is now interpreted as a real architectural context head, not a naming placeholder.

Reason:
- D-055 already fixed the corrected relative-scale mapping and reserved `fx[3]` for a dedicated `H/32` slot.
- The baseline U-Net implementation confirms that no such slot exists today: the network bottoms out at `H/16`.
- Without an explicit deeper context head, the redesign would either discard the deepest Foundation X feature or immediately reintroduce the same scale-mismatch behavior that D-054 and D-055 were meant to remove.

Alternatives considered:
- Reuse `e4` (`H/8`) as the deepest fusion site and accept large upsampling of `fx[3]`.
- Inject `fx[3]` straight into the existing `H/16` bottleneck by resize-only adaptation.
- Drop `fx[3]` entirely and keep only `fx[0:3]` in the redesign path.

Impact on experiments / methodology:
- `P1.10` now has a fixed architectural requirement: any corrected hybrid redesign must include an explicit `H/32` context head if it keeps the four-stage Foundation X hierarchy.
- The next `P1.10` blocker narrows to implementation detail: decide exactly how `fx[3]` enters that new `H/32` context head and how that head reconnects to the `H/16` bottleneck path.
- Future redesign work should treat resize-only reuse of `e4` or the current bottleneck for `fx[3]` as off-protocol unless a later explicit decision reopens that choice.

## 2026-04-20 / D-057

Decision:
- The deepest Foundation X feature `fx[3]` (`H/32`) now enters the redesign through its own dedicated `H/32` context head and not through any `FusionBlock` with U-Net encoder skips.
- That `H/32` context head consumes only `fx[3]` at native scale, applies local context processing at `H/32`, and then performs exactly one learned `2x` transition from `H/32` to `H/16`.
- After that single `H/32 -> H/16` transition, the upsampled deepest-context output reconnects to the main path only at the `fx[2]`-aligned `H/16` bottleneck/context branch fixed by D-055.
- The reconnection is therefore context-to-context: `fx[3]` may not directly concatenate with `e4`, decoder skips, or shallow encoder features, and it may not bypass the `H/16` context merge path by jumping straight into the final decoder.

Reason:
- D-055 fixed `fx[3]` to a dedicated `H/32` slot, and D-056 fixed that slot as a real deeper context head rather than a placeholder.
- The cleanest way to preserve scale semantics is to let `fx[3]` stay alone at native `H/32`, then reconnect only once at the immediately adjacent `H/16` context branch.
- Any attempt to push `fx[3]` into `e4` or decoder skips would reintroduce the same multi-level scale mismatch that the `P1.10` redesign is trying to remove.

Alternatives considered:
- Fuse `fx[3]` directly with `e4` after aggressive upsampling.
- Inject `fx[3]` straight into the existing bottleneck without a dedicated `H/32` processing step.
- Route `fx[3]` to decoder skips as an auxiliary deep supervision signal.

Impact on experiments / methodology:
- `P1.10` now has a fixed deepest-feature usage rule: `fx[3]` is a pure deeper-context input, not a skip-fusion feature.
- The next `P1.10` blocker narrows to enforcement: add explicit shape assertions and implementation checks so the eventual redesign proves `fx[0]->e3`, `fx[1]->e4`, `fx[2]->H/16`, and `fx[3]->H/32->H/16 context merge` exactly.
- Future redesign work should treat direct `fx[3]` reuse in `e4` or decoder skips as off-protocol unless a later explicit decision reopens that choice.

## 2026-04-20 / D-058

Decision:
- `src/models/hybrid.py::assert_corrected_hybrid_scale_contract()` is now the canonical in-code assertion helper for the corrected `P1.10` fusion contract.
- The helper enforces:
  - `fx[0]` spatially matches `e3`
  - `fx[1]` spatially matches `e4`
  - `fx[2]` spatially matches the `H/16` bottleneck/context branch
  - `fx[3]` spatially matches the dedicated `H/32` context head
  - `H/32` reconnects to `H/16` through exactly one `2x` transition
  - all participating tensors share the same batch dimension
- `tests/test_hybrid_scale_contract.py` is now the canonical regression harness for this helper and covers both valid (`256`, `512`) and invalid shape layouts.

Reason:
- D-055 through D-057 fixed the corrected mapping, deeper-head requirement, and deepest-feature usage rule, but those decisions still needed an executable assertion surface.
- A dedicated helper lets future redesign work fail fast on scale drift without forcing the still-legacy active forward path to pretend it already satisfies the corrected contract.

Alternatives considered:
- Wait to add assertions until after the full hybrid redesign lands.
- Encode the contract only in markdown/recovery files without an executable test harness.
- Attach the assertions directly to the current forward path even though that path still implements the old D-054 mapping.

Impact on experiments / methodology:
- `P1.10` now has executable shape guards for the corrected design contract.
- The next `P1.10` blocker narrows to code implementation: refactor the active hybrid forward path so it actually satisfies the D-055/D-056/D-057 contract under these assertions.
- Future redesign code should call or mirror this helper rather than re-inventing undocumented shape checks.

## 2026-04-20 / D-059

Decision:
- The active `HybridFoundationUNet` forward path now implements the corrected `P1.10` contract instead of the old D-054 ladder.
- Active mapping is now:
  - `fx[0] -> fusion_e3 -> e3`
  - `fx[1] -> fusion_e4 -> e4`
  - `fx[2] -> h16_fx_context -> pooled e4 / H/16 context`
  - `fx[3] -> h32_context_head -> h32_to_h16 -> context_merge -> H/16 bottleneck/context branch`
- Decoder skips are now:
  - `up4(b) + fused_e4`
  - `up3(d4) + fused_e3`
  - `up2(d3) + e2`
  - `up1(d2) + e1`
- The old direct `fx[0]->e1`, `fx[1]->e2`, `fx[2]->e3`, `fx[3]->e4` fusion path is no longer the active architecture.
- The active forward path now calls `assert_corrected_hybrid_scale_contract()` before the deepest reconnect and hard-fails if the corrected scale contract drifts.

Reason:
- D-055 through D-058 had already fixed the corrected mapping, deeper `H/32` head requirement, deepest-feature usage rule, and executable shape assertions.
- The remaining gap was that `HybridFoundationUNet.forward()` still implemented the old semantically misaligned ladder.

Alternatives considered:
- Leave the corrected contract only in helper/tests while keeping the old forward path alive.
- Partially refactor only the deepest branch and leave `fx[0]/fx[1]` attached to `e1/e2`.
- Delay the active refactor until after `P1.11`.

Impact on experiments / methodology:
- `P1.10` can now close: the active hybrid architecture is scale-aligned by design and guarded by executable shape checks.
- The next critical blocker moves to `P1.11`, which decides the branch-normalization policy for the hybrid input path.
- Future hybrid training attempts should use this refactored path as the only acceptable starting point; the old D-054 ladder is now historical context only.

## 2026-04-20 / D-060

Decision:
- The current codebase does **not** encode a trusted external mean/std normalization contract for Foundation X.
- The only code-proven current effective hybrid input contract is:
  - `src/data/dataset.py` emits grayscale tensors scaled to `[0, 1]`
  - `src/models/backbone.py::FoundationXBackbone.forward()` repeats that grayscale tensor to 3 channels
  - no per-channel mean/std normalization is applied before the Foundation X Swin-B backbone
  - the CNN branch in `src/models/hybrid.py` receives the same underlying grayscale `[0, 1]` tensor, not a separately normalized view
- Therefore any claim that Foundation X is *already* receiving ImageNet-style or checkpoint-specific normalization is unsupported by the local code and must be treated as unverified.
- Until a later explicit `P1.11` decision changes behavior, the current hybrid path should be described as using a shared raw `[0, 1]` input view with grayscale-to-RGB repetition on the Foundation X branch only.

Reason:
- `src/data/dataset.py` ends with `image.astype(np.float32) / 255.0` and returns a single-channel tensor.
- `src/models/backbone.py` only performs `x.repeat(1, 3, 1, 1)` before sending data into the Swin-B backbone.
- No config field, transform, or model-side helper in the current repo applies explicit channel-wise normalization for either Foundation X or the hybrid CNN branch.

Alternatives considered:
- Assume Foundation X should already be using ImageNet mean/std and record that as current behavior.
- Assume the checkpoint implies a hidden normalization contract even though the local code never applies one.
- Treat the ResNet34 baseline path as proof of the hybrid normalization policy.

Impact on experiments / methodology:
- `P1.11` now has a fixed current-state inventory: branch normalization is currently implicit and shared at the raw `[0, 1]` level, not explicitly matched to pretrained backbone expectations.
- The next `P1.11` blocker narrows to policy: decide whether the hybrid should keep one shared raw view or introduce branch-specific normalized views for Foundation X and the CNN path.
- Future hybrid evidence reviews must not claim normalization correctness from code unless a later explicit decision and code change make that true.

## 2026-04-20 / D-061

Decision:
- The hybrid path should **not** keep a single shared raw `[0, 1]` input view as its final normalization policy.
- If the hybrid path is used going forward, it must move to **branch-specific views**:
  - the Foundation X branch receives an explicitly normalized view appropriate for the pretrained Swin-B backbone
  - the CNN branch receives its own explicitly recorded view rather than silently inheriting the Foundation X normalization choice
- This decision fixes the direction of the policy only. It does **not** yet fix the exact normalization constants or the exact code/config surface where the dual-view policy will be applied.

Reason:
- D-060 established that the current code only repeats the grayscale `[0, 1]` tensor to RGB for Foundation X and applies no explicit mean/std normalization anywhere in the hybrid path.
- The two branches do not play the same role: Foundation X is an external pretrained backbone, while the CNN branch is the trainable U-Net-style path inside the hybrid model.
- A single implicit raw view leaves the pretrained branch’s expectations undocumented and makes later hybrid evidence harder to interpret or reproduce.

Alternatives considered:
- Keep one shared raw `[0, 1]` input view for both branches.
- Force both branches to use the same explicit normalized view even if that view is chosen only for Foundation X.
- Delay the shared-vs-branch-specific decision until after another hybrid training attempt.

Impact on experiments / methodology:
- `P1.11` now has its directional policy fixed: hybrid normalization must become branch-specific rather than remaining a shared implicit raw view.
- The next `P1.11` blocker narrows to the exact contract: choose the Foundation X normalization details and the exact implementation surface for both branch views.
- Future hybrid evidence should treat any run that still uses an undocumented shared raw view as pre-policy / non-authoritative for normalization claims.

## 2026-04-20 / D-062

Decision:
- The authoritative hybrid normalization contract is now fixed as a two-view, model-side policy:
  - the CNN branch keeps the dataset-emitted grayscale tensor scaled to raw `[0, 1]`
  - the Foundation X branch starts from that same raw grayscale tensor, repeats it to RGB, and then applies explicit ImageNet channel-wise normalization with:
    - mean = `[0.485, 0.456, 0.406]`
    - std = `[0.229, 0.224, 0.225]`
- The implementation surface is also fixed:
  - `src/data/dataset.py` remains model-agnostic and emits only the shared grayscale `[0, 1]` tensor
  - branch-specific view creation belongs to the hybrid model path, not the dataset or trainer
  - the Foundation X branch-specific transform must live adjacent to the existing grayscale-to-RGB adaptation in `src/models/backbone.py`
  - the CNN branch keeps consuming the raw grayscale tensor directly through `src/models/hybrid.py`
- Future authoritative hybrid configs and run metadata must record both views explicitly rather than describing normalization informally.

Reason:
- D-060 proved that the current dataset path already emits a single grayscale `[0, 1]` tensor and that the hybrid path today only repeats that tensor to RGB for Foundation X.
- D-061 already ruled out keeping one shared implicit raw view as the final policy.
- The dataset is shared across non-hybrid paths; pushing Foundation-X-specific normalization into `src/data/dataset.py` would silently alter unrelated models and blur branch responsibilities.
- The local Foundation X branch is instantiated from a timm Swin-B feature extractor, and no stronger checkpoint-specific normalization contract is encoded in the repo. Under that constraint, explicit ImageNet normalization is the most defensible recorded contract for the pretrained RGB branch.
- Keeping the CNN branch on raw grayscale `[0, 1]` preserves the current U-Net-side input semantics instead of forcing the segmentation branch to inherit a normalization choice made only for the pretrained Foundation X encoder.

Alternatives considered:
- Normalize both branches identically with the same ImageNet RGB statistics.
- Move all branch normalization into `src/data/dataset.py`.
- Keep the exact constants undocumented and defer the choice until after another hybrid run.

Impact on experiments / methodology:
- `P1.11` is now fully defined: both the branch split and the exact normalization contract are fixed in repo memory.
- Any future hybrid run that does not implement this exact two-view contract in code and metadata should be treated as pre-policy / non-authoritative for normalization claims.
- The current critical path no longer blocks on hybrid normalization policy design; the next main-path blocker returns to publication-grade repeated-split evaluation work.

## 2026-04-20 / D-063

Decision:
- The first concrete orchestration surface for publication-grade repeated-split evaluation is now fixed.
- A repeated-split study package must expose, at minimum, the following machine-readable layout:
  - `metadata/split_manifest.yaml`
  - `aggregations/split_level_metrics.csv`
  - `comparisons/<comparison_name>_paired_deltas.csv`
  - `summary/final_summary.yaml`
- The split manifest contract is now explicit:
  - one study-level `study_id`
  - dataset context from the trusted processed dataset:
    - `dataset_root`
    - `dataset_fingerprint`
    - `base_split_fingerprint`
  - repeated-split policy context:
    - `split_policy`
    - `selection_metric`
    - `primary_test_metric`
    - `statistical_unit=split_instance`
    - `paired_comparison_unit=shared_split_instance`
  - one entry per split instance carrying:
    - `split_instance_id`
    - `split_seed`
    - exact `train_ids`, `val_ids`, and `test_ids`
    - per-subset counts
    - a per-instance `split_fingerprint`
- Split-manifest generation must canonicalize each split instance by sorting IDs and rejecting any train/val/test overlap.
- Per-model authoritative run packages for each split instance still remain in the already trusted single-run artifact family under `artifacts/runs/`; the repeated-split study package is an orchestration/reporting layer above that family, not a replacement for it.

Reason:
- D-045 already fixed the minimum evidence package, but the repo still had no concrete code-level contract for how a repeated-split study should start materializing that package.
- The split manifest is the cleanest first operational piece because it anchors every future split-level table, paired-delta table, and final summary back to exact train/val/test IDs.
- Enforcing disjoint subsets and canonical ID ordering at manifest-build time closes a silent leakage / reproducibility hole before aggregation code exists.

Alternatives considered:
- Start with bootstrap / paired-delta summary code before a manifest exists.
- Let each repeated-split study define its own ad hoc manifest schema.
- Store only split seeds and infer the actual train/val/test IDs later.

Impact on experiments / methodology:
- `P2.1` is no longer only a methodology note; it now has its first executable orchestration contract in code.
- The next `P2.1` blocker narrows to writing the machine-readable split-level aggregation and paired-delta tables that consume this manifest and point back to authoritative per-split run artifacts.
- Any future repeated-split study that lacks a canonical split manifest with exact IDs should be treated as pre-contract / non-authoritative for publication-grade reporting.

## 2026-04-20 / D-064

Decision:
- The canonical repeated-split aggregation surface is now fixed in code.
- `src/training/run_artifacts.py` now owns two machine-readable writer families above the split manifest:
  - split-level aggregation:
    - `build_split_level_records_from_authoritative_runs(...)`
    - `build_split_level_dataframe(...)`
    - `write_split_level_csv(...)`
  - paired comparison:
    - `build_paired_delta_records(...)`
    - `build_paired_delta_dataframe(...)`
    - `write_paired_delta_csv(...)`
- The split-level aggregation contract is now explicit:
  - it consumes the canonical split manifest plus one authoritative run reference per `(split_instance_id, model_name)`
  - each row must record, at minimum:
    - study/split instance identity
    - model identity and run identity
    - dataset / base split / study split / run split fingerprint context
    - selection threshold / postprocess / selection metric
    - checkpoint path and mask-variant context
    - held-out `test` Dice and IoU means for `all` and `positive` subsets
- The paired-delta contract is now explicit:
  - it is built only from shared split instances
  - the default statistic remains `candidate_test_dice_pos_mean - reference_test_dice_pos_mean`
  - exactly one split-level row per model per shared split instance is required

Reason:
- D-063 introduced the canonical split manifest, but publication-grade repeated-split reporting still had no code path for turning authoritative per-split run artifacts into machine-readable aggregation tables.
- D-045 requires both a split-level table and paired-delta tables as part of the minimum evidence package; leaving those surfaces informal would recreate the stale-artifact ambiguity the recovery process has already removed once.
- Enforcing study-manifest membership, dataset-root consistency, and selection-metric consistency at aggregation time closes another silent drift path before summary statistics are generated.

Alternatives considered:
- Aggregate repeated-split results manually in notebooks or ad hoc scripts.
- Write only the final summary artifact and derive split-level or paired-delta rows later if needed.
- Allow unpaired model deltas across different split instances.

Impact on experiments / methodology:
- `P2.1` now has executable code for the two machine-readable tables that D-045 already required.
- The next `P2.1` blocker narrows to the remaining final-summary surface: writing the publication-grade summary artifact that consumes split-level rows and paired-delta tables.
- Any future repeated-split report that skips these helpers and ships only hand-built tables should be treated as non-authoritative for publication-grade evidence.

## 2026-04-20 / D-065

Decision:
- The final repeated-split summary artifact is now fixed in code.
- `src/training/run_artifacts.py` now owns:
  - `_bootstrap_percentile_ci(...)`
  - `build_final_repeated_split_summary_payload(...)`
  - `write_final_repeated_split_summary(...)`
- The canonical `summary/final_summary.yaml` contract is now:
  - study-level context:
    - `study_id`
    - `dataset_fingerprint`
    - `base_split_fingerprint`
    - `split_policy`
    - `selection_metric`
    - `primary_metric`
    - `ci_level`
    - `bootstrap_samples`
    - `bootstrap_seed`
  - model summaries:
    - `model_name`
    - `model_type`
    - `metric_name`
    - arithmetic mean across split-level values
    - two-sided percentile-bootstrap CI
    - `contributing_split_count`
    - `contributing_split_instance_ids`
  - paired comparison summaries:
    - `comparison_name`
    - `metric_name`
    - `reference_model`
    - `candidate_model`
    - mean paired delta
    - two-sided percentile-bootstrap CI
    - `contributing_split_count`
    - `contributing_split_instance_ids`
- The bootstrap unit is now enforced in code as the split-level value or paired split-level delta, not the image.

Reason:
- D-045 required a final summary artifact with means, 95% CIs, paired-delta CIs, and contributing split counts, but the repo still lacked the writer that actually produced that artifact.
- D-044 already fixed split-level bootstrap and paired comparison semantics, so the summary writer needed to encode that same statistical unit explicitly rather than allow notebook-side reinvention.
- Recording contributing split IDs inside the summary keeps the final headline numbers traceable back to the exact split instances included in each statistic.

Alternatives considered:
- Leave the final summary to notebooks or manual analysis.
- Report only means and omit bootstrap CI metadata.
- Bootstrap images instead of split-level values.

Impact on experiments / methodology:
- `P2.1` is now fully execution-ready from a methodology-contract perspective: split manifests, split-level tables, paired-delta tables, and final summary artifacts all have executable code paths.
- The next blocker is no longer methodology design; it is practical orchestration/execution of a repeated-split study that populates the new artifact surfaces with real authoritative runs.
- Any future repeated-split report that omits `summary/final_summary.yaml` or computes it outside this contract should be treated as non-authoritative for publication-grade evidence.

## 2026-04-20 / D-066

Decision:
- The first concrete repeated-split execution entrypoint is now fixed as `scripts/prepare_repeated_split_study.py`.
- This runner is responsible only for the study-manifest stage, not training or aggregation.
- The canonical behavior is now:
  - read trusted processed image IDs from `images/`
  - derive binary image-level labels from `original_masks/`
  - replay the already trusted `create_splits(...)` policy once per explicit split seed
  - build canonical split instances with IDs `split_001`, `split_002`, ...
  - reject duplicate split seeds
  - reject repeated split fingerprints so a study cannot silently count the same split twice
  - write `metadata/split_manifest.yaml` under `artifacts/repeated_splits/<study_id>/`
- The supporting reusable helpers are now:
  - `src/data/repeated_splits.py::load_processed_dataset_binary_labels(...)`
  - `src/data/repeated_splits.py::build_repeated_split_instances(...)`

Reason:
- D-063, D-064, and D-065 made repeated-split reporting execution-ready on paper, but the repo still had no runner that materialized the first real study artifact from the trusted dataset.
- The study manifest is the lowest-risk execution step because it does not start GPU training yet, but it does lock the exact split instances that every later authoritative run, paired comparison, and summary must reference.
- Re-deriving labels from `original_masks/` keeps repeated-split execution aligned with the already accepted image-level stratification target instead of introducing another label source.

Alternatives considered:
- Jump directly to a train-all-splits runner before a canonical study manifest exists.
- Encode repeated-split studies as only a list of seeds and infer the actual IDs later.
- Allow duplicated split seeds or duplicated split fingerprints inside one study.

Impact on experiments / methodology:
- The repo now has the first practical runner for repeated-split execution, not just reporting helpers.
- The next practical blocker narrows to authoritative per-split run orchestration that consumes the fixed study manifest and fills the per-split run packages.
- Any future repeated-split study that skips this manifest-preparation stage or counts duplicated split instances should be treated as pre-contract / non-authoritative.

## 2026-04-20 / D-067

Decision:
- Effective split context is now a first-class authoritative input surface.
- The accepted surface is `data.splits_path`:
  - if omitted, the stack uses `<dataset_root>/splits.json`
  - if provided, the stack uses that explicit split manifest instead
- `PneumothoraxDataset`, trainer dataloaders, and evaluation dataloaders now all consume that same optional override path.
- Authoritative run metadata must now record:
  - `dataset_root`
  - `splits_path`
  - `base_split_fingerprint` from the trusted dataset manifest
  - `split_fingerprint` from the effective split file actually used by the run

Reason:
- D-066 introduced the first repeated-split study manifest runner, but the actual train/eval stack still implicitly bound itself to `<dataset_root>/splits.json`.
- Without a split-override surface, repeated-split execution would require dataset copies or split-specific dataset views just to change train/val/test membership.
- Without recording both base and effective split fingerprints, authoritative per-split runs would be ambiguous once the same trusted dataset root is reused across many split instances.

Alternatives considered:
- Materialize a full split-specific dataset view directory for every repeated split instance.
- Keep the train/eval stack fixed on `<dataset_root>/splits.json` and generate one dataset root per split.
- Record only the effective split fingerprint without preserving the trusted dataset's base split fingerprint.

Impact on experiments / methodology:
- The repo can now drive the existing train/eval stack from a repeated-split manifest without cloning the trusted dataset.
- The next practical blocker narrows to selection/test/run orchestration that feeds a manifest-derived `data.splits_path` into authoritative per-split runs and then aggregates them.
- Any future repeated-split run that does not record `splits_path`, `base_split_fingerprint`, and the effective `split_fingerprint` should be treated as pre-contract / non-authoritative.

## 2026-04-20 / D-068

Decision:
- The first per-split authoritative study runner is now fixed as `scripts/run_repeated_split_pretrained_study.py`.
- This runner consumes a canonical `metadata/split_manifest.yaml` and, for each split instance:
  - materializes `metadata/split_overrides/<split_instance_id>.json`
  - materializes `metadata/config_overrides/<split_instance_id>.yaml`
  - injects `data.splits_path` into the config override
  - launches `scripts/run_authoritative_pretrained_baseline.py`
  - writes the per-split run into the existing authoritative run family under `artifacts/runs/`
- The canonical per-split run directory naming is now:
  - `artifacts/runs/<study_id>__<split_instance_id>__pretrained_resnet34_unet`
- The runner must also persist a machine-readable execution inventory at:
  - `metadata/pretrained_resnet34_run_inventory.yaml`
- That inventory must record, at minimum:
  - `study_id`
  - `study_manifest_path`
  - `base_config_path`
  - forwarded `stage`
  - one entry per completed split instance with:
    - `split_instance_id`
    - `split_seed`
    - `split_override_path`
    - `config_override_path`
    - `run_dir`

Reason:
- D-066 and D-067 made repeated-split execution possible, but the repo still lacked the first runner that actually chained the existing authoritative pretrained baseline across all split instances.
- Keeping the single-run authoritative runner as the underlying executor avoids inventing a second training/evaluation protocol for repeated-split studies.
- Persisting a run inventory closes the traceability gap between study-level split instances and the run directories later consumed by split-level aggregation.

Alternatives considered:
- Launch per-split runs manually from notebooks or shell commands.
- Build a new independent repeated-split training loop instead of reusing the authoritative single-run runner.
- Infer run directories later from naming conventions without saving a run inventory.

Impact on experiments / methodology:
- The repo now has the first end-to-end per-split orchestration surface for the trusted pretrained baseline path.
- The next practical blocker narrows to study finalization: consuming the run inventory to write `split_level_metrics.csv`, paired-delta tables, and `summary/final_summary.yaml`.
- Any future repeated-split study that launches per-split runs without `pretrained_resnet34_run_inventory.yaml` should be treated as incomplete / non-authoritative at the study level.

## Open decisions requiring evidence

### OD-005
- Topic: Whether the hybrid is retained, redesigned, or deferred from the main paper.
- Needed evidence: a future hybrid candidate that clears D-033 and ships the D-034 evidence package, plus leak-aware framing consistent with D-035, gradient-flow verification, and aligned fusion design.
