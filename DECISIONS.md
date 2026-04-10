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

## Open decisions requiring evidence

### OD-002
- Topic: Final DICOM intensity preprocessing policy.
- Needed evidence: metadata inspection plus visual sanity checks.

### OD-003
- Topic: Implementation verification of the chosen primary model-selection metric and empty-mask handling policy.
- Needed evidence: corrected metric implementation and trainer/evaluator parity.

### OD-004
- Topic: Whether ROI/crop strategy is necessary after the strong baseline is measured.
- Needed evidence: trusted baseline performance and sparsity analysis.

### OD-005
- Topic: Whether the hybrid is retained, redesigned, or deferred from the main paper.
- Needed evidence: trusted baseline, leak-aware framing, gradient-flow verification, and aligned fusion design.
