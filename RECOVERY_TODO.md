# Foundation-nnU-Net Recovery TODO

Purpose: persistent master task list for restoring trust, reproducibility, and publication readiness.

Status markers:
- [ ] not started
- [~] in progress
- [x] done
- [!] blocked

Rules:
- Treat labels and metrics as blockers for all new experiments.
- Do not close a task without recording the validation that proved it.
- When a task is completed, update this file, `AGENT_CONTEXT.md`, `DECISIONS.md`, and `VALIDATION_CHECKLIST.md` if the validation logic changed.

Current strategic direction:
- Fix trust issues first.
- Build a strong supervised baseline second.
- Revisit the hybrid only after trust and baseline gates are satisfied.

## Phase 0: Trust Recovery / Artifact Quarantine

### P0.1 Quarantine stale experiment artifacts
- Status: [x]
- Dependencies: none
- Affected files/modules: `results/`, `notebooks/train_local.ipynb`, `notebooks/train_colab.ipynb`, `src/evaluation/evaluate.py`, `src/training/trainer.py`
- Why it matters: current CSVs and plots do not match the present code path and must not influence decisions.
- Subtasks:
  - [x] Mark all existing files in `results/` as legacy/non-authoritative in repo memory and add an explicit warning inside `results/`.
  - [x] Define whether legacy artifacts will be moved, renamed, or left in place with explicit warnings.
  - [x] Add a repository convention for where authoritative experiment outputs will live.
  - [x] Document that notebook-generated outputs are not evidence unless traceable to config + checkpoint + dataset version.
- Success criteria:
  - No one can mistake current `results/` contents for authoritative evidence.
  - The authoritative output location and naming convention are documented.
- Validation needed before close:
  - Run the stale artifact invalidation checklist in `VALIDATION_CHECKLIST.md`.

### P0.2 Establish experiment provenance requirements
- Status: [x]
- Dependencies: none
- Affected files/modules: `src/training/trainer.py`, `src/evaluation/evaluate.py`, `configs/config.yaml`, future run directories
- Why it matters: every future result must be traceable to exact code, config, dataset, mask variant, checkpoint, threshold, and seed.
- Subtasks:
  - [x] Define mandatory run metadata fields and the Git-free code-fingerprint fallback in repo memory.
  - [x] Define the relative run directory metadata layout and required metadata file names in repo memory.
  - [x] Define the mandatory authoritative output set for a training run in repo memory.
  - [x] Record authoritative primary metric for model selection.
- Success criteria:
  - Provenance schema is documented and ready to implement.
  - A future run can be audited from artifacts alone.
- Validation needed before close:
  - Run reproducibility and stale artifact checklist items.

## Phase 1: Data and Label Correctness

### P0.3 Validate the SIIM RLE contract
- Status: [x]
- Dependencies: P0.1, P0.2
- Affected files/modules: `src/data/preprocess.py`, `src/data/rle_contract.py`, `scripts/validate_siim_rle_contract.py`, raw SIIM annotations
- Why it matters: if the decoder contract is wrong, every saved mask and every reported metric is untrustworthy.
- Subtasks:
  - [x] Identify the authoritative local annotation source for this workspace and mark stale doc references as non-authoritative.
  - [x] Compare current `rle2mask` behavior against authoritative references on curated positive and negative samples.
  - [x] Explicitly resolve the observed suspicious/non-monotonic positive rows rather than assuming current behavior is correct.
  - [x] Decide whether the decoder must support one format or detect multiple possible formats.
  - [x] Write down the accepted decoder contract in `DECISIONS.md`.
- Success criteria:
  - The repository has a documented, justified, testable RLE contract.
  - Curated sample decodes match the authoritative reference.
- Validation needed before close:
  - Label decoding correctness checklist.
  - Overlay sanity checks on curated samples.

### P0.4 Add golden mask decode tests
- Status: [x]
- Dependencies: P0.3
- Affected files/modules: `src/data/rle_contract.py`, `tests/test_rle_contract.py`, `tests/fixtures/siim_rle_golden_cases.json`, `scripts/validate_siim_rle_contract.py`
- Why it matters: the label pipeline must be protected against silent regressions.
- Subtasks:
  - [x] Curate a minimal set of positive, negative, multi-region, and edge-case annotation examples.
  - [x] Define expected decoded mask properties and, where possible, exact expected outputs.
  - [x] Add a decode verification harness that can be run before regenerating the dataset.
  - [x] Record how these tests are executed in repo memory.
- Success criteria:
  - Decode tests fail on incorrect semantics and pass on accepted semantics.
- Validation needed before close:
  - Label decoding checklist and reproducibility checklist.

### P0.5 Preserve original and dilated masks separately
- Status: [x]
- Dependencies: P0.3
- Affected files/modules: `src/data/preprocess.py`, `src/data/dataset.py`, `src/data/mask_variants.py`, `src/training/trainer.py`, `src/evaluation/evaluate.py`, `configs/config.yaml`, processed data layout
- Why it matters: training on dilated targets may be acceptable as an experiment, but official-mask evaluation requires the original target to remain available.
- Subtasks:
  - [x] Define processed dataset layout for `original_masks` and `dilated_masks` or equivalent variant naming.
  - [x] Define which mask variant is used for training, validation, and final reporting.
  - [x] Decide whether dilation is optional, configurable, or a separate dataset version.
  - [x] Record the scientific implications of training/evaluating on each mask variant.
- Success criteria:
  - Original and dilated masks can be addressed separately and unambiguously.
- Validation needed before close:
  - Mask visualization and image-mask alignment checklist.

### P0.6 Audit and lock the DICOM intensity policy
- Status: [x]
- Dependencies: none
- Affected files/modules: `src/data/preprocess.py`, `src/data/dicom_intensity.py`, `scripts/audit_dicom_intensity.py`, `tests/test_dicom_intensity_policy.py`, raw DICOM metadata handling
- Why it matters: slope/intercept, VOI LUT/windowing, or photometric inversion mistakes can silently create domain shift.
- Subtasks:
  - [x] Inspect raw DICOM metadata on a representative sample.
  - [x] Decide whether current min-max scaling is acceptable or must be replaced by a more medically faithful policy.
  - [x] Record final intensity-processing policy in `DECISIONS.md`.
  - [x] Define visual checks that confirm contrast/orientation sanity after preprocessing.
- Success criteria:
  - DICOM intensity handling is explicitly justified and visually verified.
- Validation needed before close:
  - Image-mask overlay sanity checklist and reproducibility checklist.

### P0.7 Regenerate processed dataset with versioned outputs
- Status: [x]
- Dependencies: P0.3, P0.4, P0.5, P0.6
- Affected files/modules: `src/data/preprocess.py`, `src/data/dataset_manifest.py`, `scripts/validate_processed_dataset.py`, `configs/config.yaml`, `data/processed/pneumothorax_trusted_v1/`
- Why it matters: all future experiments must run on a known-good dataset version.
- Subtasks:
  - [x] Define dataset versioning/fingerprint scheme.
  - [x] Regenerate processed images and both mask variants from raw data.
  - [x] Produce dataset summary statistics: counts, positive ratio, mask sparsity, variant definitions.
  - [x] Snapshot the split manifest and dataset manifest alongside the processed dataset.
- Success criteria:
  - There is a single trusted processed dataset version ready for experiments.
- Validation needed before close:
  - Label, overlay, split leakage, and reproducibility checklist items.

### P1.1 Stratify the train/val/test split
- Status: [x]
- Dependencies: P0.7
- Affected files/modules: `src/data/preprocess.py`, processed split manifest
- Why it matters: the current split has low but unnecessary class-ratio drift and is weaker for publication.
- Subtasks:
  - [x] Define stratification target at image level for positive/negative class balance.
  - [x] Decide whether to preserve current image IDs or regenerate the split.
  - [x] Record the split policy and seed in `DECISIONS.md`.
  - [x] Implement deterministic stratified split generation in `src/data/preprocess.py`.
  - [x] Regenerate `splits.json` under the stratified policy and refresh split fingerprints/manifests.
- Success criteria:
  - Split ratios are stable and leakage-free.
- Validation needed before close:
  - Split leakage checklist.

## Phase 2: Evaluation Correctness

### P0.8 Rewrite Dice / IoU aggregation to be per-image first
- Status: [x]
- Dependencies: P0.7
- Affected files/modules: `src/training/metrics.py`, `src/training/trainer.py`, `src/evaluation/evaluate.py`, `tests/test_metrics_reduction.py`, `tests/test_evaluate_metrics_backend.py`, `tests/test_trainer_validation_aggregation.py`
- Why it matters: current batch-level micro aggregation biases model selection and invalidates comparisons.
- Subtasks:
  - [x] Define canonical reduction modes: per-image mean, positive-only mean, optional micro metrics if needed.
  - [x] Wire evaluator-side per-image reporting to the shared metric backend.
  - [x] Wire trainer-side all-image validation aggregation to the shared metric backend.
  - [x] Prove trainer/evaluator parity on the same saved predictions.
  - [x] Define empty-mask handling explicitly for each metric.
  - [x] Document the primary checkpoint-selection metric.
- Success criteria:
  - Trainer and evaluator compute the same numbers on the same predictions.
- Validation needed before close:
  - Metric correctness and trainer/evaluator parity checklist.

### P0.9 Fix positive-only validation counting
- Status: [x]
- Dependencies: P0.8
- Affected files/modules: `src/training/trainer.py`
- Why it matters: the current positive-only Dice count increments per batch instead of per positive image.
- Subtasks:
  - [x] Change the counting rule to operate on image counts, not batch counts.
  - [x] Confirm the selected metric is the one used for early stopping and checkpoint ranking.
  - [x] Record the metric policy in `DECISIONS.md`.
- Success criteria:
  - Positive-only validation reflects mean performance over positive images.
- Validation needed before close:
  - Metric correctness checklist.

### P1.2 Unify trainer/evaluator output schema
- Status: [x]
- Dependencies: P0.8, P0.9, P0.2
- Affected files/modules: `src/training/trainer.py`, `src/evaluation/evaluate.py`, authoritative run output layout
- Why it matters: current outputs are inconsistent and easy to misinterpret.
- Subtasks:
  - [x] Define mandatory columns and naming for history CSVs and evaluation CSVs.
    - Validation note (2026-04-15): D-036 now fixes the canonical ordered schema for authoritative `metrics/history.csv` and `reports/test_metrics.csv`. The trainer now upgrades legacy resume-history aliases (`val_dice`, `val_dice_pos`, `val_iou`) into canonical `_mean` column names before emitting `history.csv`, the evaluator now writes `test_metrics.csv` through a shared canonical writer, and `py -3 -m unittest tests.test_run_artifacts -v`, `tests.test_evaluation_run_outputs -v`, and `tests.test_trainer_config_surface -v` all passed.
  - [x] Ensure evaluation outputs carry real image IDs and subset tags.
    - Validation note (2026-04-15): D-037 now fixes the immediate per-image evaluation traceability contract: authoritative evaluator rows and qualitative manifest entries must preserve exact dataset `image_id` values and explicit `subset_tag` values (`positive` / `negative`). `py -3 -m unittest tests.test_evaluation_run_outputs -v` and `tests.test_run_artifacts -v` passed after wiring `subset_tag` into per-image report rows and validation/test qualitative manifests.
  - [x] Ensure saved thresholds and mask variants are included in output metadata.
    - Validation note (2026-04-15): D-038 now fixes the final metadata-completeness contract for authoritative evaluation outputs. `selection_state.yaml` now carries `selection_state_path`, `train_mask_variant`, and `eval_mask_variant`; test-summary, qualitative manifests, and per-image evaluation rows now carry the reused `selection_state_path`, `train_mask_variant`, `eval_mask_variant`, `selected_threshold`, and `selected_postprocess`; and test-time selection-state reuse now validates `train_mask_variant` alongside the prior context checks. `py -3 -m unittest tests.test_threshold_selection -v`, `tests.test_evaluation_run_outputs -v`, and `tests.test_run_artifacts -v` all passed.
- Success criteria:
  - Output files are self-explanatory and traceable.
- Validation needed before close:
  - Reproducibility and stale artifact checklist items.

### P1.3 Repair Hausdorff metric or remove it from claims
- Status: [x]
- Dependencies: P0.8
- Affected files/modules: `src/training/metrics.py`, reporting docs
- Why it matters: the implementation/docstring mismatch makes the metric unsafe to report.
- Subtasks:
  - [x] Decide whether to implement the intended metric correctly or drop it from the paper path.
  - [x] Record the decision and rationale in `DECISIONS.md`.
    - Validation note (2026-04-15): D-039 now drops the current `hausdorff` path from authoritative paper-path reporting instead of trying to treat the existing helper as HD95. Authoritative evaluation outputs no longer emit `hausdorff`, and `py -3 -m unittest tests.test_evaluation_run_outputs -v`, `tests.test_run_artifacts -v`, and `tests.test_evaluate_metrics_backend -v` all passed after removing it from the authoritative report schema while leaving the helper non-authoritative.
- Success criteria:
  - No mislabeled metric remains in reported outputs.
- Validation needed before close:
  - Metric correctness checklist.

### P1.4 Add validation-only threshold and post-processing tuning
- Status: [x]
- Dependencies: P0.8, P0.9
- Affected files/modules: `src/evaluation/evaluate.py`, training/evaluation configuration, run metadata
- Why it matters: sparse-mask segmentation often improves materially with threshold and contour/min-area tuning.
- Subtasks:
  - [x] Define search space for threshold and optional post-processing.
  - [x] Decide which validation metric is optimized.
  - [x] Define how the chosen threshold is stored and reused on test.
  - [x] Define rules preventing test-set leakage.
- Success criteria:
  - Threshold selection is reproducible and test-set clean.
- Validation needed before close:
  - Threshold tuning discipline checklist.

### P1.5 Repair config-driven trainer instantiation
- Status: [x]
- Dependencies: P0.2
- Affected files/modules: `src/training/trainer.py`, `configs/config.yaml`
- Why it matters: ablations are unreliable if YAML values do not actually control loss, optimizer, and scheduler behavior.
- Subtasks:
  - [x] Inventory which config fields are currently ignored (`training.optimizer`, `training.scheduler`, `loss.type` are currently ignored by the trainer path).
  - [x] Define which options must be supported immediately versus later.
  - [x] Record the accepted configuration surface in `DECISIONS.md`.
- Success criteria:
  - Changing config changes behavior in a controlled, auditable way.
- Validation needed before close:
  - Reproducibility checklist.

## Phase 3: Strong Supervised Baseline

### P1.6 Replace the weak baseline with a pretrained encoder baseline
- Status: [x]
- Dependencies: P0.7, P0.8, P0.9, P1.4, P1.5
- Affected files/modules: `src/models/`, `src/training/trainer.py`, config surface, experiment outputs
- Why it matters: a plain randomly initialized U-Net is not a competitive paper baseline.
- Subtasks:
  - [x] Select one primary pretrained baseline family for immediate implementation.
  - [x] Define fair training protocol relative to the corrected current baseline.
  - [x] Record why this baseline is the publication anchor.
  - [x] Specify required outputs: tuned validation threshold, test report, qualitative examples.
  - [x] Implement the selected pretrained baseline model path in the current repo stack.
  - [x] Emit authoritative training-side run directory, metadata, config snapshot, history, and best-checkpoint metadata under `artifacts/runs/<run_id>/`.
  - [x] Emit evaluation-side threshold/report/qualitative artifacts under the same authoritative run directory.
    - Validation note (2026-04-11): `C:\Users\beko5\AppData\Local\Programs\Python\Python310\python.exe -m unittest tests.test_evaluation_run_outputs -v`, `tests.test_threshold_selection -v`, `tests.test_run_artifacts -v`, and `tests.test_evaluate_metrics_backend -v` all passed.
  - [x] Execute the first authoritative pretrained baseline run end-to-end on the trusted dataset.
    - Progress note (2026-04-11): `configs/pretrained_resnet34_authoritative.yaml` now locks the fixed D-028 protocol for the first run, `scripts/run_authoritative_pretrained_baseline.py` now provides a Colab-friendly single entrypoint for `train -> select -> test`, and `tests.test_authoritative_pretrained_config -v` plus `tests.test_authoritative_pretrained_runner -v` pass. The current desktop runtime reports `torch 2.11.0+cpu` with `cuda_available=False`, so the actual end-to-end run remains pending on a GPU-capable environment rather than this local machine.
    - Progress note (2026-04-12): the authoritative pretrained runner now also supports `--stage select_test` so a live GPU/Colab run can be stopped after training and then continue safely from the existing `best_checkpoint.pth` through validation threshold selection and held-out test evaluation without accidentally resuming `--stage all`. Targeted regression: `C:\Users\beko5\AppData\Local\Programs\Python\Python310\python.exe -m unittest tests.test_authoritative_pretrained_runner -v` and `tests.test_authoritative_pretrained_config -v` passed.
    - Validation note (2026-04-12): user-reported GPU/Colab run under `/content/drive/MyDrive/foundation_nnunet_runs/resnet34_authoritative_v1` trained through epoch 20, preserved the best checkpoint from epoch 9 (`val_dice_pos_mean=0.5024`), and then completed `--stage select_test`. The authoritative artifact package now includes `selection/selection_state.yaml` with `selected_threshold=0.95`, `reports/test_metrics.csv`, `reports/test_summary.yaml`, `qualitative/validation_samples/`, and `qualitative/test_samples/`. Reported held-out test summary: `1602` images (`357` positive / `1245` negative), positive-only Dice mean `0.4951`.
- Success criteria:
  - At least one strong supervised baseline is reproducible end-to-end on the trusted dataset.
- Validation needed before close:
  - Reproducibility, threshold tuning, and metric correctness checklist items.

### P1.7 Decide whether ROI / crop strategy is required
- Status: [x]
- Dependencies: P1.6
- Affected files/modules: `src/data/preprocess.py`, `src/data/dataset.py`, training configuration
- Why it matters: image-level balancing alone does not solve extreme pixel sparsity.
- Subtasks:
  - [x] Define baseline performance threshold below which crop/ROI work becomes mandatory.
    - Validation note (2026-04-12): D-030 now fixes the `P1.7` gate at held-out `test` positive-only Dice mean `< 0.60` on the first authoritative full-image pretrained baseline. The current trusted full-image baseline reported `0.4951`, so the crop/ROI gate is triggered and a controlled crop/ROI comparison is now mandatory on the critical path.
  - [x] Compare full-image training against a justified crop strategy.
    - Progress note (2026-04-12): the fixed D-031 train-only ROI crop arm is now implemented in `src/data/dataset.py` and wired through `src/training/trainer.py`; `configs/pretrained_resnet34_roi_crop_authoritative.yaml` now locks the crop-comparison protocol; and `py -3 -m unittest tests.test_train_roi_crop_policy -v`, `tests.test_authoritative_pretrained_roi_crop_config -v`, and `tests.test_authoritative_pretrained_runner -v` all passed.
    - Validation note (2026-04-15): user-reported GPU/Colab run under `/content/drive/MyDrive/foundation_nnunet_runs/resnet34_roi_crop_authoritative_v1` was manually stopped after validation collapse, preserved the best checkpoint from epoch 6 (`val_dice_pos_mean=0.4757`), and then completed `--stage select_test`. The authoritative artifact package now includes `selection/selection_state.yaml` with `selected_threshold=0.90`, `reports/test_metrics.csv`, `reports/test_summary.yaml`, `qualitative/validation_samples/`, and `qualitative/test_samples/`. Reported held-out test summary: `1602` images (`357` positive / `1245` negative), positive-only Dice mean `0.4625`, which did not beat the trusted full-image baseline result `0.4951`.
  - [x] Record any crop policy and its leakage constraints in `DECISIONS.md`.
    - Validation note (2026-04-12): D-031 now fixes the immediate `P1.7` comparison arm as a train-only mask-guided `384 x 384` ROI crop for positive train images, matched random `384 x 384` crops for negative train images, resize-back-to-`512` before the model, and full-image `val/test` evaluation with no label-guided eval crop path.
- Success criteria:
  - Either cropping is justified and planned, or full-image training is retained with evidence.
- Validation needed before close:
  - Image-mask alignment, metric correctness, and reproducibility checklist items.

## Phase 4: Hybrid Redesign Decision

### P1.8 Decide whether the current hybrid is worth further investment
- Status: [x]
- Dependencies: P1.6
- Affected files/modules: `src/models/hybrid.py`, `src/models/backbone.py`, methodological framing docs
- Why it matters: the current hybrid is both technically broken and methodologically sensitive.
- Subtasks:
  - [x] Define a keep/drop gate relative to the strong supervised baseline.
    - Validation note (2026-04-15): D-033 now fixes the default `P1.8` state as `defer`, not `keep`. Any future hybrid candidate must first clear the already-open engineering-integrity repairs (`P1.9` through `P1.11`) and then beat the trusted full-image pretrained baseline (`0.4951` held-out `test` positive-only Dice) by at least `+0.02` absolute, i.e. reach `>= 0.5151`, before hybrid work can re-enter the active critical path.
  - [x] Record the evidence required to justify continued hybrid work.
    - Validation note (2026-04-15): D-034 now fixes the minimum hybrid-reopening evidence package: a baseline-gate-equivalent authoritative run directory, an explicit held-out comparison back to the trusted full-image baseline (`0.4951`) and the D-033 keep threshold (`>= 0.5151`), and explicit engineering-integrity proofs for gradient flow, fusion alignment, and branch normalization. `VALIDATION_CHECKLIST.md` now includes a dedicated hybrid keep/drop evidence review section for this contract.
  - [x] Record the paper framing constraints imposed by Foundation X pretraining on SIIM.
    - Validation note (2026-04-15): D-035 now fixes the allowed framing under the current checkpoint provenance: Foundation X may appear only as leakage-aware in-domain transfer / ablation work, not as clean external pretraining or target-unseen generalization on SIIM. `VALIDATION_CHECKLIST.md` now includes a dedicated Foundation X framing review section for this claim boundary.
- Success criteria:
  - There is an explicit keep/drop decision rule rather than open-ended hybrid tuning.
- Validation needed before close:
  - Hybrid decision gate review in `DECISIONS.md`.

### P1.9 Remove incorrect `no_grad` usage and verify gradient flow
- Status: [x]
- Dependencies: P1.8 if hybrid is kept for active work
- Affected files/modules: `src/models/backbone.py`, `src/models/hybrid.py`, training setup
- Why it matters: `frozen=false` currently cannot behave correctly.
- Subtasks:
  - [x] Define intended frozen vs unfrozen semantics.
    - Validation note (2026-04-19): D-050 now fixes the intended contract for `foundation_x.frozen`. Frozen mode must mean `requires_grad=False`, eval-only backbone behavior, and no gradient tracking through the Foundation X path; unfrozen mode must mean `requires_grad=True`, gradient-tracked backbone forward, and no unconditional `torch.no_grad()` wrapper in either `src/models/hybrid.py` or `src/models/backbone.py`. Current code was explicitly inventoried before this decision: both forward paths still hardcode frozen semantics today, so later `P1.9` work remains necessary.
  - [x] Verify optimizer parameter filtering and backbone mode policy.
    - Validation note (2026-04-20): D-051 now records the exact current-state inventory. `src/training/trainer.py::build_optimizer()` already filters on `param.requires_grad`, so frozen backbone params would be excluded correctly once freezing is real; `src/models/hybrid.py::train()` and `src/models/backbone.py::train()` only force `eval()` when the frozen flag is true. The active conflict is higher-level: `src/training/trainer.py` still unconditionally calls `model.foundation_x.backbone.eval()` every train epoch, and both forward paths still hardcode `torch.no_grad()`. So optimizer filtering is not the main blocker; hidden frozen semantics remain the blocker.
    - Validation note (2026-04-20): D-052 now aligns the trainer-side mode policy with D-050/D-051. `src/training/trainer.py` no longer forces `foundation_x.backbone.eval()` in unfrozen mode, and `tests/test_hybrid_backbone_mode_policy.py` now proves the trainer helper keeps frozen backbones in `eval()` while leaving unfrozen backbones in training mode. The remaining `P1.9` blocker is now the unconditional `torch.no_grad()` wrappers in `src/models/hybrid.py` and `src/models/backbone.py`.
  - [x] Add explicit gradient-flow validation for both modes.
    - Validation note (2026-04-20): D-053 now removes the remaining unconditional `torch.no_grad()` wrappers from the Foundation X path: `src/models/backbone.py::FoundationXBackbone.forward()` now gates gradient tracking on `not self.frozen`, and `src/models/hybrid.py::HybridFoundationUNet.forward()` no longer wraps Foundation X extraction in its own `no_grad()` block. `py -3 -m unittest tests.test_hybrid_gradient_flow -v` now passes and proves the intended behavior: frozen mode yields no Foundation X gradients, unfrozen mode yields nonzero Foundation X gradients through the hybrid forward/backward path.
- Success criteria:
  - Frozen mode produces zero backbone gradients; unfrozen mode produces nonzero gradients where expected.
- Validation needed before close:
  - Gradient flow checklist.

### P1.10 Redesign feature fusion mapping if hybrid is kept
- Status: [~]
- Dependencies: P1.8, P1.9
- Affected files/modules: `src/models/hybrid.py`, potentially `src/models/unet.py`, design notes
- Why it matters: current fusion is semantically misaligned across scales.
- Subtasks:
  - [x] Define target stage mapping for 256 and 512 inputs.
    - Validation note (2026-04-20): D-054 now fixes the exact current-state inventory before redesign. The active code maps Foundation X `H/4,H/8,H/16,H/32` stages to U-Net `H,H/2,H/4,H/8` encoder stages (`fx[0]->e1`, `fx[1]->e2`, `fx[2]->e3`, `fx[3]->e4`), so every fusion currently requires a `4x` upsample into a shallower stage. The deepest Foundation X feature is not used at a natural `H/32` context slot; it is upsampled to `H/8` and only then pooled back to `H/16` before the bottleneck.
    - Validation note (2026-04-20): D-055 now fixes the corrected target mapping at the relative-scale level for both accepted input sizes. The intended alignments are `fx[0]->e3`, `fx[1]->e4`, `fx[2]->bottleneck/context(H/16)`, and `fx[3]->dedicated H/32 context slot`, which means no corrected redesign may fuse Foundation X directly into `e1` or `e2`.
  - [ ] Decide whether the model needs a deeper encoder/bottleneck/context head.
  - [ ] Decide how the deepest Foundation X feature is used.
  - [ ] Add explicit shape assertions for all fused stages.
- Success criteria:
  - Fusion is scale-aligned by design and validated with shape checks.
- Validation needed before close:
  - Hybrid scale alignment checklist.

### P1.11 Define hybrid branch normalization policy
- Status: [ ]
- Dependencies: P1.8
- Affected files/modules: `src/data/dataset.py`, `src/models/backbone.py`, hybrid input path
- Why it matters: repeating grayscale `[0,1]` into RGB may not match Foundation X expectations.
- Subtasks:
  - [ ] Identify expected Foundation X input normalization.
  - [ ] Decide whether the baseline and backbone branches should receive different normalized views.
  - [ ] Record final policy in `DECISIONS.md`.
- Success criteria:
  - Hybrid input preprocessing is explicit and defensible.
- Validation needed before close:
  - Reproducibility and hybrid checklist items.

## Phase 5: Paper-Grade Methodology

### P1.12 Define leak-aware Foundation X methodology
- Status: [x]
- Dependencies: P1.8
- Affected files/modules: experiment design docs, final reporting plan
- Why it matters: SIIM exposure in Foundation X pretraining constrains what can be claimed scientifically.
- Subtasks:
  - [x] Decide whether Foundation X is framed as in-domain transfer, ablation-only, or deferred from the paper.
    - Validation note (2026-04-16): D-040 now fixes the final current-state paper role: Foundation X is deferred from the main paper path unless a future hybrid candidate clears D-033 and D-034. Under the current recovered state it may appear only as a future-work / limitations note or a clearly labeled appendix-side leakage-aware ablation, not in the abstract, headline tables, or default storyline. `rg -n "D-040|P1\\.12|Foundation X|deferred from the main paper path" RECOVERY_TODO.md AGENT_CONTEXT.md DECISIONS.md VALIDATION_CHECKLIST.md` and `git diff` were reviewed for consistency.
  - [x] Define forbidden claims under the current setup.
    - Validation note (2026-04-16): D-041 now operationalizes the forbidden-claim list for Foundation X. Under the current setup, wording that presents Foundation X as clean external pretraining, target-unseen generalization, generic foundation-model advantage, or the default superior/main paper model is now explicitly non-authoritative. `rg -n "D-041|forbidden-claim|Foundation X|clean pretraining|target-unseen|default superior model" RECOVERY_TODO.md AGENT_CONTEXT.md DECISIONS.md VALIDATION_CHECKLIST.md` and `git diff` were reviewed for consistency.
  - [x] Record comparison rules versus the strong supervised baseline.
    - Validation note (2026-04-16): D-042 now fixes the mandatory comparison anchor for any future leakage-aware Foundation X discussion: the trusted full-image `pretrained_resnet34_unet` baseline with held-out test positive-only Dice `0.4951`, explicit candidate score, absolute delta, and explicit D-033 threshold status (`>= 0.5151` or not). Crop (`0.4625`), plain U-Net, legacy `results/`, and notebook-only metrics are now disallowed as primary narrative anchors. `rg -n "D-042|0.4951|0.5151|Foundation X|comparison anchor|0.4625" RECOVERY_TODO.md AGENT_CONTEXT.md DECISIONS.md VALIDATION_CHECKLIST.md` and `git diff` were reviewed for consistency.
- Success criteria:
  - The paper claim boundary is explicit and defensible.
- Validation needed before close:
  - Methodology review against `DECISIONS.md`.

### P2.1 Prepare repeated split / cross-validation upgrade path
- Status: [x]
- Dependencies: P1.6, optionally P1.8 if hybrid is kept
- Affected files/modules: evaluation pipeline, experiment orchestration, result aggregation
- Why it matters: single-split results are fragile for publication.
- Subtasks:
  - [x] Decide between repeated stratified splits and 5-fold CV.
    - Validation note (2026-04-16): D-043 now fixes repeated stratified train/val/test splits as the publication-grade evaluation direction. The current trusted pipeline depends on validation-only threshold selection plus held-out test reporting, so plain 5-fold CV was rejected as the primary path because it would require an unsolved nested-validation redesign or would blur validation/test roles. `rg -n "D-043|repeated stratified|5-fold CV|validation-only threshold selection|P2\\.1" RECOVERY_TODO.md AGENT_CONTEXT.md DECISIONS.md VALIDATION_CHECKLIST.md` and `git diff` were reviewed for consistency.
  - [x] Define bootstrap confidence intervals and paired comparison strategy.
    - Validation note (2026-04-16): D-044 now fixes split-bootstrap / paired-delta reporting for the repeated-split path. Confidence intervals are computed over split-level values, not individual images, and model comparisons must pair identical split instances before bootstrapping the mean delta. The default paired comparison target remains held-out `test` positive-only Dice mean. `rg -n "D-044|paired deltas|split-level|bootstrap confidence|P2\\.1" RECOVERY_TODO.md AGENT_CONTEXT.md DECISIONS.md VALIDATION_CHECKLIST.md` and `git diff` were reviewed for consistency.
  - [x] Define minimum evidence package for final reporting.
    - Validation note (2026-04-16): D-045 now fixes the minimum evidence package for final repeated-split reporting: split manifest, per-model-per-split authoritative run packages, split-level aggregation table, paired-delta table, and a final summary artifact with means, 95% split-bootstrap CIs, paired-delta CIs, and contributing split counts. `rg -n "D-045|split manifest|paired-delta table|final summary artifact|P2\\.1" RECOVERY_TODO.md AGENT_CONTEXT.md DECISIONS.md VALIDATION_CHECKLIST.md` and `git diff` were reviewed for consistency.
- Success criteria:
  - Publication-grade evaluation plan is specified and ready to execute.
- Validation needed before close:
  - Reproducibility checklist and methodology review.

### P2.2 Notebook and documentation cleanup
- Status: [x]
- Dependencies: P0.1, P0.2
- Affected files/modules: `notebooks/`, `docs/`, `CLAUDE.md`
- Why it matters: current docs contain assumptions now known to be unsafe or stale.
- Subtasks:
  - [x] Update docs to stop instructing unsafe assumptions about RLE, results, and hybrid behavior.
    - Progress note (2026-04-16): `docs/foundation_nnunet_dev_guide.md` now carries an explicit legacy/non-authoritative warning and redirects readers to the recovery-memory files plus current code/tests. D-046 records that this guide is preserved only as historical design context. Remaining stale operational guidance still needs cleanup in active docs such as `CLAUDE.md`.
    - Progress note (2026-04-16): `CLAUDE.md` now also carries an explicit legacy/non-authoritative warning and redirects readers to the recovery-memory files plus current code/tests. D-047 records that the file is preserved only as historical operational context and must not outrank the recovered methodology. Remaining stale operational guidance is now concentrated in the legacy training notebooks.
  - [x] Mark notebook limitations and authoritative usage rules.
    - Progress note (2026-04-16): `notebooks/train_colab.ipynb` now opens with an explicit legacy/non-authoritative warning and redirects readers to the recovery-memory files plus current code/tests. D-048 records that this notebook is preserved only as historical Colab context and must not outrank the recovered methodology. Remaining notebook cleanup is now centered on `notebooks/train_local.ipynb`.
    - Progress note (2026-04-19): `notebooks/train_local.ipynb` now also opens with an explicit legacy/non-authoritative warning and redirects readers to the recovery-memory files plus current code/tests. D-049 records that this notebook is preserved only as historical local training context and must not outrank the recovered methodology. This closes the remaining notebook-side stale workflow cleanup inside `P2.2`.
- Success criteria:
  - Documentation no longer conflicts with recovery decisions.
- Validation needed before close:
  - Stale artifact and methodology checklist items.

## Top priority queue

1. P1.10 Redesign feature fusion mapping if hybrid is kept
2. P1.11 Define hybrid branch normalization policy
