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
- Status: [ ]
- Dependencies: P0.7
- Affected files/modules: `src/data/preprocess.py`, processed split manifest
- Why it matters: the current split has low but unnecessary class-ratio drift and is weaker for publication.
- Subtasks:
  - [ ] Define stratification target at image level for positive/negative class balance.
  - [ ] Decide whether to preserve current image IDs or regenerate the split.
  - [ ] Record the split policy and seed in `DECISIONS.md`.
- Success criteria:
  - Split ratios are stable and leakage-free.
- Validation needed before close:
  - Split leakage checklist.

## Phase 2: Evaluation Correctness

### P0.8 Rewrite Dice / IoU aggregation to be per-image first
- Status: [~]
- Dependencies: P0.7
- Affected files/modules: `src/training/metrics.py`, `src/training/trainer.py`, `src/evaluation/evaluate.py`, `tests/test_metrics_reduction.py`, `tests/test_evaluate_metrics_backend.py`
- Why it matters: current batch-level micro aggregation biases model selection and invalidates comparisons.
- Subtasks:
  - [x] Define canonical reduction modes: per-image mean, positive-only mean, optional micro metrics if needed.
  - [x] Wire evaluator-side per-image reporting to the shared metric backend.
  - [ ] Wire trainer-side validation aggregation to the shared metric backend.
  - [x] Define empty-mask handling explicitly for each metric.
  - [x] Document the primary checkpoint-selection metric.
- Success criteria:
  - Trainer and evaluator compute the same numbers on the same predictions.
- Validation needed before close:
  - Metric correctness and trainer/evaluator parity checklist.

### P0.9 Fix positive-only validation counting
- Status: [ ]
- Dependencies: P0.8
- Affected files/modules: `src/training/trainer.py`
- Why it matters: the current positive-only Dice count increments per batch instead of per positive image.
- Subtasks:
  - [ ] Change the counting rule to operate on image counts, not batch counts.
  - [ ] Confirm the selected metric is the one used for early stopping and checkpoint ranking.
  - [ ] Record the metric policy in `DECISIONS.md`.
- Success criteria:
  - Positive-only validation reflects mean performance over positive images.
- Validation needed before close:
  - Metric correctness checklist.

### P1.2 Unify trainer/evaluator output schema
- Status: [ ]
- Dependencies: P0.8, P0.9, P0.2
- Affected files/modules: `src/training/trainer.py`, `src/evaluation/evaluate.py`, authoritative run output layout
- Why it matters: current outputs are inconsistent and easy to misinterpret.
- Subtasks:
  - [ ] Define mandatory columns and naming for history CSVs and evaluation CSVs.
  - [ ] Ensure evaluation outputs carry real image IDs and subset tags.
  - [ ] Ensure saved thresholds and mask variants are included in output metadata.
- Success criteria:
  - Output files are self-explanatory and traceable.
- Validation needed before close:
  - Reproducibility and stale artifact checklist items.

### P1.3 Repair Hausdorff metric or remove it from claims
- Status: [ ]
- Dependencies: P0.8
- Affected files/modules: `src/training/metrics.py`, reporting docs
- Why it matters: the implementation/docstring mismatch makes the metric unsafe to report.
- Subtasks:
  - [ ] Decide whether to implement the intended metric correctly or drop it from the paper path.
  - [ ] Record the decision and rationale in `DECISIONS.md`.
- Success criteria:
  - No mislabeled metric remains in reported outputs.
- Validation needed before close:
  - Metric correctness checklist.

### P1.4 Add validation-only threshold and post-processing tuning
- Status: [ ]
- Dependencies: P0.8, P0.9
- Affected files/modules: `src/evaluation/evaluate.py`, training/evaluation configuration, run metadata
- Why it matters: sparse-mask segmentation often improves materially with threshold and contour/min-area tuning.
- Subtasks:
  - [ ] Define search space for threshold and optional post-processing.
  - [ ] Decide which validation metric is optimized.
  - [ ] Define how the chosen threshold is stored and reused on test.
  - [ ] Define rules preventing test-set leakage.
- Success criteria:
  - Threshold selection is reproducible and test-set clean.
- Validation needed before close:
  - Threshold tuning discipline checklist.

### P1.5 Repair config-driven trainer instantiation
- Status: [ ]
- Dependencies: P0.2
- Affected files/modules: `src/training/trainer.py`, `configs/config.yaml`
- Why it matters: ablations are unreliable if YAML values do not actually control loss, optimizer, and scheduler behavior.
- Subtasks:
  - [ ] Inventory which config fields are currently ignored.
  - [ ] Define which options must be supported immediately versus later.
  - [ ] Record the accepted configuration surface in `DECISIONS.md`.
- Success criteria:
  - Changing config changes behavior in a controlled, auditable way.
- Validation needed before close:
  - Reproducibility checklist.

## Phase 3: Strong Supervised Baseline

### P1.6 Replace the weak baseline with a pretrained encoder baseline
- Status: [ ]
- Dependencies: P0.7, P0.8, P0.9, P1.4, P1.5
- Affected files/modules: `src/models/`, `src/training/trainer.py`, config surface, experiment outputs
- Why it matters: a plain randomly initialized U-Net is not a competitive paper baseline.
- Subtasks:
  - [ ] Select one primary pretrained baseline family for immediate implementation.
  - [ ] Define fair training protocol relative to the corrected current baseline.
  - [ ] Record why this baseline is the publication anchor.
  - [ ] Specify required outputs: tuned validation threshold, test report, qualitative examples.
- Success criteria:
  - At least one strong supervised baseline is reproducible end-to-end on the trusted dataset.
- Validation needed before close:
  - Reproducibility, threshold tuning, and metric correctness checklist items.

### P1.7 Decide whether ROI / crop strategy is required
- Status: [ ]
- Dependencies: P1.6
- Affected files/modules: `src/data/preprocess.py`, `src/data/dataset.py`, training configuration
- Why it matters: image-level balancing alone does not solve extreme pixel sparsity.
- Subtasks:
  - [ ] Define baseline performance threshold below which crop/ROI work becomes mandatory.
  - [ ] Compare full-image training against a justified crop strategy.
  - [ ] Record any crop policy and its leakage constraints in `DECISIONS.md`.
- Success criteria:
  - Either cropping is justified and planned, or full-image training is retained with evidence.
- Validation needed before close:
  - Image-mask alignment, metric correctness, and reproducibility checklist items.

## Phase 4: Hybrid Redesign Decision

### P1.8 Decide whether the current hybrid is worth further investment
- Status: [ ]
- Dependencies: P1.6
- Affected files/modules: `src/models/hybrid.py`, `src/models/backbone.py`, methodological framing docs
- Why it matters: the current hybrid is both technically broken and methodologically sensitive.
- Subtasks:
  - [ ] Define a keep/drop gate relative to the strong supervised baseline.
  - [ ] Record the evidence required to justify continued hybrid work.
  - [ ] Record the paper framing constraints imposed by Foundation X pretraining on SIIM.
- Success criteria:
  - There is an explicit keep/drop decision rule rather than open-ended hybrid tuning.
- Validation needed before close:
  - Hybrid decision gate review in `DECISIONS.md`.

### P1.9 Remove incorrect `no_grad` usage and verify gradient flow
- Status: [ ]
- Dependencies: P1.8 if hybrid is kept for active work
- Affected files/modules: `src/models/backbone.py`, `src/models/hybrid.py`, training setup
- Why it matters: `frozen=false` currently cannot behave correctly.
- Subtasks:
  - [ ] Define intended frozen vs unfrozen semantics.
  - [ ] Verify optimizer parameter filtering and backbone mode policy.
  - [ ] Add explicit gradient-flow validation for both modes.
- Success criteria:
  - Frozen mode produces zero backbone gradients; unfrozen mode produces nonzero gradients where expected.
- Validation needed before close:
  - Gradient flow checklist.

### P1.10 Redesign feature fusion mapping if hybrid is kept
- Status: [ ]
- Dependencies: P1.8, P1.9
- Affected files/modules: `src/models/hybrid.py`, potentially `src/models/unet.py`, design notes
- Why it matters: current fusion is semantically misaligned across scales.
- Subtasks:
  - [ ] Define target stage mapping for 256 and 512 inputs.
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
- Status: [ ]
- Dependencies: P1.8
- Affected files/modules: experiment design docs, final reporting plan
- Why it matters: SIIM exposure in Foundation X pretraining constrains what can be claimed scientifically.
- Subtasks:
  - [ ] Decide whether Foundation X is framed as in-domain transfer, ablation-only, or deferred from the paper.
  - [ ] Define forbidden claims under the current setup.
  - [ ] Record comparison rules versus the strong supervised baseline.
- Success criteria:
  - The paper claim boundary is explicit and defensible.
- Validation needed before close:
  - Methodology review against `DECISIONS.md`.

### P2.1 Prepare repeated split / cross-validation upgrade path
- Status: [ ]
- Dependencies: P1.6, optionally P1.8 if hybrid is kept
- Affected files/modules: evaluation pipeline, experiment orchestration, result aggregation
- Why it matters: single-split results are fragile for publication.
- Subtasks:
  - [ ] Decide between repeated stratified splits and 5-fold CV.
  - [ ] Define bootstrap confidence intervals and paired comparison strategy.
  - [ ] Define minimum evidence package for final reporting.
- Success criteria:
  - Publication-grade evaluation plan is specified and ready to execute.
- Validation needed before close:
  - Reproducibility checklist and methodology review.

### P2.2 Notebook and documentation cleanup
- Status: [ ]
- Dependencies: P0.1, P0.2
- Affected files/modules: `notebooks/`, `docs/`, `CLAUDE.md`
- Why it matters: current docs contain assumptions now known to be unsafe or stale.
- Subtasks:
  - [ ] Update docs to stop instructing unsafe assumptions about RLE, results, and hybrid behavior.
  - [ ] Mark notebook limitations and authoritative usage rules.
- Success criteria:
  - Documentation no longer conflicts with recovery decisions.
- Validation needed before close:
  - Stale artifact and methodology checklist items.

## Top priority queue

1. P0.8 Rewrite per-image metrics
2. P0.9 Fix positive-only validation counting
3. P1.1 Stratify the train/val/test split
4. P1.6 Build strong pretrained baseline after trust gates pass
5. P1.7 Decide whether ROI / crop strategy is required
6. P1.8 Decide whether the current hybrid is worth further investment
7. P1.5 Repair config-driven trainer instantiation
8. P1.4 Add validation-only threshold and post-processing tuning
9. P1.2 Unify trainer/evaluator output schema
10. P1.3 Repair Hausdorff metric or remove it from claims
