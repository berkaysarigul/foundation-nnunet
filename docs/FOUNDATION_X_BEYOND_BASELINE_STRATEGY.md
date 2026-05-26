# Foundation X Beyond Baseline Strategy

> Status: living strategy document. Authoritative project memory remains in
> `RECOVERY_TODO.md`, `AGENT_CONTEXT.md`, `DECISIONS.md`, and
> `VALIDATION_CHECKLIST.md`. This document does **not** override those files;
> it only proposes a research direction on top of them.

---

## 1. Executive Summary

### 1.1 What this document is

A research-direction proposal for evolving the Foundation-nnU-Net project from
its current state (a deferred-from-paper Foundation X feature-transfer
diagnostic) into a system that can credibly compete with both:

- **A**: our own pure / pretrained CNN baseline, and
- **B**: the segmentation performance reported by the original Foundation X
  paper on SIIM-ACR pneumothorax,

under a fair, reproducible, leakage-aware evaluation protocol.

### 1.2 Where we actually stand right now

Two parallel research tracks exist:

1. **Recovery / paper-anchor track (`pneumothorax_trusted_v1`):**
   - Trusted processed dataset, corrected RLE contract (D-016), corrected
     metrics (D-020/D-011), validation-only threshold selection (D-025/D-026),
     authoritative run package (D-029).
   - Single-run anchor: `pretrained_resnet34_unet` at held-out positive-only
     Dice mean **0.4951** (D-042).
   - 3-split repeated-split pilot (`resnet34_repeated_split_pilot_v1`,
     D-071): mean **0.5058**, 95% split-bootstrap CI **[0.4911, 0.5196]**.
   - Foundation X / hybrid is **deferred** from this track (D-033/D-040).
     Re-entry gate: held-out positive-only Dice mean ≥ **0.5151** plus full
     D-034 evidence package, framed as leakage-aware secondary work
     (D-035/D-041).

2. **Dataset101 / nnU-Net v2 diagnostic track:**
   - `nnUNet_raw/Dataset101_Pneumothorax/` is a read-only nnU-Net v2 export
     (PR-2) used as the substrate for the Foundation X smoke (PR-4),
     hybrid adapter sanity (PR-5), tiny hybrid training (PR-6), BN-mode bug
     fix (PR-7D), controlled short hybrid training (PR-7/PR-8), and
     inference-only heldout evaluation (PR-9).
   - User-reported PR-8 GPU/Colab run: hybrid best step ≈ 800, validation
     `dice_pos_mean_thr_050` ≈ **0.4516**; final step 1200 ≈ **0.4397**;
     no persistent collapse; frozen Foundation X verified frozen; train/eval
     mode preserved.
   - PR-9 (`scripts/evaluate_hybrid_heldout.py`) exists but only has a
     4-case dryrun artifact locally; a real GPU/Colab heldout evaluation
     has not yet been recorded in repo memory.

User-supplied pure nnU-Net heldout numbers (1602 cases): case-level
precision **0.7719**, recall **0.6919**, F1 **0.7297**, mean positive
Dice **0.3722**, detected-positive Dice **0.5380**, negative FPR
**0.0586**. These come from the Dataset101 heldout track and use the
nnU-Net v2 evaluator's case-level definitions; they are not directly
comparable to the 0.4951 figure on `pneumothorax_trusted_v1`.

### 1.3 Why current hybrid scores are not Foundation X paper scores

Our `HybridFoundationUNet` uses Foundation X **only as a frozen Swin-B
feature extractor**. The checkpoint loads cleanly but reports **124
unexpected keys** (PR-4 smoke summary), consistent with the classification,
localization, and segmentation heads of the full Foundation X model not
being instantiated by our backbone-only wrapper. Our hybrid then trains a
custom U-Net decoder + multi-scale fusion. This is materially different
from running Foundation X's own multi-task, lock-release, cyclic-pretrained,
EMA student-teacher segmentation framework end-to-end.

Therefore: **the Foundation X paper's SIIM segmentation number does not
benchmark our system today, and our hybrid number does not refute the
paper.** Any "beat Foundation X" claim must first reproduce or approximate
Foundation X direct segmentation on our heldout split.

### 1.4 The proposed direction

1. Pause and instrument before optimizing. Run authoritative heldout
   evaluation of (a) the current hybrid PR-8 best checkpoint and (b) the
   trusted pretrained baseline on the **same** evaluation surface.
2. Determine whether Foundation X *direct* segmentation can be reproduced
   from the available checkpoint and (if necessary) the original repository.
3. If direct reproduction is feasible, use Foundation X direct predictions
   as a **mask prior** consumed by an nnU-Net / refiner model — this is the
   shape originally proposed in the TÜBİTAK 2209-A application.
4. Strengthen supervision via CANDID-PTX and optionally weakly supervised
   auxiliary datasets, only after the cascade architecture has been
   demonstrated on SIIM.
5. Report results with split-bootstrap CIs and paired deltas per D-043 to
   D-045, always against the trusted pretrained baseline (D-042) and the
   reproduced Foundation X direct score (if obtained).

---

## 2. Current Project State

### 2.1 Authoritative track (recovery / paper anchor)

| Component | Status | Source of truth |
| --- | --- | --- |
| Trusted processed dataset | `pneumothorax_trusted_v1` (10,675 images, 2,379 positive, 8,296 negative) | `data/processed/pneumothorax_trusted_v1/dataset_manifest.json` (D-019) |
| RLE contract | `cumulative_gap_pairs`, Fortran order, `-1` empty | D-016 |
| Splits | Stratified 70/15/15, seed 42, two-stage `train_test_split` | D-023 |
| DICOM intensity policy | Native 8-bit, modality/VOI applied only if present | D-018 |
| Metrics | Per-image Dice/IoU; `val_dice_pos_mean` is selection metric | D-011, D-020 |
| Threshold selection | Validation-only, grid 0.05–0.95, `postprocess=none` | D-025, D-026 |
| Output schema | `history.csv` and `reports/test_metrics.csv` canonical columns; no `hausdorff` | D-036, D-037, D-038, D-039 |
| First baseline anchor | `pretrained_resnet34_unet`, held-out positive-only Dice **0.4951** | D-042 (run `resnet34_authoritative_v1`) |
| ROI/crop comparison | Train-only 384×384 ROI, full-image eval; result **0.4625** (worse) | D-031, D-032 |
| Repeated-split pilot | 3 splits × `pretrained_resnet34_unet`: mean **0.5058**, CI [0.4911, 0.5196] | D-071, D-072 |
| Hybrid posture | Deferred. Keep gate ≥ **0.5151** + D-034 evidence; D-035 claim boundary | D-033, D-034, D-035 |

### 2.2 Active diagnostic track (Dataset101 / nnU-Net v2)

| PR | Goal | Key result | Artifact |
| --- | --- | --- | --- |
| PR-2 | nnU-Net v2 export | `nnUNet_raw/Dataset101_Pneumothorax/`: 1602 imagesTs + heldout_labelsTs | `scripts/export_siim_to_nnunet.py` |
| PR-4 | Foundation X smoke | PASS; 449 keys loaded, 0 missing, **124 unexpected**, finite feature maps at all 4 scales | `artifacts/diagnostics/foundation_x_smoke/` |
| PR-5 | Hybrid adapter sanity | PASS; all fusion shapes match D-055/D-056/D-057; gradients route correctly through trainable decoder, frozen backbone confirmed | `artifacts/diagnostics/hybrid_adapter_sanity/` |
| PR-6 | Tiny hybrid training smoke | PASS; finite loss, gradients flow, no NaN/Inf | `artifacts/diagnostics/hybrid_tiny_train_smoke/` |
| PR-7 (lr3e5_50step_bnfix) | Controlled short training, BN mode fix | Loss decreasing 1.797 → 1.577 over 50 steps; foreground ratio collapsing toward 0; val Dice still <0.03 | `artifacts/diagnostics/hybrid_controlled_short_train_lr3e5_50step_bnfix/` |
| PR-7D | BN-mode bug discovery + fix | D-052/D-053: trainer no longer forces backbone `eval()` unconditionally; `torch.no_grad()` removed from forward path; eval()-only when frozen | `tests/test_hybrid_backbone_mode_policy.py`, `tests/test_hybrid_gradient_flow.py` |
| PR-7F | Final visualization scaffold | inference-only utility | `scripts/visualize_pr7f_final.py` |
| PR-8 | Larger controlled training (user-reported, GPU/Colab) | Best step ≈ 800, val `dice_pos_mean_thr_050` ≈ **0.4516**; step 1200 ≈ **0.4397**; no collapse; backbone frozen verified | local dryrun artifact only; real run reported by user, not yet in repo memory |
| PR-9 dryrun | Heldout eval skeleton | 4-case dryrun; trivial all-foreground predictions because 4 negatives only | `artifacts/diagnostics/hybrid_heldout_eval/dryrun_thr05/` |

### 2.3 The gap between the two tracks

The authoritative recovery track has been built around
`pneumothorax_trusted_v1`. The hybrid / Foundation X diagnostic track has
been built around the `nnUNet_raw/Dataset101_Pneumothorax/` export. They
share the same underlying SIIM-ACR data but **differ in image IDs, mask
target definition (`original_masks` vs Dataset101 binary labels), split
identity, and evaluator code paths**. No authoritative run currently
exists that produces a directly comparable score for both the pretrained
baseline and the hybrid on the same evaluation substrate.

This gap is the most important obstacle to any "hybrid beats baseline"
claim and is addressed in Section 5.

### 2.4 The current HybridFoundationUNet design (as built)

- **Backbone**: `FoundationXBackbone(checkpoint, frozen=True, img_size=512)`
  loads `checkpoints/foundation_x.pth` (SHA-256
  `9b7eb0215b8fdb50d1413fee306e70ed8ac7b646cff342dbce78ca44fd0f32c3`,
  ≈ 2.82 GB), strips the `backbone.0.` prefix, remaps `layers.N.* →
  layers_N.*` for timm, builds `swin_base_patch4_window7_224` with
  `features_only=True, out_indices=(0,1,2,3)`. Inputs are grayscale
  `[0,1]` → repeat to RGB → ImageNet mean/std (D-062/D-070).
- **Output features**: 4 stages with channels `(128, 256, 512, 1024)` at
  resolutions `(H/4, H/8, H/16, H/32)`.
- **Fusion (D-055/D-056/D-057/D-059)**:
  - `fx[0] → e3` (H/4)
  - `fx[1] → e4` (H/8)
  - `fx[2] → H/16 context` (with pooled `e4`)
  - `fx[3] → H/32 context head → ConvTranspose2d 2× → H/16` reconnect
  - context_merge → decoder consumes `fused_e4`, `fused_e3`, `e2`, `e1`
- **Mode policy (D-050/D-052/D-053)**: when `frozen_backbone=True`,
  Foundation X stays in `eval()` and contributes no gradients; when
  unfrozen, neither the trainer nor the forward path silently overrides
  this. (Validated by `tests/test_hybrid_gradient_flow.py`.)

### 2.5 Known limitations of the current hybrid

1. **Foundation X used only as features** — no segmentation head from the
   paper is loaded or run.
2. **No fair head-to-head** — the only authoritative pretrained ResNet34
   baseline lives on `pneumothorax_trusted_v1`, not on the Dataset101
   heldout that the hybrid was trained on.
3. **Frozen backbone** — D-053 fixed the silent `no_grad()`, but all
   training to date has kept the backbone frozen, so the hybrid has not
   yet tested partial unfreezing.
4. **Train batch size 1 on CPU dry-runs** — the local PR-7 dry-runs are
   artifacts of running with batch=1, num_train_cases=16, max_steps=5;
   the only meaningful learning curve is the user-reported PR-8 GPU run.
5. **Validation set is tiny** in the diagnostic runs (8–16 cases) and is
   not the publication validation split; the reported 0.4516 should be
   read as a *diagnostic* number, not a publication-grade metric.
6. **Dilation/training mask variant** for the Dataset101 hybrid is not
   the same contract as `pneumothorax_trusted_v1`'s
   `train_mask_variant=dilated_masks, eval_mask_variant=original_masks`
   policy (D-017).

---

## 3. What We Have Actually Built vs What Foundation X Paper Reports

### 3.1 The original Foundation X (paper) at a glance

From the Foundation X paper (Foundation_X_Integrating_Classification_
Localization_and_Segmentation_Through_Lock-Release_Pretraining_Strategy
_for_Chest_X-Ray_Analysis):

- **Backbone**: Swin-B, shared across tasks.
- **Branches / heads**:
  - Classification heads ("Cyclic" — multiple datasets, multiple labels)
  - Localization heads (bounding-box / heatmap)
  - Segmentation heads (per dataset, e.g. **S3** is the SIIM-ACR
    pneumothorax segmentation head)
- **Pretraining strategy**: Lock-Release alternation across tasks and
  datasets, with cyclic exposure; student-teacher / EMA stabilization to
  prevent catastrophic forgetting across the rolling task schedule.
- **Reported SIIM-ACR segmentation performance** (paper): substantially
  higher than a standard segmentation baseline. Exact numbers are in
  Table 2 of the paper; we do not restate them here because they are
  produced under the full Foundation X framework, not under our wrapper.

### 3.2 Our `HybridFoundationUNet` at a glance

- **Backbone**: same Swin-B, loaded from the same `foundation_x.pth`, but
  loaded **as features only** (`timm.create_model(...,
  features_only=True)`).
- **Branches / heads**:
  - **Not loaded**: classification heads, localization heads, all
    segmentation heads (S3 included).
  - 124 unexpected keys at load (PR-4 smoke) are consistent with the
    above — these are the heads / branches our wrapper does not
    instantiate.
- **Decoder**: a custom U-Net decoder we built on top of the frozen
  features, with explicit multi-scale fusion (D-055–D-059).
- **Training data**: only SIIM (Dataset101 for the diagnostic track, or
  `pneumothorax_trusted_v1` for an eventual D-034 evidence run).
- **Training strategy**: standard mini-batch SGD/AdamW with Dice+Focal
  (or Dice+BCE) loss; no lock-release schedule; no EMA; no auxiliary
  tasks.

### 3.3 Side-by-side comparison

| Component | Foundation X paper | Our HybridFoundationUNet |
| --- | --- | --- |
| Swin-B backbone | Yes | Yes (frozen by default) |
| SIIM segmentation head (S3) | Yes | **No** |
| Auxiliary classification heads | Yes | **No** |
| Auxiliary localization heads | Yes | **No** |
| Lock-Release pretraining | Yes | **No** (we use the released checkpoint as-is) |
| Cyclic multi-dataset training | Yes | **No** (SIIM only) |
| Student-teacher / EMA | Yes | **No** |
| Custom decoder | (paper uses its own segmentation decoder) | **Yes** (our U-Net decoder + fusion) |
| Decoder is trained on SIIM target | Yes | Yes |
| Input normalization | (paper-specified) | Grayscale→RGB + ImageNet mean/std (D-062/D-070) |
| Reported SIIM Dice context | Paper's own framework end-to-end | Diagnostic / non-publication runs so far |

The two systems share only the backbone weights. **Calling our system
"Foundation X" in any comparison would be inaccurate.** It is a Foundation
X feature-transfer hybrid.

---

## 4. Why Current Hybrid Scores Are Lower Than Foundation X Paper Scores

### 4.1 The headline differences

1. **Missing segmentation head**: Foundation X's S3 head was trained
   jointly with the backbone under the lock-release schedule. Replacing it
   with a freshly initialized decoder trained from scratch on SIIM alone
   gives up the multi-task representation pressure that the paper's
   decoder benefits from.
2. **Frozen backbone**: in all our hybrid runs so far, the Swin-B
   backbone is frozen. The paper's number reflects an end-to-end trained
   system, including the backbone in the segmentation regime.
3. **No multi-task supervision**: the paper's pretraining used CXR
   classification (multiple datasets), localization, and multiple
   segmentation targets. Our hybrid only sees SIIM pneumothorax masks.
4. **No EMA / student-teacher stabilization**: the paper uses these to
   manage catastrophic forgetting across the cyclic schedule; they also
   tend to give a small final accuracy bump.
5. **No cyclic exposure during fine-tuning** — only standard supervised
   training.
6. **Evaluation surfaces differ**: our hybrid was trained / validated on
   Dataset101 nnU-Net export with a small diagnostic validation set;
   neither preprocessing nor evaluation matches what the paper reports.
7. **Threshold selection**: the paper's number presumably uses a fixed or
   internally-selected threshold; ours is currently fixed at 0.5 for
   diagnostic runs.

### 4.2 The leakage caveat

D-006 records that Foundation X pretraining is **SIIM-exposed**: SIIM
appears in the Foundation X pretraining corpus. This is why D-035 limits
how we can frame Foundation X under the current checkpoint provenance.
Under this constraint:

- The Foundation X paper's reported SIIM number must be interpreted
  cautiously even within its own framework, because the same data was
  seen during the cyclic pretraining schedule.
- Our hybrid's number, even with the same backbone, **cannot** be claimed
  as evidence of clean external transfer (D-041).
- Any comparison "we beat Foundation X" requires both numbers measured on
  the same heldout split with the same evaluator and the same target
  definition.

### 4.3 What the PR-8 diagnostic number actually tells us

The user-reported PR-8 result (val `dice_pos_mean_thr_050` ≈ 0.4516 at
best step 800) tells us:

- The hybrid is **trainable** (no collapse, no NaN/Inf, finite gradients).
- The hybrid produces **non-trivial pneumothorax overlap** on positive
  validation cases, well above the early-step "all-foreground 0.029" floor
  seen in the 50-step BN-fix run.
- The PR-7D BN-mode and gradient fixes (D-052/D-053) are not preventing
  learning.

It does **not** yet tell us:

- Whether the hybrid beats the trusted pretrained ResNet34 baseline on
  the **same** evaluation substrate.
- Whether the hybrid beats Foundation X's reported direct segmentation
  score on the **same** heldout.
- Whether the hybrid generalizes to truly heldout data with the case-level
  precision/recall properties needed for clinical relevance.

---

## 5. Fair Evaluation Framework

### 5.1 Anchor selection

The recovery track has already fixed the comparison anchor:

- **Single-run anchor (D-042):** trusted full-image `pretrained_resnet34_unet`,
  held-out positive-only Dice mean **0.4951**, threshold 0.95,
  `eval_mask_variant=original_masks`, on `pneumothorax_trusted_v1`.
- **Interim repeated-split anchor (D-071/D-072):** the same model across
  `split_001`, `split_002`, `split_003`, mean **0.5058**, 95% CI
  **[0.4911, 0.5196]**. Reported as pilot/interim, not publication-final.
- **Pure nnU-Net baseline on Dataset101 heldout:** case-level numbers
  supplied by the user (precision **0.7719**, recall **0.6919**, F1
  **0.7297**, mean positive Dice **0.3722**, detected-positive Dice
  **0.5380**, negative FPR **0.0586**). This is **a separate evaluation
  surface** — it complements but does not replace the trusted anchor.

### 5.2 The evaluation ladder

| Model | Substrate | What it tells us |
| --- | --- | --- |
| **A** — Pure nnU-Net v2 (already done) | Dataset101 heldout | Bench upper end for a standard supervised pipeline on this exact heldout, with case-level statistics |
| **A'** — Pretrained ResNet34 U-Net (already done) | `pneumothorax_trusted_v1` test | Authoritative D-042 anchor 0.4951 |
| **B** — Foundation X direct segmentation (NOT done) | Same heldout as A and/or A' | The score we must reproduce before claiming we beat it (Section 7, Option 1) |
| **C** — Current HybridFoundationUNet PR-8 best | Same heldout as A and/or A' | The actual current contribution; required before deciding whether to keep working on this design |
| **D** — Foundation X direct + nnU-Net/refiner cascade | Same heldout | The TÜBİTAK 2209-A core idea: coarse prior + precise refiner |
| **E** — Extended hybrid (multi-source / partial unfreeze / distillation) | Same heldout | Realistic best-case if all of the above is set up correctly |

### 5.3 What "same heldout" means in practice

To make the ladder comparable, we need exactly one of the following two
options, chosen explicitly:

- **Option α — Dataset101 anchor:** evaluate everything (A, B, C, D, E)
  on `nnUNet_raw/Dataset101_Pneumothorax/imagesTs` against
  `heldout_labelsTs`. This is the substrate the hybrid was actually
  trained on and aligns with the pure nnU-Net case-level numbers
  already in hand. Implication: we must run the
  `pretrained_resnet34_unet` on Dataset101 heldout to get a matched
  arm, since D-042's 0.4951 is on a different split.
- **Option β — `pneumothorax_trusted_v1` anchor:** retrain (or
  fine-tune) the hybrid on the trusted dataset/split and evaluate against
  the trusted test split. This preserves D-042 but invalidates the PR-8
  best checkpoint as a comparison artifact — it was not trained on this
  substrate.

The recommended choice is **Option β** for publication-grade reporting
(D-033, D-034 require it) and **Option α** for the immediate PR-9
diagnostic snapshot. Both should be documented; neither should be silently
swapped.

### 5.4 Rules that must hold across the ladder

1. Same dataset root, split fingerprint, image preprocessing, and mask
   target definition for all arms.
2. Validation-only threshold selection per D-025/D-026; the same
   `selection_state.yaml` discipline applied to every learned-decoder arm.
3. Per-image positive-only Dice (`val_dice_pos_mean`) as the primary
   metric (D-011); case-level statistics reported as well for clinical
   interpretation (Section 10).
4. Repeated-split CIs and paired deltas where applicable (D-043, D-044,
   D-045).
5. No threshold tuning on test (D-025).
6. Authoritative run-artifact package (D-029) for any arm whose score is
   cited in the report.
7. Foundation X framing per D-035/D-040–D-042; leakage-aware secondary
   evidence only.

---

## 6. Required Baselines

To make the document actionable, here is the minimum set of authoritative
runs that must exist before any new architecture work is justified.

### 6.1 Already in repo memory

- `resnet34_authoritative_v1` (D-042): single-run trusted anchor.
- `resnet34_repeated_split_pilot_v1` (D-071): 3-split pilot.
- Pure nnU-Net v2 case-level numbers on Dataset101 heldout (user-supplied,
  not yet rendered into a per-image CSV in repo memory).

### 6.2 Still needed before hybrid claims become meaningful

1. **PR-9 authoritative heldout eval of PR-8 hybrid best** on Dataset101
   heldout (real GPU/Colab run, not the 4-case dryrun). Output: positive
   Dice mean, IoU, case-level precision/recall/F1, FPR, tiny/medium/large
   subgroup Dice (Section 10), per-case CSV, qualitative grids.
   Acceptance: report the candidate score vs. nnU-Net baseline numbers
   without claiming it beats the D-042 anchor (different substrate).
2. **Matched pretrained ResNet34 arm on Dataset101 heldout** (or matched
   hybrid arm on `pneumothorax_trusted_v1`) — whichever choice is made in
   Section 5.3, both arms must exist on the same substrate.
3. **Foundation X direct segmentation reproduction**, if feasible
   (Section 7, Option 1). Without this, Section 1.2's "B" entry remains
   open and "we beat Foundation X" claims remain unsupported.

---

## 7. Proposed Architectures to Beat Foundation X

These options are ranked by realism × scientific value × engineering cost.
None of them should be started before Section 6's baselines exist.

### 7.1 Option 1 — Foundation X Direct Baseline Reproduction *(highest priority, no new training)*

**Purpose:** establish the true Foundation X *direct segmentation* score
on our heldout split, so any later comparison has a real reference rather
than a paper-table reference.

**Sub-questions to answer first (Section 9.1):**

- Does our `checkpoints/foundation_x.pth` contain the SIIM segmentation
  head (S3) weights, or only the backbone? Inspect the 124 "unexpected"
  keys: list their prefixes and group them by branch.
- If S3 weights are present in the checkpoint, can we instantiate the
  decoder in our codebase, or do we need the original Foundation X
  repository code?
- If we need the original repository, identify the exact commit /
  release and required inputs.

**What we add:**

- A pure inference script `scripts/foundation_x_direct_segment.py` that:
  - loads the full Foundation X model (or the closest reproducible
    approximation), strict-mode if possible
  - runs SIIM segmentation head inference on a directory of images
  - outputs probability maps and binary masks at the contract-specified
    threshold
- A heldout evaluator that scores Foundation X direct against either
  `pneumothorax_trusted_v1/test` or Dataset101 `heldout_labelsTs`, using
  the same metric backend (D-020) used for all our other arms.

**Risks:** if `foundation_x.pth` does not contain the segmentation head,
or the original repository is not accessible, this option becomes
partial. Mitigation in Section 12.

**Expected output:** `artifacts/diagnostics/foundation_x_direct/` with
`per_case_metrics.csv`, `summary.yaml`, `report.md`, visual grids.

### 7.2 Option 2 — Current HybridFoundationUNet Heldout Evaluation *(immediate, no training)*

**Purpose:** convert PR-8's diagnostic best checkpoint into an
authoritative heldout score.

**What we add:** nothing new; run `scripts/evaluate_hybrid_heldout.py`
on Dataset101 heldout with the PR-8 best checkpoint, then optionally
re-evaluate on `pneumothorax_trusted_v1/test` with the same checkpoint
to expose the substrate mismatch.

**Expected output:** real `artifacts/diagnostics/hybrid_heldout_eval/
pr8_best_thr05/` with per-case Dice/IoU, case-level confusion matrix,
tiny/medium/large subgroup Dice.

**Acceptance:** the report must explicitly *not* claim hybrid beats
ResNet34 baseline if substrates differ. It must list the comparison
caveats clearly.

### 7.3 Option 3 — Foundation X Mask Prior + nnU-Net Refiner *(closest to TÜBİTAK proposal)*

**Purpose:** the original 2209-A pitch: Foundation X provides
coarse masks / probability priors, nnU-Net (or a refiner) produces the
final precise segmentation.

**Architecture:**

```
Input image (1 ch)
   │
   ├──► Foundation X (frozen, direct seg head) ──► coarse prob map (1 ch)
   │                                              ├─► coarse binary mask (1 ch)
   │                                              └─► uncertainty / entropy map (1 ch)
   │
   └─────────────── concat 4-ch input ─────────► nnU-Net / Refiner (trainable) ─► refined mask
```

**Variants:**

- **3a (lightweight refiner):** 2D U-Net trained from scratch, 4-channel
  input (image + coarse prob + coarse mask + entropy).
- **3b (pretrained refiner):** start from the trusted `pretrained_resnet34_unet`
  with stem extended to 4 channels (initialize new channels from the mean
  of the pretrained 1-channel filter; keep the rest).
- **3c (nnU-Net v2 refiner):** add Foundation X coarse-mask channel as an
  extra modality (nnU-Net v2 supports multi-modality input cleanly).
  Requires regenerating Dataset101 with multi-channel inputs.

**Why this is realistic:** if Foundation X direct segmentation is
moderately good (recall-oriented) but imprecise, a learned refiner can
exploit its coarse localization without inheriting its boundary errors.

**Risks:** if Foundation X direct is *weak*, the prior adds noise; if it
is too strong, the refiner overfits to its decisions (in particular: on
positive/negative case-level decisions).

**Expected gates:** must beat D-042 anchor by ≥ 0.02 absolute on
positive-only Dice **and** improve at least one of case-level
recall/precision over the nnU-Net baseline. Otherwise the cascade is not
worth the added complexity.

### 7.4 Option 4 — Multi-source training (SIIM + CANDID-PTX)

**Purpose:** more pneumothorax mask diversity → better generalization,
without changing the architecture family.

**Plan:** harmonize labels (binary pneumothorax), harmonize
preprocessing (size, intensity, channel), merge into a single training
manifold. Use stratified splits where the dataset-of-origin is also
balanced. Test heldout on each dataset separately and on a joint split
(D-043/D-044 apply).

**Risks:** CANDID-PTX labels may differ in annotation style; this needs
a curated sanity sample before training. License/access (Section 13).

### 7.5 Option 5 — Weakly supervised extension

**Purpose:** add TBX11K, NODE21, RSNA Pneumonia, ChestX-Det as
auxiliary tasks via pseudo-masks (e.g. SAM prompted by bounding boxes)
with a low-weight auxiliary loss.

**Constraint:** strong-label SIIM/CANDID remains the primary supervision;
pseudo-mask loss weight ≤ 0.2 (tunable as a small sweep on validation
only); per-image pseudo-mask confidence must be recorded.

**Risks:** pseudo-mask noise can hurt sparse-target segmentation
(D-030 already shows we are at < 0.30% foreground). This option should
only be tried after Option 3 or 4 is established.

### 7.6 Option 6 — Lock-release / partial unfreezing

**Purpose:** narrow the gap to the paper's training regime by unfreezing
the last 1–2 Swin-B stages with very low LR after the decoder/refiner has
converged.

**Strict safeguards:**

- Unfreeze schedule starts at epoch N (selected on validation), not from
  scratch.
- Backbone LR = decoder LR × 0.05 or less.
- Continue with EMA of backbone weights to avoid catastrophic forgetting.
- Validate every K steps; if positive Dice drops > 0.05 from the best,
  roll back to the best checkpoint and stop.
- Save the frozen-decoder baseline checkpoint before unfreezing so we
  can show the unfreeze did or did not help.

**Risks:** explicit leakage caveat (D-035): unfreezing on SIIM-exposed
weights is still in-domain transfer, not clean external pretraining.

### 7.7 Option 7 — Distillation / teacher-student

**Purpose:** treat Foundation X direct predictions as teacher logits;
train a student refiner with GT supervision + distillation loss.

**Loss:** `L = L_seg(student, gt) + λ · L_distill(student, teacher)`,
with teacher logits softened by temperature T.

**When this is the right tool:** Foundation X direct is strong on case
detection (recall) but miscalibrated or boundary-imprecise. Distillation
can transfer its inductive bias without inheriting its mistakes.

**Risks:** if Foundation X direct is weaker than our trained refiner on
positive Dice, distillation will likely hurt. Requires Section 7.1 first.

### 7.8 Ranking summary

| Option | Cost | Expected value | Should we attempt? |
| --- | --- | --- | --- |
| 1. Foundation X direct reproduction | Low–Medium (no training; depends on code availability) | High (unlocks Section 5.2 entry **B**) | **Yes, first.** |
| 2. Current hybrid heldout eval | Low (inference only) | Medium (closes PR-9, settles current state) | **Yes, in parallel.** |
| 3. Foundation X prior + refiner cascade | Medium (one new training pipeline) | High (matches TÜBİTAK pitch; realistic upside) | **Yes, after 1 + 2.** |
| 4. SIIM + CANDID-PTX | Medium (data integration + retraining) | Medium–High | After 3 is at least tied with baseline. |
| 5. Weakly supervised auxiliary | High (pseudo-mask pipeline) | Medium | Only if 3/4 plateau. |
| 6. Partial unfreezing | Medium | Medium (with leakage caveats) | Only after 3 or 4 is established. |
| 7. Distillation | Medium | Medium | Only after 1 is reproduced. |

---

## 8. Dataset Expansion Plan

| Dataset | Task | Disease target | Annotation | Helps SIIM seg? | Integration difficulty | Access / license | Priority |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SIIM-ACR | Segmentation | Pneumothorax | RLE pixel masks | (already used) | (already done) | Kaggle CC | (anchor) |
| CANDID-PTX | Segmentation | Pneumothorax | Pixel masks | **Direct** (same task) | Medium — label harmonization | Restricted access, requires application | **P1** |
| ChestX-Det | Segmentation | Multi-disease incl. pneumothorax bbox? (verify) | Bounding box + some pixel masks | Indirect — multi-disease context | Medium | Open / paper site | P3 |
| TBX11K | Detection / classification | Tuberculosis | Bounding box, image labels | Auxiliary (Option 5) | Medium — needs pseudo-masking | Open | P4 |
| NODE21 | Detection | Lung nodules | Bounding box | Auxiliary (Option 5) | Medium | Grand Challenge | P4 |
| RSNA Pneumonia | Detection | Pneumonia | Bounding box | Auxiliary (Option 5) | Easy | Kaggle CC | P4 |
| NIH ChestX-ray14 | Classification | 14 disease labels | Image-level | Auxiliary classification head | Easy | Open | P5 (only if Option 6 reopens multi-task) |
| Montgomery / JSRT | Anatomy seg | Lung / clavicle | Pixel masks | Lung-context priors | Easy | Open | P5 |

**Pragmatic plan:** SIIM-ACR + CANDID-PTX first. Everything else is
optional and only justified by Option 5 / 6 needs.

---

## 9. Training and Evaluation Strategy

### 9.1 Foundation X direct reproduction protocol

Before any new training, answer these in order:

1. **Inventory the checkpoint.** Print all top-level keys of
   `checkpoints/foundation_x.pth`, list all sub-keys, identify the
   distinct prefixes (`backbone.0.*`, `cls_heads.*`, `seg_heads.*`, etc.).
   The 124 unexpected keys at backbone load must each belong to a head /
   branch we can identify. The PR-4 smoke `load_metadata.json` already
   records counts; we need the full list.
2. **Is S3 (SIIM-ACR segmentation head) present?** If yes, can it be
   instantiated from the original Foundation X repository code, or from
   a re-implementation? If the repository is not available locally, this
   is a "User input needed" item (Section 13).
3. **What is the exact input contract?** Image size, channel layout,
   normalization, resize interpolation, mask format.
4. **What is the output contract?** Logits or probabilities, threshold
   policy.

Only then run inference and evaluate.

### 9.2 Cascade training protocol (Option 3)

1. Run Foundation X direct on the full training set, save coarse prob
   maps + binary masks + entropy. Treat these as additional data fields,
   not as model parameters.
2. Train the refiner with 4-channel input (image, coarse prob, coarse
   mask, entropy) for one epoch as a smoke test; verify finite gradients
   and decreasing loss.
3. Train to convergence with validation-only threshold selection per
   D-025/D-026. Same loss family as the trusted ResNet34 baseline
   (`dice_focal`, AdamW, ReduceLROnPlateau).
4. Save the authoritative D-029 run package; evaluate on test.
5. Report against the trusted anchor (D-042) and against Foundation X
   direct (Option 1).

### 9.3 Repeated-split discipline

For any architecture being compared to the D-071 pilot, **reuse**
`split_001`, `split_002`, `split_003` (per D-072). Paired deltas per
D-064; final summary per D-065.

### 9.4 Foundation X framing in every report

Per D-035/D-040/D-041, every Foundation X-touching artifact (Options 1,
3, 6, 7) must:

- Name `pretrained_resnet34_unet` (or its repeated-split pilot mean) as
  the comparison anchor.
- State the candidate score, the absolute delta, and whether the D-033
  threshold (≥ 0.5151) was cleared.
- Avoid wording such as "clean external transfer", "target-unseen
  generalization", or "Foundation X is the superior model" by default.
- Label the work as leakage-aware secondary evidence.

---

## 10. Metrics and Success Gates

### 10.1 Metric set (per heldout evaluation)

| Metric | Definition | Where it counts |
| --- | --- | --- |
| `val_dice_pos_mean` / `test_dice_pos_mean` | Per-image Dice on positive images only, then mean | **Primary** (D-011) |
| `dice_mean` | Per-image Dice on all images | Sanity / overall |
| `iou_mean` | Per-image IoU on all images | Secondary |
| Case-level precision / recall / F1 | Pred-positive vs GT-positive at case level | Clinical relevance |
| Specificity | TN / (TN + FP) | Clinical relevance |
| Negative FPR | FP / (FP + TN) | Operational |
| Detected-positive Dice | Mean Dice over predicted-positive ∩ GT-positive cases only | Quality of "found" detections |
| Tiny / medium / large lesion Dice | Bin by GT foreground area (e.g. <512 px / 512–8192 / >8192 at 512²) | Sparse-target failure analysis |
| Threshold sweep curve | Dice vs threshold 0.05–0.95 | Calibration diagnostic |
| Per-case CSV | image_id, subset_tag, dice, iou, precision, recall, f1, GT area | Audit |

**Empty-mask policy** (D-020):

- Both empty → 1.0
- One empty, one positive → 0.0
- `positive_mean` over an empty positive set → NaN.

**Hausdorff is NOT reported** (D-039) until a correctly specified
distance metric is reintroduced under a new decision.

### 10.2 Success gates

| Gate | Bar | Source of bar |
| --- | --- | --- |
| **G1 — Beat pure nnU-Net positive Dice on Dataset101 heldout** | mean Dice positive cases > 0.3722 | User-supplied baseline |
| **G2 — Beat or match nnU-Net case-level recall without big FPR cost** | recall ≥ 0.6919 AND negative FPR ≤ 0.10 | User-supplied baseline |
| **G3 — Reproduce or approximate Foundation X direct on the same heldout** | Foundation X direct number measurable | Section 7.1 |
| **G4 — Beat Foundation X direct with cascade / refinement** | Cascade > Foundation X direct on positive Dice; paired and significant if repeated-split | Section 7.3 |
| **G5 — Generalize to ≥ 1 additional dataset (CANDID-PTX)** | Positive Dice on CANDID-PTX heldout above its own simple-baseline reference | Section 8 |
| **D-033 hybrid keep gate** | Held-out positive Dice mean ≥ 0.5151 on `pneumothorax_trusted_v1` + full D-034 evidence | DECISIONS.md |

G1–G5 are **research gates**. D-033 is the **methodology gate** that
governs whether the hybrid re-enters the paper-path critical path.

### 10.3 What success looks like in numbers

A defensible final report would include, at minimum:

- Pretrained baseline anchor: 0.4951 (single split) / 0.5058 ± CI
  (3-split pilot).
- Foundation X direct on same substrate: number with CI.
- Cascade refiner: number with CI; paired delta vs anchor; paired delta
  vs Foundation X direct; both with 95% percentile bootstrap CIs.
- Failure analysis: tiny/medium/large lesion Dice; case-level confusion
  matrix; qualitative grids for at least 4 positives and 4 negatives per
  split.

---

## 11. PR Roadmap

The roadmap below extends the existing PR-1..PR-9 sequence. Each PR is
strictly scoped; none should be batched. All assume Foundation X work
remains within the D-035 framing boundary.

### PR-9 — Authoritative hybrid heldout evaluation *(current next step)*

- **Goal**: convert PR-8 best diagnostic checkpoint into a reportable
  Dataset101-heldout result.
- **Files**: existing `scripts/evaluate_hybrid_heldout.py`.
- **Command (illustrative)**:
  ```
  py scripts/evaluate_hybrid_heldout.py \
    --images_dir nnUNet_raw/Dataset101_Pneumothorax/imagesTs \
    --labels_dir nnUNet_raw/Dataset101_Pneumothorax/heldout_labelsTs \
    --hybrid_checkpoint artifacts/diagnostics/hybrid_controlled_short_train/<run>/diagnostic_checkpoint_best_dice_pos_mean_thr_050.pth \
    --foundation_checkpoint checkpoints/foundation_x.pth \
    --output_dir artifacts/diagnostics/hybrid_heldout_eval/pr8_best_thr05 \
    --threshold 0.5 --img_size 512 --device auto --strict
  ```
- **Artifacts**: `summary.yaml`, `per_case_metrics.csv`, `report.md`,
  visual grids, predicted masks.
- **Acceptance**: 1602 cases processed; positive Dice mean reported with
  TP/FN/FP/TN; report explicitly names Dataset101 as the substrate and
  states it is **not** comparable to D-042's 0.4951.
- **Rollback**: artifact-only; no model state changes.
- **Risks**: substrate mismatch with D-042 — must be flagged in the report.

### PR-9A — Foundation X direct segmentation reproduction

- **Goal**: produce `artifacts/diagnostics/foundation_x_direct/` with
  Foundation X direct segmentation results on the same heldout used by
  PR-9.
- **Files (new)**:
  `scripts/inspect_foundation_x_checkpoint.py` (key inventory),
  `scripts/foundation_x_direct_segment.py` (inference),
  `scripts/evaluate_foundation_x_direct_heldout.py` (evaluation).
- **User input dependencies**: original Foundation X repository / S3 head
  code (see Section 13).
- **Acceptance**: either Foundation X direct number is produced and
  reported, or a documented `BLOCKED` status with the missing-asset list.
- **Risks**: checkpoint may not contain S3; mitigation = closest
  reproduction (Section 12).

### PR-9B — Threshold calibration + tiny-lesion subgroup analysis

- **Goal**: full threshold sweep (0.05–0.95) and lesion-size subgroup
  Dice for the PR-9 hybrid, the trusted baseline, and (if PR-9A
  succeeds) Foundation X direct.
- **Files (new)**: `scripts/threshold_subgroup_analysis.py`.
- **Acceptance**: per-arm threshold curves + tiny/medium/large Dice CSV
  + report.md.

### PR-10 — Foundation X mask prior export pipeline

- **Goal**: cache Foundation X direct probability maps + binary masks
  + entropy maps for the entire SIIM training set, validation set, and
  heldout set, on `pneumothorax_trusted_v1` and (optionally) Dataset101.
- **Files (new)**: `scripts/export_foundation_x_priors.py`.
- **Artifacts**: a parallel directory tree mirroring `images/` with
  `prior_prob/`, `prior_mask/`, `prior_entropy/`.
- **Acceptance**: 100% case coverage, no NaN/Inf, manifest matching
  dataset fingerprint.
- **Risks**: storage cost — these are float16 maps at 512² × 3 channels
  × ~10,675 cases ≈ a few GB. Document in run metadata.

### PR-11 — Refiner / nnU-Net input-channel extension

- **Goal**: extend `PneumothoraxDataset` and either the trusted
  `pretrained_resnet34_unet` or nnU-Net v2 input to accept the Foundation
  X prior channels emitted in PR-10.
- **Files (modified)**: `src/data/dataset.py`,
  `src/models/resnet34_unet.py` (stem), config surface.
- **Acceptance**: forward pass succeeds with 4-channel input; tests for
  the new input contract; no regression on 1-channel mode (toggle by
  config).
- **Risks**: pretrained stem reinitialization needs care; track the new
  channels' initialization (e.g. mean of pretrained channel 1).

### PR-12 — Cascade training on SIIM (`pneumothorax_trusted_v1`)

- **Goal**: train the Foundation X prior + refiner cascade end-to-end
  on the trusted dataset, validation-only threshold selection, full
  D-029 evidence package.
- **Files (new)**:
  `configs/cascade_foundation_x_prior_v1.yaml`,
  `scripts/run_authoritative_cascade.py`.
- **Acceptance**: best `val_dice_pos_mean` ≥ 0.4951 + small margin on
  the trusted single-split; ideally ≥ 0.5151 to clear D-033.
- **Repeated-split follow-up**: rerun on D-071 pilot splits and report
  paired delta + 95% CI.
- **Risks**: cascade complexity may overfit the small validation set;
  monitor with early stopping (patience 30, same as D-028).

### PR-13 — CANDID-PTX dataset integration

- **Goal**: harmonized preprocessed dataset under
  `data/processed/candid_ptx_trusted_v1/` with the same manifest
  contract (D-019).
- **Files (new)**: `src/data/candid_ptx_preprocess.py`,
  `scripts/validate_candid_ptx_dataset.py`.
- **Acceptance**: dataset manifest, golden mask tests, sample overlay
  visuals.
- **Risks**: license / access (Section 13).

### PR-14 — Multi-source cascade training

- **Goal**: cascade refiner trained on SIIM ∪ CANDID-PTX, evaluated on
  both heldout sets separately and jointly.
- **Acceptance**: G5 gate met (cascade beats simple baseline on
  CANDID-PTX heldout).

### PR-15 — Pseudo-mask weak supervision

- **Goal**: optional auxiliary loss using SAM-prompted pseudo-masks
  from TBX11K / NODE21 / RSNA / ChestX-Det.
- **Acceptance**: no degradation on SIIM heldout; controlled experiment
  with auxiliary loss weight ablation (small sweep on validation).

### PR-16 — Partial unfreezing / lock-release ablation

- **Goal**: test Option 6 with strict safeguards.
- **Acceptance**: explicit pre-unfreeze frozen-decoder baseline checkpoint
  preserved; rollback rule triggered on Dice drop > 0.05.

### PR-17 — Final comparison report

- **Goal**: assemble final paper-grade report: all arms × all heldouts,
  paired deltas, split-bootstrap CIs, failure-mode visuals.
- **Acceptance**: D-045 evidence package complete.

---

## 12. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Foundation X checkpoint does not contain S3 / segmentation heads | Medium | Section 7.1 partially blocked | Inspect checkpoint keys early (PR-9A inspector); fall back to "feature transfer only" framing; do **not** invent a head and claim it is the paper's. |
| Original Foundation X repository / code is not available | Medium | Section 7.1 fully blocked | Document gap; reproduce only what we can; cite paper number with explicit "not reproduced on our split" caveat. |
| Substrate mismatch (Dataset101 vs `pneumothorax_trusted_v1`) silently inflates / deflates a comparison | High | Direct invalidation of any "hybrid beats baseline" claim | Pick one substrate explicitly per Section 5.3; always state the substrate in any cited number; never cross-compare. |
| Foundation X leakage caveat (D-006) gets minimized in the writeup | Medium | Methodological non-compliance with D-035 | Apply Section 9.4 checklist to every Foundation X-touching artifact; re-read VALIDATION_CHECKLIST §16 before submitting. |
| Sparse-target collapse during cascade training (foreground < 0.30%) | Medium | Empty-mask predictions | Use `dice_focal` loss as in D-028; class-balanced sampler; monitor predicted-positive ratio every K steps as in PR-7. |
| Refiner overfits to Foundation X prior errors | Medium | Cascade gain disappears at heldout | Train refiner with prior-dropout (randomly zero the prior on some training samples); evaluate with and without prior on heldout. |
| Storage cost of priors | Low | Engineering friction only | Store float16; cleanup script; track in dataset manifest. |
| CANDID-PTX access / license | Medium | PR-13 blocked | Submit application early; in parallel, develop PR-12 on SIIM only. |
| Pseudo-mask noise from SAM (Option 5) hurts SIIM Dice | Medium | G1 regression | Use small auxiliary loss weight; ablate; document. |
| Partial unfreezing destabilizes training (Option 6) | Medium | Bad checkpoint promoted | Strict rollback rule, EMA, low LR; keep frozen baseline checkpoint. |
| Repeated-split CI is too narrow / wide on 3 splits to draw conclusions | High | Publication-quality claims premature | Plan for 5–10 split expansion before publication-final claims, per D-072. |

---

## 13. Required User Inputs / Missing Assets

The following are required (or strongly preferred) inputs from the user
before specific PRs can complete:

1. **Original Foundation X repository / commit** — required for PR-9A
   (Option 1). Specifically: path or URL of the official repo, commit
   hash if you remember it, exact release tag.
2. **Foundation X checkpoint provenance** — is `checkpoints/foundation_x.pth`
   the CLS-only release, the LS release, the S-only release, or the full
   joint checkpoint? Any README / release notes that shipped with it.
3. **Whether segmentation heads (S1..Sn, S3 in particular) are included**
   in this checkpoint — confirmed by a key dump (PR-9A inspector) but
   user confirmation of intent helps.
4. **Original Foundation X inference command / script** — if you have
   the paper authors' inference recipe or example, this dramatically
   reduces reproduction risk.
5. **Dataset licenses / access**:
   - CANDID-PTX: confirm whether we have access (application-based).
   - ChestX-Det / TBX11K / NODE21 / RSNA Pneumonia: confirm we can
     download under the desired license.
6. **GPU resources for follow-on PRs** — PR-12 cascade training, PR-14
   multi-source, PR-16 partial unfreezing each need GPU time roughly
   comparable to a baseline run. Confirm Colab / on-prem availability.
7. **Whether CANDID-PTX and ChestX-Det can be downloaded** to this
   workspace — this affects PR-13.
8. **Final reporting target** — is the deliverable:
   - the **TÜBİTAK 2209-A** report (in which case Sections 3–5 of this
     document set the framing, and partial reproduction is acceptable),
   - a **paper / preprint** (in which case D-043/D-044/D-045
     publication-grade evaluation is mandatory and PR-12 must be
     followed by repeated-split + CIs),
   - or a **competition-style best score** (in which case G1–G5 take
     precedence over methodological framing, but D-035 caveats still
     apply to wording).

The answer changes the priorities in Section 7 and Section 11.

---

## 14. Recommended Immediate Next Steps

In execution order:

1. **PR-9** (Section 11): authoritative hybrid heldout evaluation on
   Dataset101 imagesTs / heldout_labelsTs using the PR-8 best
   checkpoint. No model changes. Output: real `artifacts/diagnostics/
   hybrid_heldout_eval/pr8_best_thr05/`. Acceptance per PR-9 entry.
2. **PR-9A** (Section 11): start the Foundation X checkpoint inventory
   *immediately*. The key dump alone is cheap and informs every later
   option. If S3 is present and the original repo is available, proceed
   to direct inference; if not, document the gap.
3. **PR-9B** (Section 11): threshold calibration + tiny-lesion subgroup
   analysis on the PR-9 hybrid + the trusted baseline. This costs only
   inference time and produces failure-mode evidence usable in any
   writeup.
4. **Substrate decision**: pick Option α (Dataset101) or Option β
   (`pneumothorax_trusted_v1`) per Section 5.3 and record the choice in
   `DECISIONS.md`. Do not start PR-10..PR-12 until this is fixed.
5. **PR-10 + PR-11**: build the prior export pipeline and 4-channel
   input adapter. Both can proceed in parallel once the substrate is
   chosen.
6. **PR-12**: first authoritative cascade run on the chosen substrate.
   Report results against D-042 anchor and (if PR-9A succeeded) against
   Foundation X direct.
7. **Then** (and only then) consider PR-13..PR-17.

PR-9 and PR-9A can run in parallel — they share no model state and
PR-9A is mostly inspection + offline inference.

---

## 15. Final Decision

This document does **not** make the actual experimental decisions; it
proposes the framing and the ordering. The actual binding decisions are
recorded in `DECISIONS.md`. The recommended *direction*, consistent with
existing decisions, is:

- Treat the current `HybridFoundationUNet` as a **trainable diagnostic**,
  not as the project's headline contribution. It clears engineering
  gates (D-052/D-053/D-058/D-070) and produces a non-trivial val
  `dice_pos_mean` ≈ 0.4516 on a small diagnostic split, but it has not
  cleared D-033 and remains deferred from the main paper path.
- The most realistic path to outperforming both the pretrained baseline
  *and* the Foundation X paper's number on our heldout is **Option 3**:
  Foundation X direct segmentation as a coarse mask prior consumed by an
  nnU-Net / refiner cascade, exactly as proposed in the TÜBİTAK 2209-A
  application.
- Before any new architecture work, run PR-9, PR-9A, and PR-9B in
  parallel and let the results inform whether Option 3 is feasible
  (depends on Section 7.1) or whether we should fall back to a
  feature-transfer hybrid with partial unfreezing (Option 6) under
  strict D-035 caveats.
- Use the trusted full-image `pretrained_resnet34_unet` (D-042 anchor,
  D-071 pilot CI) as the **only** comparison anchor in any Foundation
  X-touching writeup, with the absolute delta and D-033 status stated
  in the same paragraph.
- Maintain the D-035 framing boundary in every artifact: Foundation X
  is in-domain transfer / leakage-aware secondary evidence, not clean
  external pretraining.

If a future cascade run clears D-033 (held-out positive Dice mean ≥
0.5151) with the full D-034 evidence package, the hybrid / cascade may
re-enter the critical path as the project's main contribution, still
under D-035 framing.

---

## Appendix A — Glossary of repo-internal decisions referenced

| ID | What it fixes |
| --- | --- |
| D-006 | SIIM exposure in Foundation X pretraining → leakage caveat. |
| D-011 | `val_dice_pos_mean` is the authoritative model-selection metric. |
| D-017 | `original_masks` (eval) vs `dilated_masks` (train) policy. |
| D-019 | `pneumothorax_trusted_v1` is the trusted dataset root. |
| D-020 | Per-image overlap-metric reduction modes + empty-mask policy. |
| D-025/D-026 | Validation-only threshold selection + `selection_state.yaml`. |
| D-029 | Authoritative D-010 + selection + reports + qualitative package. |
| D-031/D-032 | Train-only 384² ROI crop tested and rejected (0.4625 < 0.4951). |
| D-033 | Hybrid keep gate: held-out positive Dice ≥ 0.5151 + D-034 evidence. |
| D-034 | Hybrid reopening evidence package contract. |
| D-035/D-040/D-041/D-042 | Foundation X framing boundary and comparison anchor. |
| D-043/D-044/D-045 | Repeated-split, paired-delta, final evidence package. |
| D-050–D-053 | Frozen/unfrozen hybrid gradient semantics + BN-mode fix. |
| D-054–D-059 | Hybrid fusion scale contract (fx[0]→e3, fx[1]→e4, fx[2]→H/16, fx[3]→H/32). |
| D-060–D-062, D-070 | Foundation X branch normalization (ImageNet mean/std) contract. |
| D-063–D-069 | Repeated-split orchestration contract. |
| D-071/D-072 | First repeated-split pretrained pilot (interim anchor). |
| D-073 | Foundation X smoke test PASS criteria. |

---

*End of document.*
