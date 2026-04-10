# Foundation-nnU-Net Agent Context

Current phase:
- Phase 0 -> Phase 1 transition planning

Current blocker:
- The repository does not yet have a trusted label pipeline or trusted validation metrics.

Highest-priority open tasks:
1. Document that notebook-generated outputs are not evidence unless traceable to config, checkpoint, and dataset version.
2. Prove the SIIM RLE decoding contract with golden checks.
3. Preserve original and dilated masks separately and regenerate the processed dataset.
4. Rewrite validation metrics to operate per image and fix positive-only Dice counting once the trusted dataset path is ready.
5. Define whether `P0.1` can be closed after the notebook-evidence rule is written down.

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

What is still untrusted:
- Label provenance from `src/data/preprocess.py`.
- Historical metrics and plots under `results/`, which remain legacy-only artifacts.
- Validation/model-selection numbers from the current trainer, because the code path has not yet been updated to honor the defined authoritative metric.
- Any claim involving Foundation X as clean external pretraining on SIIM.
- The scientific value of the current hybrid design.
- Git-based provenance alone is not sufficient for portability; `code_fingerprint` fallback remains necessary for environments without usable Git metadata.

Current strategic direction:
- Fix trust issues first, then build a strong pretrained CNN baseline, then decide whether the hybrid is worth redesigning.

Next 3 actions:
1. Document that notebook-generated outputs are not evidence unless traceable to config, checkpoint, and dataset version.
2. Resolve Phase 1 label correctness: authoritative RLE contract, golden checks, original+dilated mask policy.
3. Execute Phase 2 metric corrections so the trainer/evaluator actually use the defined primary metric correctly.
