# Authoritative Run Location

This directory is the canonical repository location for authoritative experiment runs.

Policy:
- Future authoritative training and evaluation runs must live under `artifacts/runs/`.
- The legacy `results/` directory is not an authoritative output location.
- Each authoritative run must use its own run directory under this location and follow the metadata/output contracts recorded in the recovery documents.

This file defines location only.
It does not replace the run-level metadata and output requirements defined elsewhere in the recovery system.
