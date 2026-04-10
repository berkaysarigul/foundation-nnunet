# Notebook Output Policy

Notebooks in this directory are for exploratory or operational use only.

Policy:
- Notebook-generated outputs are not authoritative evidence by default.
- A notebook output may be treated as evidence only if it is traceable to:
  - the exact config used
  - the exact checkpoint used
  - the exact dataset version or fingerprint used
- If that traceability is missing, the output must be treated as non-authoritative, even if it looks plausible.

Implications:
- Figures, tables, metrics, and qualitative samples produced ad hoc from notebooks must not be used for model selection, architecture comparison, or paper reporting unless the required provenance is recorded.
- Authoritative experiment outputs belong under `artifacts/runs/`, not as ad hoc notebook outputs.
