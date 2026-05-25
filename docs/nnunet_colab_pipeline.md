# Colab nnU-Net v2 Pipeline - Dataset101_Pneumothorax

> **Version:** 1.0  
> **Scope:** PR-3 documentation only - no code changes.  
> **Status:** Ready for Colab execution after gates pass.  
> **Prerequisites:** PR-1 audit PASS, PR-2 Dataset101 export PASS.

---

## 1. Confirmed PR-1 / PR-2 Baselines

These are the expected values **before** any Colab command is run.  
If Colab checks diverge, **stop and fix upstream** - do not patch in Colab.

### PR-1 Leakage Audit (source splits)

| Metric | Expected |
|---|---:|
| `processed_split_total` | 10675 |
| `matched_processed_metadata` | 10675 |
| `patient_leakage_keys` | 0 |
| `study_leakage_keys` | 0 |
| `series_leakage_keys` | 0 |
| `duplicate_sop_keys` | 0 |
| PR-1 summary status | `PASS` |

### PR-2 Dataset101 Export

| Path / Metric | Expected |
|---|---:|
| `imagesTr/*_0000.png` | 9073 |
| `labelsTr/*.png` | 9073 |
| `imagesTs/*_0000.png` | 1602 |
| `heldout_labelsTs/*.png` | 1602 |
| `case_mapping.csv` data rows | 10675 |
| `splits_final_source.json` fold0 train | 7471 |
| `splits_final_source.json` fold0 val | 1602 |
| `dataset.json` | exists, fields match spec |
| `labelsTs/` | **must not exist** |

### dataset.json expected fields

```json
{
  "channel_names": {"0": "grayscale"},
  "labels": {"background": 0, "pneumothorax": 1},
  "numTraining": 9073,
  "file_ending": ".png",
  "overwrite_image_reader_writer": "NaturalImage2DIO"
}
```

---

## 2. Google Drive Folder Layout

Mount Drive in Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Required layout under `/content/drive/MyDrive/Foundation-nnUNet/`:

```text
/content/drive/MyDrive/Foundation-nnUNet/
  repo/
  nnUNet_raw/
    Dataset101_Pneumothorax/
      dataset.json
      imagesTr/
      labelsTr/
      imagesTs/
      heldout_labelsTs/
      case_mapping.csv
      splits_final_source.json
  nnUNet_preprocessed/
    Dataset101_Pneumothorax/
      dataset_fingerprint.json
      nnUNetPlans.json
      dataset.json
      splits_final.json         # copied from splits_final_source.json
      nnUNetPlans_2d/
  nnUNet_results/
    Dataset101_Pneumothorax/
      nnUNetTrainer__nnUNetPlans__2d/
        fold_0/
  artifacts/
    nnunet_predictions/
      Dataset101_Pneumothorax/
        fold0/
    reports/
```

**Create folders:**

```bash
mkdir -p "/content/drive/MyDrive/Foundation-nnUNet/nnUNet_raw"
mkdir -p "/content/drive/MyDrive/Foundation-nnUNet/nnUNet_preprocessed"
mkdir -p "/content/drive/MyDrive/Foundation-nnUNet/nnUNet_results"
mkdir -p "/content/drive/MyDrive/Foundation-nnUNet/artifacts/nnunet_predictions/Dataset101_Pneumothorax/fold0"
mkdir -p "/content/drive/MyDrive/Foundation-nnUNet/artifacts/reports"
```

### Storage Strategy

**Default (safer):** Keep `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` all on Drive.
- Slower I/O, but protects preprocessed data, logs, and checkpoints across Colab disconnects.

**Fast mode (riskier):** Copy `Dataset101_Pneumothorax` to `/content/nnUNet_raw/`, use `/content/nnUNet_preprocessed` for speed.
- **Required:** `rsync` checkpoints to Drive periodically and at session end.
- Do **not** use fast mode for full training unless a backup cadence is explicitly followed.

**Compromise:** Drive-backed `nnUNet_results`, fast `/content` for `nnUNet_preprocessed`.
- Preprocessed data is reproducible; checkpoints are not.

Dataset101 must be fully uploaded or copied to Drive before preprocessing. Do
not run `nnUNetv2_plan_and_preprocess` against a partially synced Drive folder.
The file-count gate in section 4 is mandatory after every upload or copy.

---

## 3. Colab Environment Setup

Run these cells **in order** after every runtime restart.

### 3.1 GPU Gate

```bash
nvidia-smi
```

```python
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
```

Strict CUDA gate:

```bash
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is not available"
print(torch.cuda.get_device_name(0))
PY
```

**Gate passes when:**
- `nvidia-smi` shows a GPU.
- `torch.cuda.is_available()` is `True`.
- A CUDA device name is reported.

If either fails: switch to GPU runtime and re-run from section 3.1. Do not proceed without GPU.

### 3.2 Install nnU-Net v2

```bash
pip install -U nnunetv2
```

> PyTorch must be installed **before** `nnunetv2`. The exact install command may depend
> on the Colab base image and PyTorch version. Verify after install.

### 3.3 nnU-Net CLI Gate

```bash
python - <<'PY'
import nnunetv2
import batchgenerators
import torch
print("nnunetv2:", nnunetv2.__version__ if hasattr(nnunetv2, '__version__') else "imported")
print("batchgenerators: OK")
print("torch: OK")
PY
```

```bash
which nnUNetv2_plan_and_preprocess
which nnUNetv2_train
which nnUNetv2_predict
```

```bash
nnUNetv2_plan_and_preprocess -h 2>&1 | head -40
```

```bash
nnUNetv2_train -h 2>&1 | head -60
```

```bash
nnUNetv2_predict -h 2>&1 | head -60
```

**Gate passes when:**
- All three imports succeed.
- All three CLI commands exist on `PATH`.
- Help output confirms valid syntax.

If `nnunetv2` import fails: reinstall, check Python/PyTorch compatibility.

### 3.4 Pre-Training Report Capture

Capture package versions and CLI help output before any smoke or full training.
This records the installed nnU-Net behavior used for version-sensitive flags.

```bash
pip freeze > /content/drive/MyDrive/Foundation-nnUNet/artifacts/reports/pip_freeze_before_training.txt
nnUNetv2_train -h > /content/drive/MyDrive/Foundation-nnUNet/artifacts/reports/nnunetv2_train_help.txt
nnUNetv2_predict -h > /content/drive/MyDrive/Foundation-nnUNet/artifacts/reports/nnunetv2_predict_help.txt
nnUNetv2_plan_and_preprocess -h > /content/drive/MyDrive/Foundation-nnUNet/artifacts/reports/nnunetv2_plan_help.txt
```

If any help command fails, stop and repair the environment before training.

### 3.5 Environment Variables

```bash
export nnUNet_raw="/content/drive/MyDrive/Foundation-nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/content/drive/MyDrive/Foundation-nnUNet/nnUNet_preprocessed"
export nnUNet_results="/content/drive/MyDrive/Foundation-nnUNet/nnUNet_results"
```

```bash
echo "nnUNet_raw: $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"
echo "nnUNet_results: $nnUNet_results"
```

```bash
test -d "$nnUNet_raw/Dataset101_Pneumothorax" && echo "Dataset101 found in nnUNet_raw" || echo "MISSING: Dataset101 not in nnUNet_raw"
```

> **CRITICAL:** Environment variables are session-local. Re-run this cell after
> **every** Colab runtime restart, reconnect, or factory reset. Failing to re-export
> `nnUNet_results` before training will write checkpoints to a default location
> and lose them on disconnect.

If you use fast-mode paths (`/content/...`), update the exports accordingly.

Optional fast mode with Drive-backed results:

```bash
export nnUNet_raw="/content/nnUNet_raw"
export nnUNet_preprocessed="/content/nnUNet_preprocessed"
export nnUNet_results="/content/drive/MyDrive/Foundation-nnUNet/nnUNet_results"
```

Fastest mode, highest checkpoint-loss risk:

```bash
export nnUNet_raw="/content/nnUNet_raw"
export nnUNet_preprocessed="/content/nnUNet_preprocessed"
export nnUNet_results="/content/nnUNet_results"
```

Use fastest mode for full training only if periodic `rsync` backup to Drive is
already planned and tested.

---

## 4. Dataset Gate

Run **before** preprocessing. File, count, schema, and `labelsTs` absence checks
must pass before preprocessing. Label-value risk is cleared either by the full
direct scan in this section or by successful nnU-Net integrity verification in
section 5; sampling alone is not sufficient.

Run this gate after Dataset101 has finished uploading or copying to Drive. A
partial upload can have correct folder names but incomplete files, so counts are
the guardrail that prevents wasting GPU time on a broken dataset.

### 4.1 File Counts

```bash
DATASET="$nnUNet_raw/Dataset101_Pneumothorax"

echo "imagesTr:        $(find "$DATASET/imagesTr" -maxdepth 1 -type f -name '*.png' | wc -l)"
echo "labelsTr:        $(find "$DATASET/labelsTr" -maxdepth 1 -type f -name '*.png' | wc -l)"
echo "imagesTs:        $(find "$DATASET/imagesTs" -maxdepth 1 -type f -name '*.png' | wc -l)"
echo "heldout_labelsTs: $(find "$DATASET/heldout_labelsTs" -maxdepth 1 -type f -name '*.png' | wc -l)"
```

Strict count check:

```bash
check_count() {
  actual="$(find "$1" -maxdepth 1 -type f -name '*.png' | wc -l)"
  expected="$2"
  label="$3"
  if [ "$actual" -ne "$expected" ]; then
    echo "ERROR: $label expected $expected, got $actual"
    exit 1
  fi
  echo "$label OK: $actual"
}

check_count "$DATASET/imagesTr" 9073 "imagesTr"
check_count "$DATASET/labelsTr" 9073 "labelsTr"
check_count "$DATASET/imagesTs" 1602 "imagesTs"
check_count "$DATASET/heldout_labelsTs" 1602 "heldout_labelsTs"
```

| Path | Expected |
|---|---|
| `imagesTr/*_0000.png` | 9073 |
| `labelsTr/*.png` | 9073 |
| `imagesTs/*_0000.png` | 1602 |
| `heldout_labelsTs/*.png` | 1602 |

### 4.2 Required Files

```bash
test -f "$DATASET/dataset.json" || { echo "MISSING: dataset.json"; exit 1; }
test -f "$DATASET/splits_final_source.json" || { echo "MISSING: splits_final_source.json"; exit 1; }
test -f "$DATASET/case_mapping.csv" || { echo "MISSING: case_mapping.csv"; exit 1; }
test ! -d "$DATASET/labelsTs" || { echo "ERROR: labelsTs must not exist"; exit 1; }

echo "dataset.json OK"
echo "splits_final_source.json OK"
echo "case_mapping.csv OK"
echo "labelsTs absent OK"
```

### 4.3 case_mapping.csv Row Count

```python
import csv
from pathlib import Path
import os

p = Path(os.environ["nnUNet_raw"]) / "Dataset101_Pneumothorax" / "case_mapping.csv"
with p.open(newline="") as f:
    rows = list(csv.reader(f))
data_rows = len(rows) - 1
print(f"physical lines: {len(rows)}, data rows: {data_rows}")
assert data_rows == 10675, f"Expected 10675, got {data_rows}"
print("case_mapping.csv row count OK")
```

### 4.4 dataset.json Fields

```python
import json, os
from pathlib import Path

p = Path(os.environ["nnUNet_raw"]) / "Dataset101_Pneumothorax" / "dataset.json"
d = json.loads(p.read_text())

assert d["channel_names"] == {"0": "grayscale"}, f"channel_names: {d['channel_names']}"
assert d["labels"] == {"background": 0, "pneumothorax": 1}, f"labels: {d['labels']}"
assert d["numTraining"] == 9073, f"numTraining: {d['numTraining']}"
assert d["file_ending"] == ".png", f"file_ending: {d['file_ending']}"
reader = d.get("overwrite_image_reader_writer")
assert reader == "NaturalImage2DIO", f"overwrite_image_reader_writer: {reader}"
print("dataset.json fields OK")
```

### 4.5 Label Value Scan (optional before preprocessing)

Sampling check. This is a fast warning check only; it does not prove the full
label set is valid.

```python
from pathlib import Path
from PIL import Image
import numpy as np
import os, random

root = Path(os.environ["nnUNet_raw"]) / "Dataset101_Pneumothorax"
sample_dirs = [root / "labelsTr", root / "heldout_labelsTs"]
all_values = set()

for d in sample_dirs:
    files = list(d.glob("*.png"))
    for p in random.sample(files, min(500, len(files))):
        arr = np.asarray(Image.open(p))
        all_values.update(np.unique(arr).tolist())

print("sampled label values:", sorted(all_values))
assert all_values.issubset({0, 1}), f"Invalid label values: {sorted(all_values)}"
print("sampled label value scan OK")
```

Full direct scan. This can take time, but it is the only direct Colab-side label
scan that fully proves all training and heldout masks use only `0/1`.

```python
from pathlib import Path
from PIL import Image
import numpy as np
import os

root = Path(os.environ["nnUNet_raw"]) / "Dataset101_Pneumothorax"
values = set()

for d in [root / "labelsTr", root / "heldout_labelsTs"]:
    for p in d.glob("*.png"):
        arr = np.asarray(Image.open(p))
        values.update(np.unique(arr).tolist())
        if not values.issubset({0, 1}):
            raise RuntimeError(f"Unexpected label values so far: {sorted(values)} at {p}")

print("label values:", sorted(values))
assert values.issubset({0, 1})
print("full label scan OK")
```

Sampling alone is not enough to clear label-value risk before training. Before
full training, labels must be accepted by `nnUNetv2_plan_and_preprocess` with
`--verify_dataset_integrity` or confirmed by the full direct scan above. Run the
full scan if you want explicit validation of both `labelsTr` and
`heldout_labelsTs`, because installed nnU-Net behavior should not be assumed for
nonstandard heldout label folders.

### 4.6 Dataset Gate - Go / No-Go

| Check | Must Pass |
|---|---|
| `imagesTr = 9073` | yes |
| `labelsTr = 9073` | yes |
| `imagesTs = 1602` | yes |
| `heldout_labelsTs = 1602` | yes |
| `case_mapping.csv` = 10675 data rows | yes |
| `dataset.json` fields match spec | yes |
| `splits_final_source.json` present | yes |
| `labelsTs/` absent | yes |
| label values are subset of {0,1} by full scan, or later accepted by nnU-Net integrity verification | yes before training |

**If any check fails:** stop. Re-upload or re-export Dataset101 from PR-2. Do not patch in Colab.

---

## 5. nnU-Net Preprocessing

Run only after the pre-preprocessing Dataset Gate checks pass.

### 5.1 Dataset Integrity Check + Planning

```bash
nnUNetv2_plan_and_preprocess -d 101 --verify_dataset_integrity -c 2d
```

Expected outcome:
- Command exits with status `0`.
- No integrity assertion failures.

### 5.2 Verify Preprocessing Output

```bash
PP="$nnUNet_preprocessed/Dataset101_Pneumothorax"

test -d "$PP"                                         && echo "preprocessed dir OK"   || echo "MISSING: preprocessed dir"
test -f "$PP/dataset_fingerprint.json"                  && echo "fingerprint OK"         || echo "MISSING: fingerprint"
test -f "$PP/nnUNetPlans.json"                         && echo "plans OK"               || echo "MISSING: plans"
test -f "$PP/dataset.json"                              && echo "dataset.json OK"        || echo "MISSING: dataset.json"
test -d "$PP/nnUNetPlans_2d"                            && echo "plans 2d dir OK"        || echo "MISSING: plans 2d dir"
```

```bash
find "$PP" -maxdepth 2 -type d | sort
```

### 5.3 Common Preprocessing Failures

| Failure | Response |
|---|---|
| `NaturalImage2DIO` not recognized | Check installed nnU-Net version. Official docs list `NaturalImage2DIO` for PNG. May indicate version/install mismatch. If persistent, verify with `nnUNetv2_plan_and_preprocess -h`. Do **not** change the dataset silently. |
| `dataset.json` rejected | Print and validate JSON. Confirm `channel_names`, `labels`, `numTraining`, `file_ending`. |
| File naming rejection | Confirm `imagesTr` uses `_0000.png` channel suffix. Labels use matching `CASEID.png`. |
| PNG reader failure | Confirm Pillow opens representative PNGs. Confirm lossless PNG, readable, dimensions match. |
| Labels not `0/1` | Identify offending files. Requires dataset export correction upstream. |
| Preprocessing stalls on Drive | Switch `nnUNet_preprocessed` to `/content/nnUNet_preprocessed` for fast mode. Rerun from clean preprocessed dir. |
| Env vars not set | Re-run section 3.5. Verify with `echo $nnUNet_preprocessed`. |

### 5.4 nnU-Net Gate - Go / No-Go

| Check | Must Pass |
|---|---|
| `nnUNetv2_plan_and_preprocess` exits 0 | yes |
| `$nnUNet_preprocessed/Dataset101_Pneumothorax/` exists | yes |
| `dataset_fingerprint.json` exists | yes |
| `nnUNetPlans.json` exists | yes |
| `nnUNetPlans_2d/` exists | yes |

---

## 6. Fold-0 Split Placement

nnU-Net v2 reads `splits_final.json` from the preprocessed dataset directory.
If missing, nnU-Net may auto-generate its own split - breaking PR-1 leakage guarantees.

### 6.1 Copy Split

```bash
cp "$nnUNet_raw/Dataset101_Pneumothorax/splits_final_source.json" \
   "$nnUNet_preprocessed/Dataset101_Pneumothorax/splits_final.json"
```

### 6.2 Split Gate Verification

```python
import json, os
from pathlib import Path

p = Path(os.environ["nnUNet_preprocessed"]) / "Dataset101_Pneumothorax" / "splits_final.json"
splits = json.loads(p.read_text())

assert isinstance(splits, list), f"top-level must be list, got {type(splits)}"
assert len(splits) >= 1, f"expected at least 1 fold, got {len(splits)}"

fold0 = splits[0]
assert set(fold0.keys()) == {"train", "val"}, f"fold0 keys: {set(fold0.keys())}"

n_train = len(fold0["train"])
n_val   = len(fold0["val"])
assert n_train == 7471, f"fold0 train expected 7471, got {n_train}"
assert n_val == 1602,   f"fold0 val expected 1602, got {n_val}"

overlap = set(fold0["train"]) & set(fold0["val"])
assert not overlap, f"train/val overlap: {len(overlap)} cases"

print(f"fold0 train: {n_train}")
print(f"fold0 val:   {n_val}")
print("Split Gate PASS")
```

### 6.3 Split Gate - Go / No-Go

| Check | Must Pass |
|---|---|
| `splits_final.json` exists in preprocessed dir | yes |
| Fold 0 is a dict with keys `train`, `val` | yes |
| Fold 0 train length = 7471 | yes |
| Fold 0 val length = 1602 | yes |
| Train/val overlap = 0 | yes |

> **Full training is forbidden until the Split Gate passes.**

---

## 7. Smoke Training Gate

Prove CUDA, dataset loading, model construction, loss computation, logging, and
output-folder creation work before spending long GPU time.

### 7.1 Verify CLI

```bash
nnUNetv2_train -h 2>&1 | head -80
```

The full help output should already be captured in section 3.4 before training.
Look for supported flags: `-device`, `--c` (continue/resume), short-run or
epoch-limit options. Exact options depend on installed nnU-Net version. Note
available flags before starting.

### 7.2 Use a Separate Smoke Results Root

Prefer a separate Drive-backed smoke results root so smoke outputs cannot
accidentally become the full fold-0 training run:

```bash
export nnUNet_results="/content/drive/MyDrive/Foundation-nnUNet/nnUNet_results_smoke"
mkdir -p "$nnUNet_results"
echo "smoke nnUNet_results: $nnUNet_results"
```

Smoke outputs are disposable unless you intentionally decide to resume from
them. Before full training, reset `nnUNet_results` to the real results root in
section 8.

### 7.3 Start Smoke Training

```bash
nnUNetv2_train 101 2d 0 -device cuda
```

> The `-device cuda` flag is documented by nnU-Net. Verify with `nnUNetv2_train -h`
> before running. If unsupported, omit and rely on default CUDA device.

### 7.4 What to Watch

Start training and let it run **only** until all of:
1. Output folder appears under `$nnUNet_results/Dataset101_Pneumothorax/`.
2. Data loading begins (dataset unpacking messages).
3. Network is initialized (architecture summary).
4. First training iterations complete.
5. Loss values are printed / logged and are finite.
6. `nvidia-smi` shows GPU memory usage.

**Stop manually** after roughly 5-15 minutes or after the first clear successful
iterations - **whichever comes first.**

**Do not let smoke training run unattended.** Treat any partial smoke
checkpoint/output as disposable unless intentionally resumed.

### 7.5 Monitor GPU

```bash
nvidia-smi
```

Re-run periodically during smoke training.

### 7.6 Check Smoke Output Folder

```bash
find "$nnUNet_results/Dataset101_Pneumothorax" -maxdepth 4 -type d | sort
```

```bash
find "$nnUNet_results/Dataset101_Pneumothorax" -maxdepth 5 -type f | sort | head -80
```

### 7.7 Immediate Stop Signs

| Symptom | Action |
|---|---|
| CUDA OOM | Stop. Do not start full training. |
| NaN or Inf loss | Stop. Inspect labels, normalization, and training logs. |
| Dataset loading failure | Stop. Check dataset.json, file naming, reader. |
| Label value error | Stop. Run full label scan. |
| Shape mismatch | Stop. Check image/label dimensions. |
| `NaturalImage2DIO` / PNG reader failure | Stop. Verify nnU-Net version and dataset format. |
| Output folder not created | Stop. Check `nnUNet_results` env var and disk permissions. |
| Training stuck (no CPU/GPU/disk activity) | Stop. Check I/O path; switch to fast mode if on Drive. |
| Split warning (custom split not loaded) | Stop. Re-run section 6. |
| Drive I/O errors | Stop. Check Drive mount and connectivity. |

### 7.8 Smoke Training Gate - Go / No-Go

| Check | Must Pass |
|---|---|
| Training started without errors | yes |
| First iterations ran | yes |
| Loss is finite (not NaN, not Inf) | yes |
| GPU is active (nvidia-smi confirms) | yes |
| Results/log folder created under `nnUNet_results` | yes |
| No dataset/label/split/reader/CUDA errors | yes |

> Smoke outputs are disposable. Do not let a smoke result folder become the full
> training run by accident. If you smoke-tested in the real results root, clean
> that folder before full training, or consciously use `--c` to resume.

---

## 8. Full Training

Only after **all prior gates pass**: Dataset Gate, nnU-Net Gate, Split Gate,
GPU Gate, Smoke Training Gate.

### 8.1 Reset Results Root

Before full training, reset `nnUNet_results` away from the smoke root:

```bash
export nnUNet_results="/content/drive/MyDrive/Foundation-nnUNet/nnUNet_results"
mkdir -p "$nnUNet_results"
echo "full-training nnUNet_results: $nnUNet_results"
```

### 8.2 Clean Full-Training Result Folder Check

```bash
test ! -d "$nnUNet_results/Dataset101_Pneumothorax/nnUNetTrainer__nnUNetPlans__2d/fold_0" && echo "clean full-training result folder OK" || echo "WARNING: existing fold_0 result folder exists"
```

If the folder exists, do not start full training casually. Either intentionally
resume from it after verifying the checkpoint and command, or back it up and
remove it before starting a fresh full run.

### 8.3 Command

```bash
nnUNetv2_train 101 2d 0 -device cuda
```

### 8.4 Results Location

```text
$nnUNet_results/Dataset101_Pneumothorax/nnUNetTrainer__nnUNetPlans__2d/fold_0/
```

Expected files during/after training:
- `debug.json`
- Training log files
- `progress.png`
- `checkpoint_latest.pth` (or similar latest checkpoint)
- `checkpoint_best.pth`
- `checkpoint_final.pth`
- `validation/summary.json` (after validation/finalization)

> Exact checkpoint names depend on installed nnU-Net version.

### 8.5 Storage Recommendation

**Safe mode (preferred):** Keep `nnUNet_results` on Drive.
- Pro: checkpoints survive Colab disconnect.
- Con: slower checkpoint writes.

**Fast mode:** Use `/content/nnUNet_results` for speed.
- Required: periodic `rsync` backup to Drive and backup at session end.

### 8.6 Fast-Mode Backup

Run periodically and at session end:

```bash
mkdir -p "/content/drive/MyDrive/Foundation-nnUNet/nnUNet_results"
rsync -avh --progress "/content/nnUNet_results/" \
  "/content/drive/MyDrive/Foundation-nnUNet/nnUNet_results/"
```

### 8.7 Resume After Disconnect

The resume flag needs to be verified with the installed nnU-Net version.
Commonly:

```bash
nnUNetv2_train 101 2d 0 --c
```

> Verify exact resume flag with `nnUNetv2_train -h`. The `--c` flag is
> documented by nnU-Net but exact behavior and checkpoint filenames depend
> on the installed version.

### 8.8 Full Training Gate - Go / No-Go

| Check | Must Pass |
|---|---|
| All prior gates passed | yes |
| `nnUNet_results` reset to intended full-training root | yes |
| Existing `fold_0` folder is intentionally handled | yes |
| Training completes or reaches desired epoch count | yes |
| `checkpoint_final.pth` exists | yes |
| `checkpoint_best.pth` exists | yes |
| Validation output exists | yes |

---

## 9. Prediction

After successful fold-0 training.

### 9.1 Command

```bash
nnUNetv2_predict \
  -i "$nnUNet_raw/Dataset101_Pneumothorax/imagesTs" \
  -o "/content/drive/MyDrive/Foundation-nnUNet/artifacts/nnunet_predictions/Dataset101_Pneumothorax/fold0" \
  -d 101 \
  -c 2d \
  -f 0
```

> The `-f 0` fold flag needs to be verified with `nnUNetv2_predict -h`. Official
> nnU-Net docs document fold-specific inference. Confirm before running.

### 9.2 Verify Output

```bash
PRED="/content/drive/MyDrive/Foundation-nnUNet/artifacts/nnunet_predictions/Dataset101_Pneumothorax/fold0"

find "$PRED" -maxdepth 1 -type f -name '*.png' | wc -l
```

Expected: **1602** prediction files (one per `imagesTs` case).

### 9.3 Do Not Save Probabilities

Do **not** use `--save_probabilities` unless a later PR explicitly needs
probability maps. It increases storage use and is unnecessary for PR-3 baseline.

---

## 10. Evaluation Preparation

PR-3 does **not** implement evaluation. PR-8 will handle full metric comparison.

### 10.1 Preserve for PR-8

```bash
test -d "$nnUNet_raw/Dataset101_Pneumothorax/heldout_labelsTs" \
  && echo "heldout labels OK" || echo "MISSING: heldout labels"
test -f "$nnUNet_raw/Dataset101_Pneumothorax/case_mapping.csv" \
  && echo "case mapping OK" || echo "MISSING: case mapping"
test -d "/content/drive/MyDrive/Foundation-nnUNet/artifacts/nnunet_predictions/Dataset101_Pneumothorax/fold0" \
  && echo "predictions OK" || echo "MISSING: predictions"
```

### 10.2 What PR-8 Needs

| Artifact | Path |
|---|---|
| Heldout labels | `$nnUNet_raw/Dataset101_Pneumothorax/heldout_labelsTs/` |
| Predictions | `artifacts/nnunet_predictions/Dataset101_Pneumothorax/fold0/` |
| Case mapping | `$nnUNet_raw/Dataset101_Pneumothorax/case_mapping.csv` |
| Training command | Captured in report |
| Prediction command | Captured in report |
| Installed package versions | `pip_freeze_before_training.txt` saved to report |
| Checkpoint | `$nnUNet_results/.../fold_0/checkpoint_best.pth` |
| Training logs | `$nnUNet_results/.../fold_0/debug.json` |

### 10.3 Confirm Environment Snapshot

```bash
test -f /content/drive/MyDrive/Foundation-nnUNet/artifacts/reports/pip_freeze_before_training.txt
test -f /content/drive/MyDrive/Foundation-nnUNet/artifacts/reports/nnunetv2_train_help.txt
test -f /content/drive/MyDrive/Foundation-nnUNet/artifacts/reports/nnunetv2_predict_help.txt
test -f /content/drive/MyDrive/Foundation-nnUNet/artifacts/reports/nnunetv2_plan_help.txt
```

---

## 11. Go / No-Go Gate Summary

Full training is **explicitly forbidden** unless ALL prior gates pass.

| Gate | Section | Required for full training |
|---|---|---|
| GPU Gate | 3.1 | yes |
| nnU-Net CLI Gate | 3.3 | yes |
| Dataset Gate | 4 | yes |
| nnU-Net Gate (preprocessing) | 5 | yes |
| Split Gate | 6 | yes |
| Smoke Training Gate | 7 | yes |
| Full Training Gate | 8 | final |

**If any gate fails before full training, stop and resolve - do not proceed.**

---

## 12. Backup and Artifact Preservation

### 12.1 Save to Drive

- `nnUNet_raw/Dataset101_Pneumothorax/`
- `nnUNet_preprocessed/Dataset101_Pneumothorax/`
- `nnUNet_results/Dataset101_Pneumothorax/`
- `artifacts/nnunet_predictions/Dataset101_Pneumothorax/fold0/`
- `artifacts/reports/`

### 12.2 Do Not Commit to Git

- `nnUNet_raw/`
- `nnUNet_preprocessed/`
- `nnUNet_results/`
- `artifacts/nnunet_predictions/`
- Large `.pth`, `.npz`, preprocessed arrays, predictions, temporary Colab outputs.
- Generated logs (unless a small curated report is intentionally committed).

### 12.3 Recommended .gitignore Additions

These are recommendations only. Do not edit `.gitignore` in PR-3 unless that is
explicitly requested in a later change.

```gitignore
# nnU-Net working directories
/nnUNet_raw/
/nnUNet_preprocessed/
/nnUNet_results/

# nnU-Net prediction artifacts
/artifacts/nnunet_predictions/
```

### 12.4 Post-Training Backup Checklist

- [ ] `checkpoint_final.pth`
- [ ] `checkpoint_best.pth`
- [ ] `checkpoint_latest.pth` (if present)
- [ ] `debug.json`
- [ ] Training logs
- [ ] `progress.png`
- [ ] `validation/summary.json` (if present)
- [ ] `dataset.json`, `dataset_fingerprint.json`, `nnUNetPlans.json`
- [ ] `splits_final.json`
- [ ] Predictions folder
- [ ] `pip_freeze_before_training.txt` report
- [ ] `nnunetv2_train -h` output

---

## 13. B Plans - Common Failures

| No-Go | B Plan |
|---|---|
| No GPU | Switch to GPU runtime. Restart from section 3.1. Do not train on CPU. |
| PyTorch CUDA unavailable | Verify Colab runtime type. Reinstall compatible PyTorch only if needed, then reinstall/verify `nnunetv2`. |
| Dataset counts mismatch | Stop. Re-upload Dataset101 from local PR-2 export. Do not patch in Colab. |
| `labelsTs/` exists | Stop. Fix by re-exporting Dataset101 with correct PR-2 parameters. |
| Labels contain values outside {0,1} | Stop. Fix export or mask encoding upstream. |
| `NaturalImage2DIO` not recognized | Verify installed nnU-Net version. Official docs list `NaturalImage2DIO` for PNG. May indicate version/install mismatch. |
| Preprocessing fails | Inspect exact error. Verify naming, schema, reader, env vars. Rerun only after root cause is understood. |
| Split file wrong / missing | Re-copy from `splits_final_source.json`. Verify JSON shape and case IDs. Re-run Split Gate. |
| Smoke training OOM | Inspect plans and GPU memory. Consider smaller configuration only in a future explicit PR. |
| Smoke training NaN loss | Stop. Inspect labels, normalization, reader behavior, training logs. |
| Drive I/O too slow | Use `/content` for `nnUNet_preprocessed`. Consider `/content` for `nnUNet_results` only with `rsync` backup. |
| Colab disconnect risk | Use Drive-backed `nnUNet_results`. Keep checkpoints persistent. Resume with `--c` (verify exact flag). |
| `nnUNet_raw` env var missing | Re-run section 3.5. Must re-export after every restart. |

---

## 14. Quick-Reference Command Index

```bash
# --- Env vars (re-run after every restart) ---
export nnUNet_raw="/content/drive/MyDrive/Foundation-nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/content/drive/MyDrive/Foundation-nnUNet/nnUNet_preprocessed"
export nnUNet_results="/content/drive/MyDrive/Foundation-nnUNet/nnUNet_results"

# --- Dataset Gate checks ---
find "$nnUNet_raw/Dataset101_Pneumothorax/imagesTr" -maxdepth 1 -type f -name '*.png' | wc -l
find "$nnUNet_raw/Dataset101_Pneumothorax/labelsTr" -maxdepth 1 -type f -name '*.png' | wc -l
find "$nnUNet_raw/Dataset101_Pneumothorax/imagesTs" -maxdepth 1 -type f -name '*.png' | wc -l
find "$nnUNet_raw/Dataset101_Pneumothorax/heldout_labelsTs" -maxdepth 1 -type f -name '*.png' | wc -l
test ! -d "$nnUNet_raw/Dataset101_Pneumothorax/labelsTs" && echo "labelsTs absent OK"

# --- Preprocessing ---
nnUNetv2_plan_and_preprocess -d 101 --verify_dataset_integrity -c 2d

# --- Split placement ---
cp "$nnUNet_raw/Dataset101_Pneumothorax/splits_final_source.json" \
   "$nnUNet_preprocessed/Dataset101_Pneumothorax/splits_final.json"

# --- Pre-training reports ---
pip freeze > /content/drive/MyDrive/Foundation-nnUNet/artifacts/reports/pip_freeze_before_training.txt
nnUNetv2_train -h > /content/drive/MyDrive/Foundation-nnUNet/artifacts/reports/nnunetv2_train_help.txt
nnUNetv2_predict -h > /content/drive/MyDrive/Foundation-nnUNet/artifacts/reports/nnunetv2_predict_help.txt
nnUNetv2_plan_and_preprocess -h > /content/drive/MyDrive/Foundation-nnUNet/artifacts/reports/nnunetv2_plan_help.txt

# --- Smoke training ---
export nnUNet_results="/content/drive/MyDrive/Foundation-nnUNet/nnUNet_results_smoke"
nnUNetv2_train 101 2d 0 -device cuda

# --- Full training ---
export nnUNet_results="/content/drive/MyDrive/Foundation-nnUNet/nnUNet_results"
test ! -d "$nnUNet_results/Dataset101_Pneumothorax/nnUNetTrainer__nnUNetPlans__2d/fold_0" && echo "clean full-training result folder OK" || echo "WARNING: existing fold_0 result folder exists"
nnUNetv2_train 101 2d 0 -device cuda

# --- Resume ---
nnUNetv2_train 101 2d 0 --c

# --- Prediction ---
nnUNetv2_predict \
  -i "$nnUNet_raw/Dataset101_Pneumothorax/imagesTs" \
  -o "/content/drive/MyDrive/Foundation-nnUNet/artifacts/nnunet_predictions/Dataset101_Pneumothorax/fold0" \
  -d 101 \
  -c 2d \
  -f 0

# --- Backup (fast mode) ---
rsync -avh --progress "/content/nnUNet_results/" \
  "/content/drive/MyDrive/Foundation-nnUNet/nnUNet_results/"

# --- Confirm pre-training report capture ---
test -f /content/drive/MyDrive/Foundation-nnUNet/artifacts/reports/pip_freeze_before_training.txt
test -f /content/drive/MyDrive/Foundation-nnUNet/artifacts/reports/nnunetv2_train_help.txt
test -f /content/drive/MyDrive/Foundation-nnUNet/artifacts/reports/nnunetv2_predict_help.txt
test -f /content/drive/MyDrive/Foundation-nnUNet/artifacts/reports/nnunetv2_plan_help.txt
```

---

## 15. Appendix: Version-Sensitive Flags

The following flags/options are documented by nnU-Net but depend on the
**installed version**. Verify each with its `-h` output before use:

| Flag / Feature | Verify With |
|---|---|
| `-device cuda` | `nnUNetv2_train -h` |
| `--c` (continue/resume) | `nnUNetv2_train -h` |
| `-f 0` (fold selection) | `nnUNetv2_predict -h` |
| `NaturalImage2DIO` support | `nnUNetv2_plan_and_preprocess -h` and import check |
| Short-run / epoch-limit options | `nnUNetv2_train -h` |

Section 3.4 captures package versions and CLI help output before training. Do
not wait until after smoke or full training to save this information.

---

> **Reminder:** This is a documentation-only PR. No training/model/preprocessing/export code has been modified.
> PR-3 is ready for Colab execution once Dataset101 is uploaded to Drive and all gates are verified.
