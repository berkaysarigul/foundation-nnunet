# PR-11 SIIM-ACR + PTX-498 Foundation X / nnU-Net v2 Pipeline

This document records the runnable PR-11 command sequence. It intentionally
does not launch training by itself.

## Data Preparation

```bash
py -3 scripts/extract_raw_datasets.py \
  --siim-zip data/raw/SIIM-ACR.zip \
  --ptx-zip data/raw/PTX-498.zip \
  --out-root data/extracted \
  --force false

py -3 scripts/inventory_siim_acr_png.py \
  --root data/extracted/SIIM-ACR \
  --out artifacts/siim_original/audits/inventory

py -3 scripts/build_siim_acr_png_manifest.py \
  --root data/extracted/SIIM-ACR \
  --out artifacts/siim_original/manifests \
  --allow-conflicts false \
  --hash-files false

py -3 scripts/audit_siim_acr_png_manifest.py \
  --manifest artifacts/siim_original/manifests/siim_acr_png_manifest.csv \
  --out artifacts/siim_original/audits/manifest_audit \
  --sample-count 24 \
  --seed 42
```

```bash
py -3 scripts/inventory_ptx498.py \
  --root data/extracted/PTX-498 \
  --out artifacts/ptx498/audits/inventory

py -3 scripts/build_ptx498_manifest.py \
  --root data/extracted/PTX-498 \
  --out artifacts/ptx498/manifests \
  --hash-files false

py -3 scripts/audit_ptx498_manifest.py \
  --manifest artifacts/ptx498/manifests/ptx498_manifest.csv \
  --out artifacts/ptx498/audits/manifest_audit
```

## Leakage Analysis

```bash
py -3 scripts/compare_siim_with_dataset101.py \
  --siim-manifest artifacts/siim_original/manifests/siim_acr_png_manifest.csv \
  --dataset101-root nnUNet_raw/Dataset101_Pneumothorax \
  --out artifacts/siim_original/audits/dataset101_overlap \
  --hash-mode normalized_thumbnail \
  --seed 42
```

Protocol B is the primary PR-11 protocol: train/validate from SIIM
`stage_1_train`, evaluate on SIIM `stage_1_test`. Protocol A is the bridge:
exclude Dataset101 heldout overlaps from SIIM training and evaluate on
Dataset101 heldout.

## Foundation X Prior Export

```bash
py -3 scripts/export_fx_priors_from_manifest.py \
  --manifest artifacts/siim_original/manifests/siim_acr_png_manifest.csv \
  --image-col image_path \
  --label-col mask_path \
  --case-id-col case_id \
  --checkpoint checkpoints/foundation_x.pth \
  --state-key teacher_model \
  --head 5 \
  --preprocess-variant official_siim_224 \
  --device cuda \
  --out artifacts/siim_original/fx_priors/teacher_head5_official224 \
  --save-prob-maps true \
  --save-aligned-prob-maps true \
  --save-binary-masks false \
  --save-visuals false \
  --threshold 0.5

py -3 scripts/validate_fx_prior_manifests.py \
  --manifest artifacts/siim_original/fx_priors/teacher_head5_official224/fx_prior_manifest.csv \
  --dataset-root data/extracted/SIIM-ACR \
  --out artifacts/siim_original/audits/fx_prior_validation \
  --expected-state-key teacher_model \
  --expected-preprocess-variant official_siim_224 \
  --expected-head-key head_5 \
  --sample-size 64 \
  --strict
```

Repeat the same export/validation with the PTX manifest and
`artifacts/ptx498/fx_priors/teacher_head5_official224`.

## nnU-Net Dataset Creation

```bash
py -3 scripts/create_nnunetv2_dataset_from_fx_prior.py \
  --siim-manifest artifacts/siim_original/manifests/siim_acr_png_manifest.csv \
  --protocol siim_stage \
  --dataset-id 200 \
  --dataset-name SIIMACR_ImageOnly \
  --nnunet-raw nnUNet_raw \
  --channels image \
  --split-source stage1_train_test \
  --val-fraction 0.2 \
  --seed 42 \
  --overwrite false

py -3 scripts/create_nnunetv2_dataset_from_fx_prior.py \
  --siim-manifest artifacts/siim_original/manifests/siim_acr_png_manifest.csv \
  --fx-prior-manifest artifacts/siim_original/fx_priors/teacher_head5_official224/fx_prior_manifest.csv \
  --protocol siim_stage \
  --dataset-id 201 \
  --dataset-name SIIMACR_FXPrior \
  --nnunet-raw nnUNet_raw \
  --channels image,fx_prior \
  --split-source stage1_train_test \
  --val-fraction 0.2 \
  --seed 42 \
  --overwrite false
```

## Training Commands

```bash
nnUNetv2_plan_and_preprocess -d 200 --verify_dataset_integrity -c 2d
nnUNetv2_plan_and_preprocess -d 201 --verify_dataset_integrity -c 2d

cp nnUNet_raw/Dataset201_SIIMACR_FXPrior/splits_final_source.json \
   nnUNet_preprocessed/Dataset201_SIIMACR_FXPrior/splits_final.json

nnUNetv2_train 200 2d 0 -device cuda
nnUNetv2_train 201 2d 0 -device cuda

nnUNetv2_predict \
  -i nnUNet_raw/Dataset201_SIIMACR_FXPrior/imagesTs \
  -o artifacts/nnunet_v2/predictions/siim_stage_protocol/Dataset201/fold0 \
  -d 201 -c 2d -f 0 -device cuda --save_probabilities
```

Run 5-fold CV, TTA, and ensembles only after the fold 0 image-only and
image+prior reports are locked.

## Evaluation and Visuals

```bash
py -3 scripts/evaluate_segmentation_predictions.py \
  --pred-dir artifacts/nnunet_v2/predictions/siim_stage_protocol/Dataset201/fold0 \
  --label-dir nnUNet_raw/Dataset201_SIIMACR_FXPrior/heldout_labelsTs \
  --out artifacts/nnunet_v2/reports/siim_stage_protocol/Dataset201/fold0 \
  --selected-threshold 0.5 \
  --selected-min-pixels 0 \
  --size-bin-source-manifest artifacts/siim_original/manifests/siim_acr_png_manifest.csv

py -3 scripts/create_segmentation_comparison_panels.py \
  --manifest artifacts/siim_original/manifests/siim_acr_png_manifest.csv \
  --fx-prior-manifest artifacts/siim_original/fx_priors/teacher_head5_official224/fx_prior_manifest.csv \
  --image-only-preds artifacts/nnunet_v2/predictions/siim_stage_protocol/Dataset200/fold0 \
  --image-prior-preds artifacts/nnunet_v2/predictions/siim_stage_protocol/Dataset201/fold0 \
  --metrics artifacts/nnunet_v2/reports/siim_stage_protocol/comparison_case_metrics.csv \
  --out artifacts/nnunet_v2/visuals/siim_stage_protocol
```

PTX-498 is external validation only. Do not tune threshold, postprocessing, or
checkpoint selection on PTX-498 in PR-11.

