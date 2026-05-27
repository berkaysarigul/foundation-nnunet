# PR-10A Official Foundation_X Training Prior Export

## Status

- official commit: `5e0b473cf3f595320ce982697d4365e1d9a4f642`
- split: `train`
- images dir: `nnUNet_raw\Dataset101_Pneumothorax\imagesTr`
- labels dir: `nnUNet_raw\Dataset101_Pneumothorax\labelsTr`
- state keys requested: `teacher_model`
- state keys loaded: `teacher_model`
- heads evaluated: `head_5`
- preprocess variants: `official_siim_224`
- selected cases: 2
- Stage B rows: 2
- manifest: `artifacts\diagnostics\foundation_x_official_direct\teacher_head5_official224_export_imagesTr_dryrun\teacher_head5_prior_manifest_train.csv`
- export validation: `passed`
- non-trivial mask rows: 0
- consider Stage C: False

> Current reverse-engineered PR-9A predictions remain invalid as nnU-Net priors. Only confirmed official Foundation_X exports should feed later PR-10 refiner work.

## Checkpoint Loading

- `teacher_model`: loaded, compatible keys 449, missing 2, unexpected 0, shape mismatches 0

## Export Validation

- row count: 2
- expected rows: 2
- unique case IDs: 2
- expected Dataset101 train cases: 9073
- full train export requested: False

## Stage A Output Shapes

### teacher_model
- `head_5` raw [1, 1, 224, 224] selected [1, 1, 224, 224]

## Stage B Metrics

| state | preprocess | head | gt+ | pred+ | Dice | prob std | pred px |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| teacher_model | official_siim_224 | head_5 | 0 | 0 | 1.0000 | 0.000000 | 0 |
| teacher_model | official_siim_224 | head_5 | 0 | 0 | 1.0000 | 0.000000 | 0 |

## Colab Full Training-Prior Export Command

```bash
python scripts/foundation_x_official_direct_segment.py \
  --dataset-root /content/nnUNet_raw/Dataset101_Pneumothorax \
  --images-dir /content/nnUNet_raw/Dataset101_Pneumothorax/imagesTr \
  --labels-dir /content/nnUNet_raw/Dataset101_Pneumothorax/labelsTr \
  --split-name train \
  --checkpoint checkpoints/foundation_x.pth \
  --state-keys teacher_model \
  --heads 5 \
  --preprocess-variants official_siim_224 \
  --device cuda \
  --max-cases 9073 \
  --case-sampling sequential \
  --out artifacts/diagnostics/foundation_x_official_direct/teacher_head5_official224_export_imagesTr \
  --save-prob-maps true \
  --save-binary-masks true \
  --save-visuals false \
  --save-histograms false \
  --threshold 0.5
```
