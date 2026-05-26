# PR-9A-FIX Stage A/B Official Foundation_X Direct Segmentation

## Status

- official commit: `5e0b473cf3f595320ce982697d4365e1d9a4f642`
- state keys requested: `model, teacher_model`
- state keys loaded: `model, teacher_model`
- selected cases: 2
- Stage B rows: 36
- non-trivial mask rows: 19
- consider Stage C: True

> Current reverse-engineered PR-9A predictions remain invalid as nnU-Net priors.

## Checkpoint Loading

- `model`: loaded, compatible keys 449, missing 2, unexpected 0, shape mismatches 0
- `teacher_model`: loaded, compatible keys 449, missing 2, unexpected 0, shape mismatches 0

## Stage A Output Shapes

### model
- `head_5` raw [1, 1, 224, 224] selected [1, 1, 224, 224]
- `head_2` raw [1, 1, 224, 224] selected [1, 1, 224, 224]
- `head_4_ch12` raw [1, 13, 224, 224] selected [1, 1, 224, 224]
### teacher_model
- `head_5` raw [1, 1, 224, 224] selected [1, 1, 224, 224]
- `head_2` raw [1, 1, 224, 224] selected [1, 1, 224, 224]
- `head_4_ch12` raw [1, 13, 224, 224] selected [1, 1, 224, 224]

## Stage B Metrics

| state | preprocess | head | gt+ | pred+ | Dice | prob std | pred px |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| model | official_siim_224 | head_5 | 1 | 1 | 0.4244 | 0.031231 | 66 |
| model | official_siim_224 | head_2 | 1 | 1 | 0.3045 | 0.025780 | 44 |
| model | official_siim_224 | head_4_ch12 | 1 | 0 | 0.0000 | 0.000112 | 0 |
| model | official_siim_512 | head_5 | 1 | 1 | 0.7954 | 0.074104 | 1521 |
| model | official_siim_512 | head_2 | 1 | 1 | 0.7872 | 0.073020 | 1474 |
| model | official_siim_512 | head_4_ch12 | 1 | 0 | 0.0000 | 0.004378 | 0 |
| model | current_pr9a_512 | head_5 | 1 | 1 | 0.4042 | 0.034119 | 353 |
| model | current_pr9a_512 | head_2 | 1 | 1 | 0.4308 | 0.037541 | 415 |
| model | current_pr9a_512 | head_4_ch12 | 1 | 0 | 0.0000 | 0.000150 | 0 |
| model | official_siim_224 | head_5 | 0 | 0 | 1.0000 | 0.000000 | 0 |
| model | official_siim_224 | head_2 | 0 | 0 | 1.0000 | 0.000000 | 0 |
| model | official_siim_224 | head_4_ch12 | 0 | 0 | 1.0000 | 0.000954 | 0 |
| model | official_siim_512 | head_5 | 0 | 0 | 1.0000 | 0.000001 | 0 |
| model | official_siim_512 | head_2 | 0 | 0 | 1.0000 | 0.000000 | 0 |
| model | official_siim_512 | head_4_ch12 | 0 | 1 | 0.0000 | 0.076179 | 2033 |
| model | current_pr9a_512 | head_5 | 0 | 0 | 1.0000 | 0.000001 | 0 |
| model | current_pr9a_512 | head_2 | 0 | 0 | 1.0000 | 0.000001 | 0 |
| model | current_pr9a_512 | head_4_ch12 | 0 | 1 | 0.0000 | 0.016134 | 74 |
| teacher_model | official_siim_224 | head_5 | 1 | 1 | 0.5376 | 0.042060 | 101 |
| teacher_model | official_siim_224 | head_2 | 1 | 1 | 0.3557 | 0.028595 | 53 |
| teacher_model | official_siim_224 | head_4_ch12 | 1 | 0 | 0.0000 | 0.008549 | 0 |
| teacher_model | official_siim_512 | head_5 | 1 | 1 | 0.7758 | 0.079719 | 1736 |
| teacher_model | official_siim_512 | head_2 | 1 | 1 | 0.7795 | 0.074909 | 1532 |
| teacher_model | official_siim_512 | head_4_ch12 | 1 | 1 | 0.0000 | 0.095623 | 3114 |
| teacher_model | current_pr9a_512 | head_5 | 1 | 1 | 0.5595 | 0.048074 | 634 |
| teacher_model | current_pr9a_512 | head_2 | 1 | 1 | 0.5369 | 0.047039 | 610 |
| teacher_model | current_pr9a_512 | head_4_ch12 | 1 | 1 | 0.0000 | 0.057475 | 1166 |
| teacher_model | official_siim_224 | head_5 | 0 | 0 | 1.0000 | 0.000000 | 0 |
| teacher_model | official_siim_224 | head_2 | 0 | 0 | 1.0000 | 0.000000 | 0 |
| teacher_model | official_siim_224 | head_4_ch12 | 0 | 1 | 0.0000 | 0.197920 | 2283 |
| teacher_model | official_siim_512 | head_5 | 0 | 0 | 1.0000 | 0.000000 | 0 |
| teacher_model | official_siim_512 | head_2 | 0 | 0 | 1.0000 | 0.000000 | 0 |
| teacher_model | official_siim_512 | head_4_ch12 | 0 | 1 | 0.0000 | 0.339801 | 38354 |
| teacher_model | current_pr9a_512 | head_5 | 0 | 0 | 1.0000 | 0.000000 | 0 |
| teacher_model | current_pr9a_512 | head_2 | 0 | 0 | 1.0000 | 0.000000 | 0 |
| teacher_model | current_pr9a_512 | head_4_ch12 | 0 | 1 | 0.0000 | 0.322604 | 33811 |

## Colab Stage C Command

```bash
python scripts/foundation_x_official_direct_segment.py \
  --dataset-root nnUNet_raw/Dataset101_Pneumothorax \
  --checkpoint checkpoints/foundation_x.pth \
  --state-keys model teacher_model \
  --heads 5 2 \
  --head4-channel 12 \
  --preprocess-variants official_siim_224 official_siim_512 current_pr9a_512 \
  --device cuda \
  --max-cases 20 \
  --positive-cases 10 \
  --negative-cases 10 \
  --case-sampling balanced \
  --out artifacts/diagnostics/foundation_x_official_direct/sanity_20case \
  --save-visuals true \
  --save-histograms true
```
