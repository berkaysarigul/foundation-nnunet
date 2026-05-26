# Foundation X Direct Segmentation — Head Sweep Report

## Scope

- This is **direct Foundation X segmentation inference**, not training.
- This run is a **head sweep** to identify which segmentation head best matches
  SIIM-ACR pneumothorax on Dataset101 heldout.
- The official Foundation X README states that the best SIIM-ACR segmentation
  model is `(Student) 896`, so `checkpoint["model"]` is the primary state dict.
- `checkpoint["teacher_model"]` may be evaluated separately but should not
  replace `model` unless it scores better and is explicitly documented.
- Head 4 (`segmentation_heads.4.weight` shape `(13, 128, 3, 3)`) is multiclass
  (likely ChestX-Det) and is **skipped** from the binary head sweep.
- No claim is made that the best head is the official SIIM head. The mapping
  must be confirmed from the Foundation X config or repository code before it
  is treated as authoritative (D-035 framing boundary).

## Framing boundary (D-035 / D-040–D-042)

Foundation X is a SIIM-exposed pretraining source under the current checkpoint
provenance. Any number below is leakage-aware diagnostic evidence and must be
compared back to the trusted `pretrained_resnet34_unet` baseline
(D-042 anchor: held-out positive-only Dice mean **0.4951**).

## Run setup

- checkpoint: `checkpoints\foundation_x.pth`
- checkpoint SHA-256: `48332e84df124c6434a502fc0b93f7eee8183e702b673eccb9a4222b78cb69a6`
- selected state key: `model`
- top-level keys: `['model', 'teacher_model', 'optimizer', 'epoch']`
- dataset root: `nnUNet_raw\Dataset101_Pneumothorax`
- cases evaluated: 2
- heads evaluated: `[0, 1, 2, 3, 5]`
- heads skipped (non-binary): `[]`
- thresholds: `[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]`
- device: `cpu`
- img_size: `512`

## Checkpoint load diagnostics

- raw_total: 2340
- raw_with_prefix `backbone.0.`: 449
- routed_swin: 351
- routed_segnorm: 8
- routed_locnorm: 6
- routed_ppn: 30
- routed_fpn: 18
- routed_heads: 12
- skipped_non_backbone (e.g. teacher_model, optimizer, epoch): 1891
- skipped_classification_heads: 22
- skipped_swin_head: 2
- missing keys (model expected but checkpoint did not provide): 0
- unexpected keys (checkpoint had, model does not declare): 26
- sample unexpected keys:
  - `swin.norm.weight`
  - `swin.norm.bias`
  - `swin.layers_0.blocks.0.attn.relative_position_index`
  - `swin.layers_0.blocks.1.attn.relative_position_index`
  - `swin.layers_1.blocks.0.attn.relative_position_index`
  - `swin.layers_1.blocks.1.attn.relative_position_index`
  - `swin.layers_2.blocks.0.attn.relative_position_index`
  - `swin.layers_2.blocks.1.attn.relative_position_index`
  - `swin.layers_2.blocks.2.attn.relative_position_index`
  - `swin.layers_2.blocks.3.attn.relative_position_index`

## Head-by-head results

| head | best threshold | pos Dice (pos cases) | det. pos Dice | mean Dice all | case F1 | case recall | case spec | neg FPR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.05 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 1 | 0.05 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 2 | 0.05 | n/a | n/a | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 3 | 0.05 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 5 | 0.05 | n/a | n/a | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |

## Best overall (by mean Dice on positive cases)

- head: **0**
- threshold: **0.05**
- positive-cases mean Dice: **n/a**

> The 'best head' label here is a fit to the evaluated subset only.
> It is NOT confirmed to be the official SIIM-ACR pneumothorax head until
> the head-to-task mapping is read from the Foundation X repository.

## Local dry-run command

```
python scripts/foundation_x_direct_segment.py \
  --dataset-root nnUNet_raw/Dataset101_Pneumothorax \
  --checkpoint checkpoints/ckpt_E896_TH15.pth \
  --state-key model \
  --heads 0 1 2 3 5 \
  --device cpu \
  --max-cases 2 \
  --out artifacts/diagnostics/foundation_x_direct/dryrun_cpu \
  --save-visuals true
```

## Intended Colab full-run command (DO NOT run locally)

```
python scripts/foundation_x_direct_segment.py \
  --dataset-root nnUNet_raw/Dataset101_Pneumothorax \
  --checkpoint checkpoints/ckpt_E896_TH15.pth \
  --state-key model \
  --heads 0 1 2 3 5 \
  --device cuda \
  --out artifacts/diagnostics/foundation_x_direct/head_sweep \
  --save-prob-maps true \
  --save-binary-masks true \
  --save-visuals true
```

## Reverse-engineering caveats

- `src/models/foundation_x_full.py` is a **best-effort reverse-engineered
  reconstruction** of the Foundation X model from the checkpoint key shapes.
  The forward graph for `segmentation_PPN` and `segmentation_FPN` is
  inferred from the parameter shapes (PSPNet-style PPM + UPerNet-style top-down
  FPN with conv_fusion) and may differ in details from the official
  JLiangLab/Foundation_X implementation.
- If the load diagnostic above reports a small `missing` set covering only
  optional Swin/timm-versioning keys and a `routed_*` count matching the
  inventory in `foundationx_checkpoint_inventory.txt`, the segmentation path
  is loaded; otherwise the inference numbers should be treated as
  diagnostic-only.
