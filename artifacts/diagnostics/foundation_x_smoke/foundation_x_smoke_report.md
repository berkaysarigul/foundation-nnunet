# Foundation X Smoke Test Report

## Repository findings

- **Foundation X model class:** `FoundationXBackbone` in `src/models/backbone.py`
- **Checkpoint:** `checkpoints/foundation_x.pth` (Swin-B, embed_dim=128, patch_size=4, window_size=7)
- **Checkpoint top-level key:** `ckpt['model']`
- **Key prefix stripped:** `backbone.0.` (note: `docs/foundation_nnunet_dev_guide.md` incorrectly lists `backbone.` — not authoritative per D-046)
- **Key remapping:** `_remap_key()` in `src/models/backbone.py` — converts `layers.N.*` → `layers_N.*`, shifts downsample index to N+1
- **Expected input:** `(B, 1, H, W)` grayscale float32 in `[0,1]`; H=W=img_size (256 or 512)
- **Internal normalization:** grayscale→RGB repeat then ImageNet mean/std (D-062/D-070)
- **Output:** list of 4 spatial feature maps `(B, {128,256,512,1024}, H/{4,8,16,32}, W/{4,8,16,32})`
- **No standalone smoke script existed prior to PR-4** (all existing tests stub the checkpoint)
- **Governance note:** PR-4 operates on the nnU-Net v2/Dataset101 track. Recovery memory updated in §P1.13 / D-073 / VALIDATION_CHECKLIST §20.

## Smoke test setup

```
python scripts/smoke_foundation_x.py \
  --input_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\nnUNet_raw\Dataset101_Pneumothorax\imagesTs \
  --labels_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\nnUNet_raw\Dataset101_Pneumothorax\heldout_labelsTs \
  --checkpoint C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\checkpoints\foundation_x.pth \
  --output_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\foundation_x_smoke \
  --img_size 512 \
  --device cpu \
  --num_cases 12
```

- **Device:** `cpu`
- **Cases requested:** 12
- **Cases processed:** 12
- **Case list source:** `priority_plus_filler`
- **Checkpoint SHA-256:** `9b7eb0215b8fdb50d1413fee306e70ed8ac7b646cff342dbce78ca44fd0f32c3`
- **Checkpoint size:** 2,816,610,642 bytes
- **Preprocessing:** grayscale uint8 PNG → /255.0 → float32 [0,1] → (1,1,512,512) → internal RGB repeat + ImageNet norm

## Results

**Load status:** OK

- Loaded 449 backbone keys (prefix `backbone.0.`)
- Missing keys vs. timm state dict: **0**
- Unexpected keys vs. timm state dict: **124**

**Per-stage inference summary** (aggregated across all processed cases):

| Stage | Channels | Spatial | Shape OK | Mean | Std | NaN total | Inf total |
|-------|----------|---------|----------|------|-----|-----------|-----------|
| 0 | 128 | H/4 | ✓ | -0.2663 | 1.1180 | 0 | 0 |
| 1 | 256 | H/8 | ✓ | 0.0663 | 1.5652 | 0 | 0 |
| 2 | 512 | H/16 | ✓ | 0.1692 | 11.5286 | 0 | 0 |
| 3 | 1024 | H/32 | ✓ | 0.0084 | 12.0586 | 0 | 0 |

**Total inference wall time:** 16.1s for 12 cases

**Visual artifacts:** `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\foundation_x_smoke\visuals/`

## Failure cases

None.

**Warnings:**

- Large unexpected_keys_count=124; checkpoint has extra keys.

## Decision

**Status: `PASS`**

Pass criteria:

- [✓] `checkpoint_loads`
- [✓] `num_cases_ok`
- [✓] `shapes_match_contract`
- [✓] `features_finite`
- [✓] `features_non_degenerate`

> **Framing boundary (D-035 / D-040–D-042):** This report documents only that the
> Foundation X code path loads and produces finite, image-aligned feature maps on
> Dataset101 imagery. It does not constitute evidence that Foundation X generalizes
> to unseen data or is the superior model. The authoritative comparison anchor
> remains `pretrained_resnet34_unet` at held-out positive-only Dice 0.4951.

## Next recommended PR

**PR-5: Hybrid adapter sanity test.**

Foundation X loads cleanly and produces well-formed, finite, spatially-structured
feature maps on Dataset101 imagery. The next step is to confirm that the
`HybridFoundationUNet` (D-054–D-059 scale mapping) runs a correct forward pass
end-to-end and that the fused outputs are valid segmentation logits.