# Hybrid Adapter Sanity Test Report

## Repository findings

- Existing hybrid model class: `src/models/hybrid.py::HybridFoundationUNet`
- Existing backbone loader: `src/models/backbone.py::FoundationXBackbone`
- Existing scale contract helper: `assert_corrected_hybrid_scale_contract()`
- Existing related tests: `tests/test_hybrid_gradient_flow.py`, `tests/test_hybrid_scale_contract.py`
- Existing Foundation X smoke artifacts found under `artifacts/diagnostics/foundation_x_smoke/`

## Setup

```
python scripts/sanity_hybrid_adapter.py --input_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\nnUNet_raw\Dataset101_Pneumothorax\imagesTs --labels_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\nnUNet_raw\Dataset101_Pneumothorax\heldout_labelsTs --foundation_checkpoint C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\checkpoints\foundation_x.pth --output_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_adapter_sanity --img_size 512 --device auto --num_cases 8 --strict
```

- Device: `cpu`
- Number of cases requested: 8
- Number of cases processed: 8
- Input dir: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\nnUNet_raw\Dataset101_Pneumothorax\imagesTs`
- Labels dir: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\nnUNet_raw\Dataset101_Pneumothorax\heldout_labelsTs`
- Foundation checkpoint: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\checkpoints\foundation_x.pth`
- Foundation checkpoint SHA-256: `9b7eb0215b8fdb50d1413fee306e70ed8ac7b646cff342dbce78ca44fd0f32c3`
- Model config: frozen_backbone=True, img_size=512

## Forward pass results

- Input shape sample: `[1, 1, 512, 512]`
- Foundation X feature shapes sample: `[[1, 128, 128, 128], [1, 256, 64, 64], [1, 512, 32, 32], [1, 1024, 16, 16]]`
- Adapter/fusion shapes sample: `{'fusion_e3': {'input_shape': [[1, 128, 128, 128], [1, 256, 128, 128]], 'output_shape': [1, 256, 128, 128]}, 'fusion_e4': {'input_shape': [[1, 256, 64, 64], [1, 512, 64, 64]], 'output_shape': [1, 512, 64, 64]}, 'h16_fx_context': {'input_shape': [[1, 512, 32, 32], [1, 1024, 32, 32]], 'output_shape': [1, 1024, 32, 32]}, 'h32_context_head': {'input_shape': [[1, 1024, 16, 16]], 'output_shape': [1, 1024, 16, 16]}, 'h32_to_h16': {'input_shape': [[1, 1024, 16, 16]], 'output_shape': [1, 1024, 32, 32]}, 'context_merge': {'input_shape': [[1, 2048, 32, 32]], 'output_shape': [1, 1024, 32, 32]}, 'dec4': {'input_shape': [[1, 1024, 64, 64]], 'output_shape': [1, 512, 64, 64]}, 'dec3': {'input_shape': [[1, 512, 128, 128]], 'output_shape': [1, 256, 128, 128]}, 'dec2': {'input_shape': [[1, 256, 256, 256]], 'output_shape': [1, 128, 256, 256]}, 'dec1': {'input_shape': [[1, 128, 512, 512]], 'output_shape': [1, 64, 512, 512]}}`
- Output probability shape sample: `[1, 1, 512, 512]`
- Raw logits shape sample (from model.final hook): `[1, 1, 512, 512]`
- Probability stats aggregate: min=0.520046, max=0.526028, mean=0.522675, std=0.000374
- Raw logits stats aggregate: min=0.080226, max=0.104207, mean=0.090763, std=0.001500
- NaN/Inf status: prob_nan_total=0, prob_inf_total=0, logits_nan_total=0, logits_inf_total=0

## Loss and gradient results

- Loss functions used: `DiceFocalLoss` on sigmoid output and `BCEWithLogitsLoss` on raw logits
- DiceFocal loss value: 0.9132792353630066
- BCEWithLogits loss value: 0.7540329098701477
- Total backward loss value: 1.6673121452331543
- Frozen backbone gradient check: True
- Trainable adapter/decoder gradient check: True
- Finite gradient check: True

## Visual diagnostics

- Visual files saved: 24
- Brief note: diagnostic overlays/probability/logit maps are saved for non-empty/non-degenerate sanity only.
- No performance claims are made in this report.

## Failure cases

None.

## Decision

- Status: `PASS`
- Next recommended PR: `PR-6 Tiny Hybrid Training Smoke`