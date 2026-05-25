# Controlled Short Hybrid Training Report

## Repository findings

- Existing hybrid path reused: `src/models/hybrid.py::HybridFoundationUNet`.
- Existing backbone path reused: `src/models/backbone.py::FoundationXBackbone`.
- Existing PR-6 diagnostics were extended for longer bounded training and scheduled validation.
- The full single-split runner was intentionally not reused because it writes to `artifacts/runs/` and exceeds PR-7 scope.

## Setup

```
py scripts/controlled_train_hybrid_short.py --input_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\nnUNet_raw\Dataset101_Pneumothorax\imagesTr --labels_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\nnUNet_raw\Dataset101_Pneumothorax\labelsTr --foundation_checkpoint C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\checkpoints\foundation_x.pth --output_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_bce01 --img_size 512 --device auto --num_train_cases 32 --num_val_cases 16 --max_steps 25 --batch_size 1 --lr 0.0001 --bce_loss_weight 0.1 --val_every 5 --strict
```

- Device: `cpu`
- Train cases: 32 (pos=16, neg=16)
- Val cases: 16 (pos=8, neg=8)
- Max steps: 25
- Validation interval: 5
- Batch size: 1
- Learning rate: 0.0001
- BCE loss weight: 0.1
- Frozen backbone: True

## Dataset selection

- Train/val disjoint overlap count: 0
- Train case IDs count: 32
- Val case IDs count: 16

## Training results

- Steps completed: 25 / 25
- Train total loss series: [1.1021711826324463, 1.113381028175354, 1.071927547454834, 1.1124826669692993, 1.0302075147628784, 1.1099886894226074, 1.0982310771942139, 1.1087095737457275, 1.0797171592712402, 1.1066720485687256, 1.1027095317840576, 1.103266716003418, 1.0836812257766724, 1.0915440320968628, 1.059152603149414, 1.0205672979354858, 1.0107439756393433, 0.9978693127632141, 1.3940315246582031, 0.9394320249557495, 3.963495969772339, 0.9520395994186401, 1.7216176986694336, 0.9832298159599304, 1.11020028591156]
- Train predicted positive pixel ratio series: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8441734313964844, 0.55133056640625, 0.3930816650390625, 0.1091461181640625, 0.0139923095703125, 0.003139495849609375, 0.00016021728515625, 1.1444091796875e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- Train non-empty prediction rate series: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- Probability output shape sample: `[1, 1, 512, 512]`
- Raw logits shape sample: `[1, 1, 512, 512]`
- NaN/Inf totals: prob_nan=0, prob_inf=0, logits_nan=0, logits_inf=0

## Validation results

- Validation schedule steps: [0, 5, 10, 15, 20, 25]
- Validation executed steps: [0, 5, 10, 15, 20, 25]
- Validation mean total loss per step: [1.105231799185276, 1.1014695167541504, 1.0959687381982803, 1.0205119624733925, 1.2480593211948872, 1.029782708734274]
- Validation Dice mean per step: [0.009647867700550705, 0.009647867700550705, 0.010428256726299878, 0.0, 0.5, 0.5]
- Validation probability min per step: [0.5203309655189514, 0.5092699527740479, 0.48972275853157043, 0.04961617663502693, 0.0, 6.537480279272502e-23]
- Validation probability max per step: [0.5255031585693359, 0.5208638310432434, 0.5194992423057556, 0.5148447155952454, 0.3622772991657257, 0.47199368476867676]
- Validation probability mean per step: [0.5226851312996246, 0.5145571641870674, 0.5021462196871482, 0.18948832259282788, 1.7209460767766746e-05, 0.0016825656862255016]
- Validation probability std per step: [0.0004404916687739596, 0.002479832112126958, 0.0056717317395634835, 0.07060653843943955, 0.001479335259767842, 0.01364608194448868]
- Validation raw logit min per step: [0.08136864751577377, 0.03708402439951897, -0.04111487790942192, -2.9525489807128906, -447.8597717285156, -51.081905364990234]
- Validation raw logit max per step: [0.1021011620759964, 0.08350377529859543, 0.0780365839600563, 0.059396181255578995, -0.565493643283844, -0.11214268207550049]
- Validation raw logit mean per step: [0.09080293469134659, 0.05824657064865324, 0.008586185655457775, -1.50791009846335, -176.15391848490248, -20.767606094920446]
- Validation raw logit std per step: [0.001765603372757835, 0.009928232841642534, 0.022689601028436583, 0.4255184361528427, 87.58499411417039, 9.563004318955404]
- Validation probability foreground ratio @0.5 per step: [1.0, 1.0, 0.6204278469085693, 0.00010204315185546875, 0.0, 0.0]
- Validation non-empty rate per step: [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]

## Gradient and frozen-backbone checks

- Frozen backbone no-grad check: True
- Trainable non-backbone gradient present check: True
- Trainable non-backbone finite gradient check: True
- Optimizer excludes frozen backbone: True

## Degeneracy checks

- Any degenerate validation point: True
- Persistent collapse from start: False
- All-background validation points: 2
- All-foreground validation points: 2

## Artifact outputs

- Summary JSON: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_bce01\hybrid_controlled_short_train_summary.json`
- Summary YAML: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_bce01\hybrid_controlled_short_train_summary.yaml`
- Report: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_bce01\hybrid_controlled_short_train_report.md`
- Train steps CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_bce01\train_steps.csv`
- Validation steps CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_bce01\validation_steps.csv`
- Gradient stats CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_bce01\gradient_stats.csv`
- Selection log CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_bce01\selection_log.csv`
- Progress plot: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_bce01\progress.png`
- Visual file count: 24

## Failure cases

None.

## Decision

- Status: `BLOCKED`

## Next recommended PR

- `Blocking fix PR`