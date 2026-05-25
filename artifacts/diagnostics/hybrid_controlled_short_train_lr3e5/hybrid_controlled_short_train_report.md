# Controlled Short Hybrid Training Report

## Repository findings

- Existing hybrid path reused: `src/models/hybrid.py::HybridFoundationUNet`.
- Existing backbone path reused: `src/models/backbone.py::FoundationXBackbone`.
- Existing PR-6 diagnostics were extended for longer bounded training and scheduled validation.
- The full single-split runner was intentionally not reused because it writes to `artifacts/runs/` and exceeds PR-7 scope.

## Setup

```
py scripts/controlled_train_hybrid_short.py --input_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\nnUNet_raw\Dataset101_Pneumothorax\imagesTr --labels_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\nnUNet_raw\Dataset101_Pneumothorax\labelsTr --foundation_checkpoint C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\checkpoints\foundation_x.pth --output_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr3e5 --img_size 512 --device auto --num_train_cases 32 --num_val_cases 16 --max_steps 25 --batch_size 1 --lr 3e-05 --bce_loss_weight 1.0 --val_every 5 --strict
```

- Device: `cpu`
- Train cases: 32 (pos=16, neg=16)
- Val cases: 16 (pos=8, neg=8)
- Max steps: 25
- Validation interval: 5
- Batch size: 1
- Learning rate: 3e-05
- BCE loss weight: 1.0
- Frozen backbone: True

## Dataset selection

- Train/val disjoint overlap count: 0
- Train case IDs count: 32
- Val case IDs count: 16

## Training results

- Steps completed: 25 / 25
- Train total loss series: [1.7671746015548706, 1.7790683507919312, 1.7347596883773804, 1.7766053676605225, 1.6901960372924805, 1.7729194164276123, 1.7604442834854126, 1.7709121704101562, 1.740135669708252, 1.7689517736434937, 1.765061378479004, 1.7670544385910034, 1.7490745782852173, 1.7653143405914307, 1.754357099533081, 1.766374111175537, 1.7579305171966553, 1.7597315311431885, 1.7471171617507935, 1.752949595451355, 1.6754186153411865, 1.7477467060089111, 1.7323980331420898, 1.7491226196289062, 1.722793698310852]
- Train predicted positive pixel ratio series: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9997673034667969, 0.8525848388671875]
- Train non-empty prediction rate series: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
- Probability output shape sample: `[1, 1, 512, 512]`
- Raw logits shape sample: `[1, 1, 512, 512]`
- NaN/Inf totals: prob_nan=0, prob_inf=0, logits_nan=0, logits_inf=0

## Validation results

- Validation schedule steps: [0, 5, 10, 15, 20, 25]
- Validation executed steps: [0, 5, 10, 15, 20, 25]
- Validation mean total loss per step: [1.7704522907733917, 1.7641577497124672, 1.758097104728222, 1.7523833215236664, 1.7446203082799911, 1.718184232711792]
- Validation Dice mean per step: [0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.010354844023822807]
- Validation probability min per step: [0.5203309655189514, 0.5183029174804688, 0.5139852166175842, 0.5098246335983276, 0.5045891404151917, 0.4852506220340729]
- Validation probability max per step: [0.5255031585693359, 0.5225749611854553, 0.5215298533439636, 0.5210412740707397, 0.5204997062683105, 0.5200168490409851]
- Validation probability mean per step: [0.5226851312996246, 0.5199861434332149, 0.5173672152361632, 0.5148812582540074, 0.5114783308839606, 0.4996742684556139]
- Validation probability std per step: [0.0004404916687739596, 0.0006426086297634859, 0.001588798551633823, 0.0023836230541564233, 0.0031784244565159045, 0.004980639763362085]
- Validation raw logit min per step: [0.08136864751577377, 0.07324434071779251, 0.05595557391643524, 0.039303552359342575, 0.0183570496737957, -0.05901468172669411]
- Validation raw logit max per step: [0.1021011620759964, 0.09036111831665039, 0.08617249876260757, 0.08421479910612106, 0.08204483240842819, 0.08011013269424438]
- Validation raw logit mean per step: [0.09080293469134659, 0.07998732562963085, 0.06949752781075702, 0.059543990550072756, 0.045923287867111195, -0.0013026991444357052]
- Validation raw logit std per step: [0.001765603372757835, 0.0025745942909971013, 0.006363105591611003, 0.009543411670137553, 0.012721190364627278, 0.0199246523112521]
- Validation probability foreground ratio @0.5 per step: [1.0, 1.0, 1.0, 1.0, 1.0, 0.43982815742492676]
- Validation non-empty rate per step: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

## Gradient and frozen-backbone checks

- Frozen backbone no-grad check: True
- Trainable non-backbone gradient present check: True
- Trainable non-backbone finite gradient check: True
- Optimizer excludes frozen backbone: True

## Degeneracy checks

- Any degenerate validation point: True
- Persistent collapse from start: False
- All-background validation points: 0
- All-foreground validation points: 5

## Artifact outputs

- Summary JSON: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr3e5\hybrid_controlled_short_train_summary.json`
- Summary YAML: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr3e5\hybrid_controlled_short_train_summary.yaml`
- Report: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr3e5\hybrid_controlled_short_train_report.md`
- Train steps CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr3e5\train_steps.csv`
- Validation steps CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr3e5\validation_steps.csv`
- Gradient stats CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr3e5\gradient_stats.csv`
- Selection log CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr3e5\selection_log.csv`
- Progress plot: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr3e5\progress.png`
- Visual file count: 24

## Failure cases

None.

## Decision

- Status: `BLOCKED`

## Next recommended PR

- `Blocking fix PR`