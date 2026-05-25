# Controlled Short Hybrid Training Report

## Repository findings

- Existing hybrid path reused: `src/models/hybrid.py::HybridFoundationUNet`.
- Existing backbone path reused: `src/models/backbone.py::FoundationXBackbone`.
- Existing PR-6 diagnostics were extended for longer bounded training and scheduled validation.
- The full single-split runner was intentionally not reused because it writes to `artifacts/runs/` and exceeds PR-7 scope.

## Setup

```
py scripts/controlled_train_hybrid_short.py --input_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\nnUNet_raw\Dataset101_Pneumothorax\imagesTr --labels_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\nnUNet_raw\Dataset101_Pneumothorax\labelsTr --foundation_checkpoint C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\checkpoints\foundation_x.pth --output_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr1e5 --img_size 512 --device auto --num_train_cases 32 --num_val_cases 16 --max_steps 25 --batch_size 1 --lr 1e-05 --bce_loss_weight 1.0 --val_every 5 --strict
```

- Device: `cpu`
- Train cases: 32 (pos=16, neg=16)
- Val cases: 16 (pos=8, neg=8)
- Max steps: 25
- Validation interval: 5
- Batch size: 1
- Learning rate: 1e-05
- BCE loss weight: 1.0
- Frozen backbone: True

## Dataset selection

- Train/val disjoint overlap count: 0
- Train case IDs count: 32
- Val case IDs count: 16

## Training results

- Steps completed: 25 / 25
- Train total loss series: [1.7671746015548706, 1.780179500579834, 1.7361736297607422, 1.7783925533294678, 1.6933913230895996, 1.7779566049575806, 1.764997959136963, 1.7770204544067383, 1.747248649597168, 1.7762113809585571, 1.77365243434906, 1.7754192352294922, 1.7583553791046143, 1.7746877670288086, 1.7647788524627686, 1.7748076915740967, 1.770380973815918, 1.7728588581085205, 1.760040283203125, 1.7708594799041748, 1.6912951469421387, 1.7697051763534546, 1.754838466644287, 1.771791696548462, 1.7548000812530518]
- Train predicted positive pixel ratio series: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
- Train non-empty prediction rate series: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
- Probability output shape sample: `[1, 1, 512, 512]`
- Raw logits shape sample: `[1, 1, 512, 512]`
- NaN/Inf totals: prob_nan=0, prob_inf=0, logits_nan=0, logits_inf=0

## Validation results

- Validation schedule steps: [0, 5, 10, 15, 20, 25]
- Validation executed steps: [0, 5, 10, 15, 20, 25]
- Validation mean total loss per step: [1.7704522907733917, 1.768368624150753, 1.7662285938858986, 1.764055535197258, 1.7619008049368858, 1.7597222030162811]
- Validation Dice mean per step: [0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705]
- Validation probability min per step: [0.5203309655189514, 0.519738495349884, 0.5191394686698914, 0.5182223320007324, 0.5167309045791626, 0.51524817943573]
- Validation probability max per step: [0.5255031585693359, 0.5243183374404907, 0.5233866572380066, 0.5225931406021118, 0.5219947695732117, 0.5216546654701233]
- Validation probability mean per step: [0.5226851312996246, 0.5217941001667867, 0.5208764871298825, 0.519942037357751, 0.5190129081562702, 0.5180709657110896]
- Validation probability std per step: [0.0004404916687739596, 0.0001464352869277356, 0.0003267055029744905, 0.0006666104972296329, 0.00099425513462411, 0.0013306573615920244]
- Validation raw logit min per step: [0.08136864751577377, 0.07899486273527145, 0.07659528404474258, 0.07292158156633377, 0.06694860756397247, 0.06101158261299133]
- Validation raw logit max per step: [0.1021011620759964, 0.09735022485256195, 0.09361498802900314, 0.0904342383146286, 0.08803591877222061, 0.0866728201508522]
- Validation raw logit mean per step: [0.09080293469134659, 0.08723168099896661, 0.08355456077102552, 0.07981062941955663, 0.07608862396876681, 0.07231587850945242]
- Validation raw logit std per step: [0.001765603372757835, 0.0005868698579548078, 0.0013091251474523515, 0.0026707435266831578, 0.003982894544021128, 0.005329790639027734]
- Validation probability foreground ratio @0.5 per step: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
- Validation non-empty rate per step: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

## Gradient and frozen-backbone checks

- Frozen backbone no-grad check: True
- Trainable non-backbone gradient present check: True
- Trainable non-backbone finite gradient check: True
- Optimizer excludes frozen backbone: True

## Degeneracy checks

- Any degenerate validation point: True
- Persistent collapse from start: True
- All-background validation points: 0
- All-foreground validation points: 6

## Artifact outputs

- Summary JSON: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr1e5\hybrid_controlled_short_train_summary.json`
- Summary YAML: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr1e5\hybrid_controlled_short_train_summary.yaml`
- Report: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr1e5\hybrid_controlled_short_train_report.md`
- Train steps CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr1e5\train_steps.csv`
- Validation steps CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr1e5\validation_steps.csv`
- Gradient stats CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr1e5\gradient_stats.csv`
- Selection log CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr1e5\selection_log.csv`
- Progress plot: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr1e5\progress.png`
- Visual file count: 24

## Failure cases

None.

## Decision

- Status: `BLOCKED`

## Next recommended PR

- `Blocking fix PR`