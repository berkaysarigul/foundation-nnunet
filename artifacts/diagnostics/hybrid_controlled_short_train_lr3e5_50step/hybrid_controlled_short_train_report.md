# Controlled Short Hybrid Training Report

## Repository findings

- Existing hybrid path reused: `src/models/hybrid.py::HybridFoundationUNet`.
- Existing backbone path reused: `src/models/backbone.py::FoundationXBackbone`.
- Existing PR-6 diagnostics were extended for longer bounded training and scheduled validation.
- The full single-split runner was intentionally not reused because it writes to `artifacts/runs/` and exceeds PR-7 scope.

## Setup

```
py scripts/controlled_train_hybrid_short.py --input_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\nnUNet_raw\Dataset101_Pneumothorax\imagesTr --labels_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\nnUNet_raw\Dataset101_Pneumothorax\labelsTr --foundation_checkpoint C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\checkpoints\foundation_x.pth --output_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr3e5_50step --img_size 512 --device auto --num_train_cases 32 --num_val_cases 16 --max_steps 50 --batch_size 1 --lr 3e-05 --bce_loss_weight 1.0 --val_every 5 --strict
```

- Device: `cpu`
- Train cases: 32 (pos=16, neg=16)
- Val cases: 16 (pos=8, neg=8)
- Max steps: 50
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

- Steps completed: 50 / 50
- Train total loss series: [1.7671746015548706, 1.7790683507919312, 1.7347596883773804, 1.7766053676605225, 1.6901960372924805, 1.7729194164276123, 1.7604442834854126, 1.7709121704101562, 1.740135669708252, 1.7689517736434937, 1.765061378479004, 1.7670544385910034, 1.7490745782852173, 1.7653143405914307, 1.754357099533081, 1.766374111175537, 1.7579305171966553, 1.7597315311431885, 1.7471171617507935, 1.752949595451355, 1.6754186153411865, 1.7477467060089111, 1.7323980331420898, 1.7491226196289062, 1.722793698310852, 1.7232253551483154, 1.6972980499267578, 1.6612144708633423, 1.6111369132995605, 1.5011358261108398, 1.385802984237671, 1.169735312461853, 1.0993965864181519, 1.016658902168274, 1.6821407079696655, 1.0052350759506226, 1.7643566131591797, 1.006171464920044, 1.0708892345428467, 1.0129209756851196, 1.1089060306549072, 1.0513577461242676, 1.0392985343933105, 1.041106939315796, 1.1120457649230957, 1.0335347652435303, 1.070286750793457, 1.0310442447662354, 1.039624571800232, 1.0207467079162598]
- Train predicted positive pixel ratio series: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9997673034667969, 0.8525848388671875, 0.3470611572265625, 0.059200286865234375, 0.0182037353515625, 0.020923614501953125, 0.005859375, 0.001148223876953125, 0.0003814697265625, 7.2479248046875e-05, 2.288818359375e-05, 1.1444091796875e-05, 1.1444091796875e-05, 7.62939453125e-06, 7.62939453125e-06, 1.52587890625e-05, 1.9073486328125e-05, 2.6702880859375e-05, 3.814697265625e-05, 4.1961669921875e-05, 5.340576171875e-05, 8.0108642578125e-05, 4.57763671875e-05, 5.7220458984375e-05, 4.57763671875e-05, 2.288818359375e-05, 2.288818359375e-05]
- Train non-empty prediction rate series: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
- Probability output shape sample: `[1, 1, 512, 512]`
- Raw logits shape sample: `[1, 1, 512, 512]`
- NaN/Inf totals: prob_nan=0, prob_inf=0, logits_nan=0, logits_inf=0

## Validation results

- Validation schedule steps: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
- Validation executed steps: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
- Validation mean total loss per step: [1.7704522907733917, 1.7641577497124672, 1.758097104728222, 1.7523833215236664, 1.7446203082799911, 1.718184232711792, 1.3651202917099, 1.096989706158638, 1.0477695614099503, 1.0569886714220047, 1.0354984402656555]
- Validation Dice mean per step: [0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.010354844023822807, 0.0, 0.0, 0.0, 0.0, 0.0]
- Validation probability min per step: [0.5203309655189514, 0.5183029174804688, 0.5139852166175842, 0.5098246335983276, 0.5045891404151917, 0.4852506220340729, 0.15261109173297882, 6.7185380522860605e-12, 3.0825221983832307e-06, 0.00019405341299716383, 1.8972843918163562e-06]
- Validation probability max per step: [0.5255031585693359, 0.5225749611854553, 0.5215298533439636, 0.5210412740707397, 0.5204997062683105, 0.5200168490409851, 0.5183802843093872, 0.5053296685218811, 0.5114314556121826, 0.5134673714637756, 0.5104433298110962]
- Validation probability mean per step: [0.5226851312996246, 0.5199861434332149, 0.5173672152361632, 0.5148812582540074, 0.5114783308839606, 0.4996742684556139, 0.30059089921960336, 0.005366380721464369, 0.01986145059003429, 0.04771797467305365, 0.023094248992567512]
- Validation probability std per step: [0.0004404916687739596, 0.0006426086297634859, 0.001588798551633823, 0.0023836230541564233, 0.0031784244565159045, 0.004980639763362085, 0.05934132687315496, 0.031151959362008926, 0.05566365534415773, 0.07536721174930566, 0.05733748487480664]
- Validation raw logit min per step: [0.08136864751577377, 0.07324434071779251, 0.05595557391643524, 0.039303552359342575, 0.0183570496737957, -0.05901468172669411, -1.7142670154571533, -25.726150512695312, -12.689759254455566, -8.5471830368042, -13.175085067749023]
- Validation raw logit max per step: [0.1021011620759964, 0.09036111831665039, 0.08617249876260757, 0.08421479910612106, 0.08204483240842819, 0.08011013269424438, 0.07355435192584991, 0.02131955511868, 0.04573366418480873, 0.05388259515166283, 0.041779279708862305]
- Validation raw logit mean per step: [0.09080293469134659, 0.07998732562963085, 0.06949752781075702, 0.059543990550072756, 0.045923287867111195, -0.0013026991444357052, -0.8585803412799888, -13.531019606906058, -6.219257479814827, -3.9843015856072617, -6.102986044333198]
- Validation raw logit std per step: [0.001765603372757835, 0.0025745942909971013, 0.006363105591611003, 0.009543411670137553, 0.012721190364627278, 0.0199246523112521, 0.2751565490271232, 4.775492783632329, 2.30006450869993, 1.536062180099273, 2.4699688455274127]
- Validation probability foreground ratio @0.5 per step: [1.0, 1.0, 1.0, 1.0, 1.0, 0.43982815742492676, 0.0015201568603515625, 6.198883056640625e-06, 2.0503997802734375e-05, 4.410743713378906e-05, 1.6450881958007812e-05]
- Validation non-empty rate per step: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9375, 1.0, 1.0, 1.0]

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

- Summary JSON: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr3e5_50step\hybrid_controlled_short_train_summary.json`
- Summary YAML: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr3e5_50step\hybrid_controlled_short_train_summary.yaml`
- Report: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr3e5_50step\hybrid_controlled_short_train_report.md`
- Train steps CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr3e5_50step\train_steps.csv`
- Validation steps CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr3e5_50step\validation_steps.csv`
- Gradient stats CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr3e5_50step\gradient_stats.csv`
- Selection log CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr3e5_50step\selection_log.csv`
- Progress plot: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr3e5_50step\progress.png`
- Visual file count: 24

## Failure cases

None.

## Warnings

- Degenerate validation points observed, but no persistent collapse; classified as PARTIAL_PASS.

## Decision

- Status: `PARTIAL_PASS`

## Next recommended PR

- `Blocking fix PR`