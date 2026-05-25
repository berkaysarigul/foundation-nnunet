# Controlled Short Hybrid Training Report

## Repository findings

- Existing hybrid path reused: `src/models/hybrid.py::HybridFoundationUNet`.
- Existing backbone path reused: `src/models/backbone.py::FoundationXBackbone`.
- Existing PR-6 diagnostics were extended for longer bounded training and scheduled validation.
- The full single-split runner was intentionally not reused because it writes to `artifacts/runs/` and exceeds PR-7 scope.

## Setup

```
py scripts/controlled_train_hybrid_short.py --input_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\nnUNet_raw\Dataset101_Pneumothorax\imagesTr --labels_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\nnUNet_raw\Dataset101_Pneumothorax\labelsTr --foundation_checkpoint C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\checkpoints\foundation_x.pth --output_dir C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr3e5_50step_bnfix --img_size 512 --device auto --num_train_cases 32 --num_val_cases 16 --max_steps 50 --batch_size 1 --lr 3e-05 --bce_loss_weight 1.0 --val_every 5 --strict
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
- Train total loss series: [1.7968134880065918, 1.7972044944763184, 1.736493468284607, 1.7832646369934082, 1.6766847372055054, 1.7449018955230713, 1.7381728887557983, 1.7364628314971924, 1.703096628189087, 1.7366663217544556, 1.7098493576049805, 1.7006404399871826, 1.7016860246658325, 1.6729774475097656, 1.6780459880828857, 1.7349436283111572, 1.6936535835266113, 1.6881521940231323, 1.6724308729171753, 1.6774497032165527, 1.5952351093292236, 1.668715000152588, 1.6067261695861816, 1.6818647384643555, 1.6454449892044067, 1.6245098114013672, 1.6399648189544678, 1.6216442584991455, 1.6322768926620483, 1.614050269126892, 1.5986204147338867, 1.6314663887023926, 1.6082828044891357, 1.6136025190353394, 1.561966061592102, 1.6327629089355469, 1.5311919450759888, 1.592422604560852, 1.597650170326233, 1.5856516361236572, 1.5687339305877686, 1.5991616249084473, 1.5732136964797974, 1.5783215761184692, 1.5811021327972412, 1.5545475482940674, 1.5592918395996094, 1.6290438175201416, 1.5781580209732056, 1.5772404670715332]
- Train predicted positive pixel ratio series: [0.7262802124023438, 0.7490768432617188, 0.6585578918457031, 0.664794921875, 0.5306015014648438, 0.54296875, 0.5380744934082031, 0.4175148010253906, 0.44469451904296875, 0.4308891296386719, 0.3475341796875, 0.2895317077636719, 0.35794830322265625, 0.19818878173828125, 0.24864959716796875, 0.36429595947265625, 0.3166923522949219, 0.281402587890625, 0.2741355895996094, 0.2460479736328125, 0.19807815551757812, 0.17156600952148438, 0.04808807373046875, 0.20557403564453125, 0.1728363037109375, 0.08018875122070312, 0.14052200317382812, 0.09135818481445312, 0.16585159301757812, 0.055438995361328125, 0.08671188354492188, 0.13331222534179688, 0.0920867919921875, 0.0745391845703125, 0.06427383422851562, 0.10091400146484375, 0.060153961181640625, 0.03693389892578125, 0.0878143310546875, 0.05048370361328125, 0.05141448974609375, 0.0700225830078125, 0.038021087646484375, 0.041179656982421875, 0.0861968994140625, 0.0206451416015625, 0.0301361083984375, 0.12702178955078125, 0.045440673828125, 0.0442657470703125]
- Train non-empty prediction rate series: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
- Probability output shape sample: `[1, 1, 512, 512]`
- Raw logits shape sample: `[1, 1, 512, 512]`
- NaN/Inf totals: prob_nan=0, prob_inf=0, logits_nan=0, logits_inf=0

## Validation results

- Validation schedule steps: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
- Validation executed steps: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
- Validation mean total loss per step: [1.7704522907733917, 1.7790506482124329, 1.7942934334278107, 1.7987403869628906, 1.7892336249351501, 1.7598677948117256, 1.716927394270897, 1.6777605935931206, 1.636041559278965, 1.6004372015595436, 1.570911891758442]
- Validation Dice mean per step: [0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009648090970586054, 0.009672927000792697, 0.009519385508610867, 0.013256685575470328, 0.01289151082164608, 0.007005794177530333, 0.002645544162078295]
- Validation probability min per step: [0.5203309655189514, 0.5214030742645264, 0.5211383700370789, 0.5139815807342529, 0.4941306412220001, 0.45540571212768555, 0.3905934691429138, 0.32892468571662903, 0.22945010662078857, 0.1710410863161087, 0.1474607139825821]
- Validation probability max per step: [0.5255031585693359, 0.5316306352615356, 0.5456856489181519, 0.5581854581832886, 0.5725918412208557, 0.5980157256126404, 0.6300091743469238, 0.6765350103378296, 0.7978029847145081, 0.9145212173461914, 0.9735480546951294]
- Validation probability mean per step: [0.5226851312996246, 0.5263455479125554, 0.5327442634519031, 0.5345651867938273, 0.5305412128399567, 0.5179012463224311, 0.4987776026511952, 0.48067010720473746, 0.4605424766192101, 0.4424965584233185, 0.42689449437091653]
- Validation probability std per step: [0.0004404916687739596, 0.0015682804035190626, 0.0031326019071238813, 0.005425948224250619, 0.008724855059252851, 0.013453268503191295, 0.017335032171971805, 0.018749292775355336, 0.02114886840180224, 0.02484983640920118, 0.028639824306735572]
- Validation raw logit min per step: [0.08136864751577377, 0.08566457033157349, 0.08460390567779541, 0.055940862745046616, -0.0234784297645092, -0.1788524091243744, -0.44481825828552246, -0.7130525708198547, -1.2114187479019165, -1.5782668590545654, -1.754657506942749]
- Validation raw logit max per step: [0.1021011620759964, 0.12669166922569275, 0.18325363099575043, 0.23380103707313538, 0.29243364930152893, 0.3972041606903076, 0.5322561860084534, 0.7378934025764465, 1.372619390487671, 2.3701331615448, 3.6056158542633057]
- Validation raw logit mean per step: [0.09080293469134659, 0.10548092924954489, 0.13117000122320555, 0.1384984986139557, 0.12235642539750202, 0.07169511374472615, -0.004893899970629348, -0.07746060501190982, -0.15841527587696233, -0.2315363514424915, -0.2953474320218695]
- Validation raw logit std per step: [0.001765603372757835, 0.006290945045575341, 0.012585897382242094, 0.021818598948880042, 0.035058741372535285, 0.05394266358265261, 0.0694070711545461, 0.07520131512368404, 0.0851684876497808, 0.10077593576214387, 0.11741370096515302]
- Validation probability foreground ratio @0.5 per step: [1.0, 1.0, 1.0, 1.0, 0.9999794960021973, 0.9520158767700195, 0.46436309814453125, 0.1604630947113037, 0.04412078857421875, 0.017889022827148438, 0.01350545883178711]
- Validation non-empty rate per step: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

## Validation threshold sweep

### threshold = 0.05

- dice_mean per step: [0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705]
- dice_pos_mean per step: [0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141]
- prediction_non_empty_rate per step: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
- pred_pos_pixel_ratio per step: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
- all_background per step: [False, False, False, False, False, False, False, False, False, False, False]
- all_foreground per step: [True, True, True, True, True, True, True, True, True, True, True]

### threshold = 0.10

- dice_mean per step: [0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705]
- dice_pos_mean per step: [0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141]
- prediction_non_empty_rate per step: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
- pred_pos_pixel_ratio per step: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
- all_background per step: [False, False, False, False, False, False, False, False, False, False, False]
- all_foreground per step: [True, True, True, True, True, True, True, True, True, True, True]

### threshold = 0.20

- dice_mean per step: [0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.00964787305565551, 0.00964787305565551]
- dice_pos_mean per step: [0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929574611131102, 0.01929574611131102]
- prediction_non_empty_rate per step: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
- pred_pos_pixel_ratio per step: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9999992847442627, 0.9999995231628418]
- all_background per step: [False, False, False, False, False, False, False, False, False, False, False]
- all_foreground per step: [True, True, True, True, True, True, True, True, True, True, True]

### threshold = 0.30

- dice_mean per step: [0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.00964787305565551, 0.009647889455663972, 0.009648657724028453]
- dice_pos_mean per step: [0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929574611131102, 0.019295778911327943, 0.019297315448056906]
- prediction_non_empty_rate per step: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
- pred_pos_pixel_ratio per step: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9999985694885254, 0.9999914169311523, 0.9999732971191406]
- all_background per step: [False, False, False, False, False, False, False, False, False, False, False]
- all_foreground per step: [True, True, True, True, True, True, True, True, True, True, True]

### threshold = 0.40

- dice_mean per step: [0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647903527365997, 0.009649398431065492, 0.009717914523207583, 0.011455957850557752]
- dice_pos_mean per step: [0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.019295807054731995, 0.019298796862130985, 0.019435829046415165, 0.022911915701115504]
- prediction_non_empty_rate per step: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
- pred_pos_pixel_ratio per step: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9999997615814209, 0.9999752044677734, 0.9998328685760498, 0.9879581928253174, 0.8334743976593018]
- all_background per step: [False, False, False, False, False, False, False, False, False, False, False]
- all_foreground per step: [True, True, True, True, True, True, True, True, True, True, False]

### threshold = 0.50

- dice_mean per step: [0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009647867700550705, 0.009648090970586054, 0.009672927000792697, 0.009519385508610867, 0.013256685575470328, 0.01289151082164608, 0.007005794177530333, 0.002645544162078295]
- dice_pos_mean per step: [0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.01929573540110141, 0.019296181941172108, 0.019345854001585394, 0.019038771017221734, 0.026513371150940657, 0.02578302164329216, 0.014011588355060667, 0.00529108832415659]
- prediction_non_empty_rate per step: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
- pred_pos_pixel_ratio per step: [1.0, 1.0, 1.0, 1.0, 0.9999794960021973, 0.9520158767700195, 0.46436309814453125, 0.1604630947113037, 0.04412078857421875, 0.017889022827148438, 0.01350545883178711]
- all_background per step: [False, False, False, False, False, False, False, False, False, False, False]
- all_foreground per step: [True, True, True, True, True, True, False, False, False, False, False]

## Train-mode preservation

- train_mode_restored_after_validation: True
- trainable_in_train_mode_before_step: True
- steps_checked / steps_passed: 50 / 50
- training_mode_violation_steps: []
- post_validation_checks: [{'after_val_step': 0, 'ok': True, 'model_training': True, 'message': ''}, {'after_val_step': 5, 'ok': True, 'model_training': True, 'message': ''}, {'after_val_step': 10, 'ok': True, 'model_training': True, 'message': ''}, {'after_val_step': 15, 'ok': True, 'model_training': True, 'message': ''}, {'after_val_step': 20, 'ok': True, 'model_training': True, 'message': ''}, {'after_val_step': 25, 'ok': True, 'model_training': True, 'message': ''}, {'after_val_step': 30, 'ok': True, 'model_training': True, 'message': ''}, {'after_val_step': 35, 'ok': True, 'model_training': True, 'message': ''}, {'after_val_step': 40, 'ok': True, 'model_training': True, 'message': ''}, {'after_val_step': 45, 'ok': True, 'model_training': True, 'message': ''}, {'after_val_step': 50, 'ok': True, 'model_training': True, 'message': ''}]

## Gradient and frozen-backbone checks

- Frozen backbone no-grad check: True
- Trainable non-backbone gradient present check: True
- Trainable non-backbone finite gradient check: True
- Optimizer excludes frozen backbone: True

## Degeneracy checks

- Any degenerate validation point: True
- Persistent collapse from start: False
- All-background validation points: 0
- All-foreground validation points: 6

## Artifact outputs

- Summary JSON: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr3e5_50step_bnfix\hybrid_controlled_short_train_summary.json`
- Summary YAML: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr3e5_50step_bnfix\hybrid_controlled_short_train_summary.yaml`
- Report: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr3e5_50step_bnfix\hybrid_controlled_short_train_report.md`
- Train steps CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr3e5_50step_bnfix\train_steps.csv`
- Validation steps CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr3e5_50step_bnfix\validation_steps.csv`
- Gradient stats CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr3e5_50step_bnfix\gradient_stats.csv`
- Selection log CSV: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr3e5_50step_bnfix\selection_log.csv`
- Progress plot: `C:\Users\beko5\Desktop\Foundation-nnU-Net\foundation-nnunet\artifacts\diagnostics\hybrid_controlled_short_train_lr3e5_50step_bnfix\progress.png`
- Visual file count: 24

## Failure cases

None.

## Warnings

- Degenerate validation points observed, but no persistent collapse; classified as PARTIAL_PASS.

## Decision

- Status: `PARTIAL_PASS`

## Next recommended PR

- `Blocking fix PR`