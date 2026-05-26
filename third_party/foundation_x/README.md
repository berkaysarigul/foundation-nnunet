Minimal vendored Foundation_X segmentation code
================================================

Source repository: https://github.com/JLiangLab/Foundation_X

Vendored commit:

```
5e0b473cf3f595320ce982697d4365e1d9a4f642
```

This folder intentionally contains only the upstream files needed to exercise
the official Swin + cyclic segmentation path for PR-9A-FIX Stage A/B:

- `models/dino/swin_transformer_CyclicSegmentation.py`
- `util/misc.py`
- `LICENSE`

The project wrapper in `src/models/foundation_x_official.py` imports the
vendored Swin file directly and loads only the `checkpoint[state_key]` keys
under `backbone.0.*`. It does not instantiate the full DINO detection model,
train, or export nnU-Net priors.
