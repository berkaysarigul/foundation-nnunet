# Foundation-nnU-Net: Claude Code Development Guide

> Legacy / non-authoritative guide.
>
> This file is preserved as historical design context only. It is **not** the
> current source of truth for methodology, data contracts, experiment outputs,
> or paper framing.
>
> Before using anything below, defer to these repository memory files:
> - `RECOVERY_TODO.md`
> - `AGENT_CONTEXT.md`
> - `DECISIONS.md`
> - `VALIDATION_CHECKLIST.md`
>
> Known stale assumptions in this guide include, but are not limited to:
> - old raw annotation references such as `stage_2_train.csv` and `mask_functions.py`
> - the legacy processed dataset layout under `data/processed/pneumothorax`
> - use of `results/` as if it were an authoritative output location
> - hybrid/Foundation X wording that no longer matches the recovered methodology
>
> Current authoritative conventions are:
> - trusted dataset root: `data/processed/pneumothorax_trusted_v1`
> - authoritative run outputs: `artifacts/runs/`
> - notebook outputs are non-authoritative unless fully traceable
> - Foundation X is not on the current main paper path


## PROJECT SUMMARY

**Project:** Foundation-nnU-Net — Hybrid deep learning architecture for multi-disease segmentation in chest X-rays.

**Core Idea:** Inject multi-scale feature maps from Foundation X's (WACV 2025) Swin-B backbone into a U-Net-based segmentation decoder to achieve better segmentation performance than a standalone U-Net.

**Target Diseases:** Pneumothorax (primary), then pneumonia and tuberculosis.

**Tech Stack:** Python, PyTorch, albumentations, pydicom, timm, numpy, pandas, matplotlib.

---

## PROJECT DIRECTORY STRUCTURE

```
foundation-nnunet/
├── data/
│   ├── raw/                        ← Original data (do not modify)
│   │   └── SIIM-ACR/
│   │       ├── stage_2_images/     ← 3205 DICOM files
│   │       ├── stage_2_train.csv   ← RLE mask annotations
│   │       └── mask_functions.py   ← Original RLE decode function
│   └── processed/
│       └── pneumothorax/
│           ├── images/             ← PNG images (512x512)
│           ├── masks/              ← PNG binary masks (512x512)
│           └── splits.json         ← Train/val/test split
│
├── src/
│   ├── data/
│   │   ├── preprocess.py           ← DICOM→PNG, RLE→mask conversion
│   │   ├── dataset.py              ← PyTorch Dataset class
│   │   └── augmentations.py        ← Data augmentation (albumentations)
│   │
│   ├── models/
│   │   ├── backbone.py             ← Foundation X Swin-B loading
│   │   ├── unet.py                 ← Baseline U-Net model
│   │   └── hybrid.py              ← Hybrid Foundation-nnU-Net
│   │
│   ├── training/
│   │   ├── trainer.py              ← Training loop
│   │   ├── losses.py               ← Dice + BCE loss
│   │   └── metrics.py              ← Dice, IoU, Hausdorff, Precision, Recall, F1
│   │
│   └── evaluation/
│       ├── evaluate.py             ← Test set final evaluation
│       └── visualize.py            ← Result visualization
│
├── configs/
│   └── config.yaml                 ← All hyperparameters
│
├── checkpoints/
│   └── foundation_x.pth           ← Foundation X pretrained weights (3GB, user has this)
│
├── results/                        ← Metric tables, charts
├── notebooks/                      ← Jupyter experiment notebooks
├── requirements.txt
└── README.md
```

---

## DEVELOPMENT PHASES (IN ORDER)

Code each phase sequentially. Ensure the current phase works before moving to the next.

---

### PHASE 1: Data Preparation

**File: `src/data/preprocess.py`**

This script runs from the command line. Arguments:
- `--raw_dir`: Raw data directory (default: `data/raw/SIIM-ACR`)
- `--output_dir`: Processed data directory (default: `data/processed/pneumothorax`)
- `--img_size`: Target image size (default: 512)
- `--seed`: Random seed (default: 42)

**Flow:**

1. Read `stage_2_train.csv` with pandas.
   - Columns: `ImageId`, `EncodedPixels`
   - If `EncodedPixels` is " -1" → no pneumothorax (empty mask)
   - Same `ImageId` can have multiple rows (multiple pneumothorax regions) → merge into same mask with OR.

2. For each unique `ImageId`:
   a. Read DICOM file: `pydicom.dcmread(path)` → get numpy array via `ds.pixel_array`
   b. Normalize pixel array to uint8:
      - DICOM pixel values can be 0-4095
      - `pixel_array = (pixel_array / pixel_array.max() * 255).astype(np.uint8)` to normalize to 0-255
   c. Resize to 512x512 as PIL Image (BILINEAR interpolation)
   d. Save as PNG: `{output_dir}/images/{ImageId}.png`

   e. RLE decode:
      - Parse EncodedPixels string
      - Use `rle2mask` function from `mask_functions.py` or rewrite:
        ```python
        def rle2mask(rle_string, height=1024, width=1024):
            """Converts RLE string to binary mask."""
            if rle_string == " -1" or rle_string == "-1":
                return np.zeros((height, width), dtype=np.uint8)
            s = rle_string.split()
            starts = np.array(s[0::2], dtype=int) - 1  # 1-indexed → 0-indexed
            lengths = np.array(s[1::2], dtype=int)
            mask = np.zeros(height * width, dtype=np.uint8)
            for start, length in zip(starts, lengths):
                mask[start:start+length] = 1
            mask = mask.reshape((height, width), order='F')  # Fortran order (column-major)
            return mask * 255  # 0/255 binary mask
        ```
      - For same ImageId with multiple RLEs, merge masks (OR): `final_mask = np.maximum(mask1, mask2)`
   f. Resize mask to 512x512 (PIL, NEAREST interpolation — critical, do NOT use BILINEAR)
   g. Save as PNG: `{output_dir}/masks/{ImageId}.png`

3. Create split:
   - Collect all ImageIds
   - Using `sklearn.model_selection.train_test_split`:
     - First split: 85% train_val, 15% test (random_state=seed)
     - Second split: train_val into 82.35% train, 17.65% val (yields overall 70% train, 15% val) (random_state=seed)
   - Save as JSON:
     ```json
     {
       "train": ["img_001", "img_002", ...],
       "val": ["img_100", "img_101", ...],
       "test": ["img_200", "img_201", ...]
     }
     ```

4. Print statistics:
   - Total image count
   - Pneumothorax positive / negative distribution
   - Train/val/test counts

**Dependencies:** pydicom, numpy, pandas, Pillow, scikit-learn

---

### PHASE 2: Dataset and Augmentations

**File: `src/data/dataset.py`**

PyTorch Dataset class. Feeds data during training and evaluation.

```python
class PneumothoraxDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split="train", img_size=256, transform=None):
        """
        Args:
            data_dir: processed data directory (data/processed/pneumothorax)
            split: "train", "val", or "test"
            img_size: runtime resize dimension (256 or 512)
            transform: albumentations augmentation pipeline
        """
        # Load image ID list for the given split from splits.json
        # self.image_ids = [...]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # 1. Read image (PNG, grayscale)
        #    image = cv2.imread(..., cv2.IMREAD_GRAYSCALE)  → (H, W)
        # 2. Read mask (PNG, grayscale)
        #    mask = cv2.imread(..., cv2.IMREAD_GRAYSCALE)   → (H, W)
        # 3. Resize to img_size (image: BILINEAR, mask: NEAREST)
        # 4. Apply augmentation (only for train split)
        #    if self.transform:
        #        augmented = self.transform(image=image, mask=mask)
        #        image, mask = augmented['image'], augmented['mask']
        # 5. Normalize: image = image / 255.0
        # 6. Binarize mask: mask = (mask > 127).float()
        # 7. Convert to tensor:
        #    image: (1, H, W) float32  → grayscale single channel
        #    mask:  (1, H, W) float32  → binary mask
        return image, mask
```

**File: `src/data/augmentations.py`**

Data augmentation using albumentations. Same transforms applied to both image and mask.

```python
import albumentations as A

def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.ElasticTransform(alpha=50, sigma=5, p=0.2),
        A.GaussNoise(var_limit=(5, 25), p=0.2),
    ])

def get_val_transforms():
    return None  # No augmentation for validation and test
```

**Notes:**
- Do NOT use VerticalFlip — inverted patients don't exist in X-rays.
- Use ElasticTransform lightly — excessive deformation distorts anatomical structures in medical images.
- albumentations automatically applies the same geometric transforms to both image and mask.

---

### PHASE 3: Foundation X Backbone

**File: `src/models/backbone.py`**

Loads Foundation X's Swin-B backbone and extracts multi-scale feature maps.

```python
class FoundationXBackbone(nn.Module):
    def __init__(self, checkpoint_path, frozen=True):
        """
        Args:
            checkpoint_path: Path to Foundation X .pth file
            frozen: If True, weights are not updated (default: True)
        """
        super().__init__()

        # 1. Create Swin-B using timm
        #    self.backbone = timm.create_model('swin_base_patch4_window7_224',
        #                                       pretrained=False,
        #                                       features_only=True,
        #                                       out_indices=(0, 1, 2, 3))
        #
        # 2. Load only backbone weights from checkpoint:
        #    checkpoint = torch.load(checkpoint_path, map_location='cpu')
        #
        #    Foundation X checkpoint keys likely start with:
        #    - "backbone...." or "model.backbone...." or "student...."
        #    - First inspect the key structure:
        #      print([k for k in checkpoint.keys()][:20])
        #
        #    Filter only backbone-related keys:
        #    backbone_weights = {}
        #    for k, v in checkpoint.items():
        #        if k.startswith("backbone."):
        #            new_key = k.replace("backbone.", "")
        #            backbone_weights[new_key] = v
        #
        #    self.backbone.load_state_dict(backbone_weights, strict=False)
        #
        # 3. Freeze:
        #    if frozen:
        #        for param in self.backbone.parameters():
        #            param.requires_grad = False
        #        self.backbone.eval()

    def forward(self, x):
        """
        Args:
            x: (batch, 1, H, W) grayscale image

        Returns:
            list of 4 feature maps:
              f1: (batch, 128,  H/4,  H/4)   → Stage 1
              f2: (batch, 256,  H/8,  H/8)   → Stage 2
              f3: (batch, 512,  H/16, H/16)  → Stage 3
              f4: (batch, 1024, H/32, H/32)  → Stage 4
        """
        # Grayscale → 3 channels (Swin-B expects RGB)
        x = x.repeat(1, 3, 1, 1)  # (batch, 1, H, W) → (batch, 3, H, W)

        # timm features_only=True returns 4 stage outputs
        features = self.backbone(x)
        return features  # [f1, f2, f3, f4]
```

**IMPORTANT NOTES:**
- Inspect the Foundation X checkpoint key structure first with `print`. Key prefix could be "backbone.", "model.", "student." or direct.
- Use `load_state_dict(..., strict=False)` — unmatched keys (classification head etc.) will be skipped.
- `features_only=True` in timm removes the classification head and uses the model purely as a feature extractor.
- Verify the exact Swin-B variant used by Foundation X by checking their repo config files.
- If timm compatibility issues arise, use Foundation X repo's `load_weights.py`:
  ```python
  from load_weights import build_model
  foundationx_model = build_model(checkpoint_path, num_classes=0)
  ```

---

### PHASE 4: Baseline U-Net

**File: `src/models/unet.py`**

Plain U-Net model — without Foundation X. Serves as the comparison (baseline) model.

```python
class ConvBlock(nn.Module):
    """Two consecutive Conv → BatchNorm → ReLU blocks."""
    def __init__(self, in_ch, out_ch):
        # Conv2d(in_ch, out_ch, 3, padding=1) → BatchNorm2d → ReLU
        # Conv2d(out_ch, out_ch, 3, padding=1) → BatchNorm2d → ReLU

class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, base_filters=64):
        """
        Args:
            in_channels: 1 (grayscale)
            num_classes: 1 (binary segmentation)
            base_filters: 64 (starting channel count for encoder)
        """
        super().__init__()

        # Encoder (contracting path)
        # enc1: 1  → 64   (input size preserved)
        # enc2: 64 → 128  (halved via MaxPool2d)
        # enc3: 128 → 256
        # enc4: 256 → 512

        # Bottleneck
        # bottleneck: 512 → 1024

        # Decoder (expanding path)
        # ConvTranspose2d for upsampling
        # Skip connection via concat from encoder
        # dec4: 1024+512 → 512
        # dec3: 512+256 → 256
        # dec2: 256+128 → 128
        # dec1: 128+64  → 64

        # Final layer
        # self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        # Sigmoid → output

    def forward(self, x):
        """
        Args:
            x: (batch, 1, H, W)
        Returns:
            (batch, 1, H, W) — sigmoid output, each pixel 0-1
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder + skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.final(d1))
```

**Notes:**
- `pool`: `nn.MaxPool2d(2)`
- `up`: `nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)`
- Skip connection = concatenating encoder output to decoder. Preserves detail information.
- Use `sigmoid` at output — each pixel represents disease probability (0-1).

---

### PHASE 5: Hybrid Foundation-nnU-Net

**File: `src/models/hybrid.py`**

The project's main model. Injects Foundation X features into U-Net encoder.

```python
class FusionBlock(nn.Module):
    """Fuses Foundation X and U-Net encoder features."""
    def __init__(self, fx_channels, unet_channels, out_channels):
        super().__init__()
        # 1x1 Conv: reduces channel count after concat
        # (fx_channels + unet_channels) → out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(fx_channels + unet_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, fx_feat, unet_feat):
        """
        fx_feat and unet_feat must have the same spatial dimensions.
        If different, interpolate fx_feat to match unet_feat size.
        """
        if fx_feat.shape[2:] != unet_feat.shape[2:]:
            fx_feat = F.interpolate(fx_feat, size=unet_feat.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([fx_feat, unet_feat], dim=1))


class HybridFoundationUNet(nn.Module):
    def __init__(self, backbone_checkpoint, in_channels=1, num_classes=1,
                 base_filters=64, frozen_backbone=True):
        super().__init__()

        # Foundation X backbone (frozen)
        self.foundation_x = FoundationXBackbone(backbone_checkpoint, frozen=frozen_backbone)

        # U-Net encoder (trainable)
        self.enc1 = ConvBlock(in_channels, base_filters)        # → 64ch
        self.enc2 = ConvBlock(base_filters, base_filters*2)     # → 128ch
        self.enc3 = ConvBlock(base_filters*2, base_filters*4)   # → 256ch
        self.enc4 = ConvBlock(base_filters*4, base_filters*8)   # → 512ch
        self.pool = nn.MaxPool2d(2)

        # Fusion blocks (Foundation X + U-Net encoder → fused)
        # Foundation X Swin-B channel counts: 128, 256, 512, 1024
        self.fusion1 = FusionBlock(128,  64,   64)    # 128+64  → 64
        self.fusion2 = FusionBlock(256,  128,  128)   # 256+128 → 128
        self.fusion3 = FusionBlock(512,  256,  256)   # 512+256 → 256
        self.fusion4 = FusionBlock(1024, 512,  512)   # 1024+512 → 512

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder (trainable)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = ConvBlock(1024, 512)  # 512 (up) + 512 (fused skip) = 1024
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (batch, 1, H, W) grayscale X-ray
        Returns:
            (batch, 1, H, W) sigmoid mask output
        """
        # Foundation X feature extraction (frozen, no gradients)
        with torch.no_grad():
            fx_features = self.foundation_x(x)  # [f1, f2, f3, f4]

        # U-Net encoder
        e1 = self.enc1(x)              # (batch, 64, H, W)
        e2 = self.enc2(self.pool(e1))  # (batch, 128, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # (batch, 256, H/4, W/4)
        e4 = self.enc4(self.pool(e3))  # (batch, 512, H/8, W/8)

        # Fusion: Foundation X + U-Net encoder
        fused1 = self.fusion1(fx_features[0], e1)  # → 64ch
        fused2 = self.fusion2(fx_features[1], e2)  # → 128ch
        fused3 = self.fusion3(fx_features[2], e3)  # → 256ch
        fused4 = self.fusion4(fx_features[3], e4)  # → 512ch

        # Bottleneck
        b = self.bottleneck(self.pool(fused4))

        # Decoder + fused skip connections
        d4 = self.dec4(torch.cat([self.up4(b), fused4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), fused3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), fused2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), fused1], dim=1))

        return torch.sigmoid(self.final(d1))
```

**IMPORTANT NOTES:**
- Foundation X feature map sizes depend on input size. 256x256 input → stage outputs are 64, 32, 16, 8. 512x512 → 128, 64, 32, 16.
- U-Net encoder spatial dimensions at each level may not match Foundation X's corresponding stage. `FusionBlock`'s `F.interpolate` handles this.
- Foundation X runs inside `torch.no_grad()` — saves memory and speed.
- Input size must be a multiple of 32 due to Swin-B patch embedding (256 and 512 are both valid).

---

### PHASE 6: Loss and Metrics

**File: `src/training/losses.py`**

```python
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        pred:   (batch, 1, H, W) — sigmoid output (0-1)
        target: (batch, 1, H, W) — binary mask (0 or 1)
        """
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.dice = DiceLoss(smooth)
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        return self.dice(pred, target) + self.bce(pred, target)
```

**File: `src/training/metrics.py`**

```python
def dice_score(pred, target, threshold=0.5, smooth=1e-6):
    """
    pred:   (batch, 1, H, W) sigmoid output
    target: (batch, 1, H, W) binary mask
    Returns: float (0-1)
    """
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    return (2. * intersection + smooth) / (pred_binary.sum() + target.sum() + smooth)


def iou_score(pred, target, threshold=0.5, smooth=1e-6):
    """Intersection over Union."""
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def hausdorff_distance(pred, target, threshold=0.5):
    """
    Hausdorff distance between two mask boundaries.
    Use scipy.ndimage or medpy.
    Return NaN for empty masks.
    """
    from scipy.ndimage import distance_transform_edt
    # Convert pred and target to numpy, binarize
    # Find boundary pixels
    # Compute max distance from each boundary pixel to the other boundary
    pass


def precision_score(pred, target, threshold=0.5, smooth=1e-6):
    pred_binary = (pred > threshold).float()
    tp = (pred_binary * target).sum()
    return (tp + smooth) / (pred_binary.sum() + smooth)


def recall_score(pred, target, threshold=0.5, smooth=1e-6):
    pred_binary = (pred > threshold).float()
    tp = (pred_binary * target).sum()
    return (tp + smooth) / (target.sum() + smooth)


def f1_score(pred, target, threshold=0.5):
    p = precision_score(pred, target, threshold)
    r = recall_score(pred, target, threshold)
    return 2 * p * r / (p + r + 1e-6)
```

---

### PHASE 7: Training Loop

**File: `src/training/trainer.py`**

Main training script. Runs from command line:
```bash
python -m src.training.trainer --config configs/config.yaml
```

**Flow:**

```python
def train(config):
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Dataset and DataLoader
    train_dataset = PneumothoraxDataset(config.data_dir, split="train",
                                         img_size=config.input_size,
                                         transform=get_train_transforms())
    val_dataset = PneumothoraxDataset(config.data_dir, split="val",
                                       img_size=config.input_size,
                                       transform=None)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers, pin_memory=True)

    # 3. Model
    if config.model_type == "baseline":
        model = UNet(in_channels=1, num_classes=1).to(device)
    elif config.model_type == "hybrid":
        model = HybridFoundationUNet(
            backbone_checkpoint=config.foundation_x_checkpoint,
            frozen_backbone=config.frozen_backbone
        ).to(device)

    # 4. Loss, Optimizer, Scheduler
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # 5. Training loop
    best_dice = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_iou': []}

    for epoch in range(config.epochs):
        # === TRAIN ===
        model.train()
        # CRITICAL: Keep Foundation X backbone in eval mode
        if hasattr(model, 'foundation_x'):
            model.foundation_x.backbone.eval()

        train_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # === VALIDATION ===
        model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                val_loss += criterion(preds, masks).item()
                val_dice += dice_score(preds, masks).item()
                val_iou += iou_score(preds, masks).item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)

        scheduler.step()

        # === LOG ===
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)

        print(f"Epoch {epoch+1}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Dice: {val_dice:.4f} | "
              f"Val IoU: {val_iou:.4f}")

        # === CHECKPOINT ===
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), f"checkpoints/best_{config.model_type}.pth")
            print(f"  → Best model saved! Dice: {best_dice:.4f}")
        else:
            patience_counter += 1

        # === EARLY STOPPING ===
        if patience_counter >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # 6. Save history
    pd.DataFrame(history).to_csv(f"results/{config.model_type}_history.csv", index=False)

    return best_dice
```

---

### PHASE 8: Evaluation and Visualization

**File: `src/evaluation/evaluate.py`**

```python
def evaluate(config):
    # 1. Load best model
    # 2. Create test DataLoader
    # 3. For each batch:
    #    - Forward pass (torch.no_grad)
    #    - Compute Dice, IoU, Hausdorff, Precision, Recall, F1 per image
    # 4. Report mean and std
    # 5. Report positive (has pneumothorax) and negative samples separately
    # 6. Save as results/test_metrics_{model_type}.csv
    # 7. Print summary table to terminal
```

**File: `src/evaluation/visualize.py`**

```python
def plot_training_curves(history_csv_path, save_dir="results/"):
    # 1. Read CSV
    # 2. matplotlib subplots:
    #    Left: Epoch vs Train/Val Loss
    #    Right: Epoch vs Val Dice
    # 3. Save as PNG

def plot_predictions(model, dataset, num_samples=8, save_dir="results/"):
    # 1. Select best 4 + worst 4 predictions (by Dice)
    # 2. 3 columns per sample: original | ground truth mask | predicted mask
    # 3. Show Dice score in title
    # 4. Save as PNG

def plot_comparison(baseline_csv, hybrid_csv, save_dir="results/"):
    # 1. Compare Dice distributions of both models via box plot
    # 2. Display metric differences as a table
    # 3. Save as PNG
```

---

### PHASE 9: Config File

**File: `configs/config.yaml`**

```yaml
# ========================
# Foundation-nnU-Net Config
# ========================

# --- Model ---
model:
  type: "baseline"              # "baseline" or "hybrid"
  in_channels: 1
  num_classes: 1
  base_filters: 64

# --- Foundation X ---
foundation_x:
  checkpoint_path: "checkpoints/foundation_x.pth"
  frozen: true

# --- Data ---
data:
  processed_dir: "data/processed/pneumothorax"
  input_size: 256               # 256 or 512
  num_workers: 4

# --- Training ---
training:
  epochs: 150
  batch_size: 16                # 256: bs=16, 512: bs=8 or 4
  learning_rate: 0.001
  optimizer: "AdamW"
  weight_decay: 0.0001
  scheduler: "CosineAnnealing"
  early_stopping_patience: 20

# --- Loss ---
loss:
  type: "dice_bce"

# --- Other ---
seed: 42
device: "auto"                  # "auto", "cuda", "cpu"
```

---

### PHASE 10: Requirements

**File: `requirements.txt`**

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
pydicom>=2.4.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0
scikit-learn>=1.3.0
albumentations>=1.3.0
opencv-python-headless>=4.8.0
matplotlib>=3.7.0
pyyaml>=6.0
scipy>=1.11.0
tqdm>=4.65.0
```

---

## CODING ORDER

Follow this order in Claude Code:

1. Create `requirements.txt` and install dependencies
2. Create directory structure
3. Create `configs/config.yaml`
4. Code `src/data/preprocess.py` and test it
5. Code `src/data/augmentations.py`
6. Code `src/data/dataset.py` and test it (load a few samples, check shapes)
7. Code `src/training/losses.py`
8. Code `src/training/metrics.py`
9. Code `src/models/unet.py` and test it (forward pass with random input)
10. Code `src/models/backbone.py` and test it (load checkpoint, check feature shapes)
11. Code `src/models/hybrid.py` and test it (forward pass with random input)
12. Code `src/training/trainer.py`
13. **Run baseline training** (model.type: "baseline")
14. **Run hybrid training** (model.type: "hybrid")
15. Code `src/evaluation/evaluate.py` and run for both models
16. Code `src/evaluation/visualize.py` and generate visuals
17. Compare results

---

## CRITICAL WARNINGS

1. **Mask resize must always use NEAREST interpolation.** BILINEAR blurs masks and breaks binary values.

2. **Check Foundation X checkpoint key structure first.** Use `print(list(checkpoint.keys())[:30])` to identify the key prefix.

3. **Empty mask metric handling:** Dice is undefined for pneumothorax-negative images (fully black masks). Separate these during evaluation or handle with smooth factor.

4. **GPU memory:** 256x256 + bs=16 ≈ 8-10GB VRAM. 512x512 + bs=4 ≈ 12-16GB. A100 (40GB) handles both scenarios comfortably.

5. **Seed must be consistent everywhere:** numpy, torch, random all initialized with seed=42 (reproducibility).

6. **Foundation X backbone must stay in eval mode:** Even when `model.train()` is called, backbone must remain in `eval()` mode (to prevent BatchNorm stats corruption). Handle this in the trainer.

---

## FUTURE PHASES (AFTER MAIN PIPELINE IS COMPLETE)

### Multi-Disease Expansion
- Download ChestX-Det dataset (Kaggle: mathurinache/chestxdetdataset)
- Convert JSON polygon annotations → masks
- Update Dataset class to support multi-class (num_classes > 1)
- Update loss to multi-class Dice + CE

### Weak Supervision (SAM Pseudo-Masks)
- Get bounding boxes from TBX11K, NODE21, RSNA Pneumonia datasets
- Generate pseudo-masks using SAM (Segment Anything Model)
- Include as auxiliary loss with low weight (e.g., 0.3) during training

### Ablation Study
- Frozen vs fine-tuned backbone comparison
- Without Foundation X (baseline) vs with Foundation X (hybrid)
- 256x256 vs 512x512 comparison
- Augmentation effect (with vs without augmentation)
- Strong labels only vs strong + weak labels

---

## EXPECTED OUTPUTS

| Metric | Baseline (U-Net) | Hybrid (Foundation-nnU-Net) |
|--------|------|--------|
| Dice | ~0.70-0.80 | ~0.80-0.88 (target) |
| IoU | ~0.55-0.70 | ~0.68-0.80 (target) |

These values are estimates. Actual results will be determined after training.
