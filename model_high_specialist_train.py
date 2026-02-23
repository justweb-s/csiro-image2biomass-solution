# ============================================================
# 0. SETUP
# ============================================================
import os, json, random, gc
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = SCRIPT_DIR / "SETTINGS.json"

def _resolve_path(p: Optional[str]) -> Path:
    if p is None:
        raise ValueError("Missing path in SETTINGS.json")
    candidate = Path(p).expanduser()
    if candidate.is_absolute():
        return candidate
    return (SCRIPT_DIR / candidate).resolve()

def _load_settings(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing SETTINGS.json at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

SETTINGS = _load_settings(SETTINGS_PATH)
RAW_DATA_DIR = _resolve_path(SETTINGS.get("RAW_DATA_DIR"))
PROCESSED_DATA_DIR = _resolve_path(SETTINGS.get("PROCESSED_DATA_DIR", "./data/processed"))
ARTIFACTS_DIR = _resolve_path(SETTINGS.get("ARTIFACTS_DIR", "./artifacts"))
MODEL_DIR = _resolve_path(SETTINGS.get("MODEL_DIR"))
BACKBONE_DIR = _resolve_path(SETTINGS.get("BACKBONE_DIR", "./backbone/dinov3-vit7b-backbone"))
BACKBONE_NAME_OR_PATH = SETTINGS.get("BACKBONE_NAME_OR_PATH", "facebook/dinov3-vit7b16-pretrain-lvd1689m")

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. IMPORT E CONFIGURAZIONI BASE
# ============================================================

import os
import random
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import cv2

import torch
from torch import nn

from sklearn.model_selection import GroupKFold

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm.auto import tqdm
from transformers import AutoModel, AutoConfig
from torch.utils.data import WeightedRandomSampler, DataLoader, Dataset

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

SMOKE_TEST = str(os.environ.get("SMOKE_TEST", "0")).lower() in ("1", "true", "yes", "y")
SMOKE_IMG_SIZE = int(os.environ.get("SMOKE_IMG_SIZE", "224"))
SMOKE_MAX_SAMPLES = int(os.environ.get("SMOKE_MAX_SAMPLES", "64"))
if SMOKE_TEST:
    print(f"SMOKE_TEST enabled (IMG_SIZE={SMOKE_IMG_SIZE}, MAX_SAMPLES={SMOKE_MAX_SAMPLES})")

def seed_everything(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(123)

# ============================================================
# 2. DATA LOADING & TRAIN_WIDE PREPARATION + FOLDS
# ============================================================

# The competition dataset must be available under RAW_DATA_DIR (see SETTINGS.json)
# and must contain train.csv, test.csv, and the train/ + test/ image folders.
DATA_DIR = RAW_DATA_DIR

train_csv_path = DATA_DIR / "train.csv"
test_csv_path  = DATA_DIR / "test.csv"

train_long = pd.read_csv(train_csv_path)
test_long  = pd.read_csv(test_csv_path)

print("train_long shape:", train_long.shape)
print("test_long  shape:", test_long.shape)
print(train_long.head())

ID_COLS = [
    "image_path",
    "Sampling_Date",
    "State",
    "Species",
    "Pre_GSHH_NDVI",
    "Height_Ave_cm",
]
TARGET_ORDER = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]

train_wide = (
    train_long
    .pivot_table(
        index=ID_COLS,
        columns="target_name",
        values="target",
    )
    .reset_index()
)
train_wide = train_wide[ID_COLS + TARGET_ORDER]
print("train_wide shape:", train_wide.shape)
print(train_wide.head())

# ------------------------------------------------------------
# Weighted R2 (competition metric) - for evaluation during training
# ------------------------------------------------------------

TARGET_WEIGHTS = {
    "Dry_Green_g": 0.1,
    "Dry_Dead_g":  0.1,
    "Dry_Clover_g":0.1,
    "GDM_g":       0.2,
    "Dry_Total_g": 0.5,
}

def weighted_r2_long(df_long: pd.DataFrame,
                     y_col: str = "target",
                     yhat_col: str = "pred") -> float:
    df = df_long.copy()
    df["w"] = df["target_name"].map(TARGET_WEIGHTS).astype(float)

    y = df[y_col].values.astype(np.float64)
    yhat = df[yhat_col].values.astype(np.float64)
    w = df["w"].values.astype(np.float64)

    w_sum = np.sum(w)
    if w_sum == 0:
        return 0.0

    y_mean = np.sum(w * y) / w_sum
    ss_res = np.sum(w * (y - yhat) ** 2)
    ss_tot = np.sum(w * (y - y_mean) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)

tmp = train_long.copy()
tmp["pred"] = tmp.groupby("target_name")["target"].transform("mean")
baseline_r2 = weighted_r2_long(tmp, y_col="target", yhat_col="pred")
print(f"Baseline R2w (mean per target_name): {baseline_r2:.4f}")

# ------------------------------------------------------------
# Folds: GroupKFold by Sampling_Date at image level
# ------------------------------------------------------------

N_FOLDS = 5
train_wide["fold"] = -1

gkf = GroupKFold(n_splits=N_FOLDS)
groups = train_wide["Sampling_Date"].values

for fold, (_, val_idx) in enumerate(gkf.split(train_wide, groups=groups)):
    train_wide.loc[val_idx, "fold"] = fold

print("\nFold distribution (images):")
print(train_wide["fold"].value_counts().sort_index())
print("\nCrosstab State x fold:")
print(pd.crosstab(train_wide["State"], train_wide["fold"]))

train_wide.to_csv(PROCESSED_DATA_DIR / "train_wide_dinov3_multicrop.csv", index=False)

# ============================================================
# 3. TRAINING CONFIGURATION (CENTRALIZED)
# ============================================================

# Diversity strategy: different seed for head initialization
SEED = 123
seed_everything(SEED)

# Unique image size definition
# All subsequent sections will use this variable.
IMG_SIZE = 768  # Multiple of 14 (Dinov3 patch size)

if SMOKE_TEST:
    IMG_SIZE = SMOKE_IMG_SIZE

BACKBONE_NAME = str(BACKBONE_DIR) if BACKBONE_DIR.exists() else BACKBONE_NAME_OR_PATH

# Batch configuration
TRAIN_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
VALID_BATCH_SIZE = 4

# Hyperparameters
BASE_LR_HEADS = 1e-4
LORA_LR = 2e-4            # LR specific to LoRA layers
WEIGHT_DECAY = 0.05       # Slightly reduced for stability
DROPOUT = 0.5

EPOCHS_STAGE1 = 3         # Warmup heads only
EPOCHS_STAGE2 = 15        # Full finetuning (LoRA)
PATIENCE = 5

if SMOKE_TEST:
    TRAIN_BATCH_SIZE = 1
    GRAD_ACCUM_STEPS = 1
    VALID_BATCH_SIZE = 1
    EPOCHS_STAGE1 = 1
    EPOCHS_STAGE2 = 1

# ============================================================
# 4. AUGMENTATIONS
# ============================================================
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Ensure IMG_SIZE is defined
if 'IMG_SIZE' not in locals():
    IMG_SIZE = 336
    print(f"Warning: IMG_SIZE not found, set to {IMG_SIZE}")

print(f"Configuring Transforms with IMG_SIZE: {IMG_SIZE}...")

try:
    rrc = A.RandomResizedCrop(
        size=(IMG_SIZE, IMG_SIZE),
        scale=(0.8, 1.0),
        ratio=(0.9, 1.1),
        p=1.0
    )
except TypeError:
    rrc = A.RandomResizedCrop(
        height=IMG_SIZE,
        width=IMG_SIZE,
        scale=(0.8, 1.0),
        ratio=(0.9, 1.1),
        p=1.0
    )

train_transform = A.Compose([
    # NOTE: recent Albumentations expects size=(h, w) instead of separate height/width
    rrc,
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.1,
        rotate_limit=15,
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.7
    ),
    A.RandomBrightnessContrast(
        p=0.7,
        brightness_limit=0.2,
        contrast_limit=0.2
    ),
    A.HueSaturationValue(
        p=0.5,
        hue_shift_limit=10,
        sat_shift_limit=15,
        val_shift_limit=10
    ),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

valid_transform = A.Compose([
    # Albumentations Resize signature can vary by version; height/width is usually safe.
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

print("Transforms ready.")

# ============================================================
# 5. DATASET THREE-STREAM (left, center, right)
# ============================================================

class BiomassThreeStreamDataset(Dataset):
    def __init__(self, df, root_dir, transform, target_order):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_order = target_order

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_rel_path = row["image_path"]
        img_path = self.root_dir / Path(img_rel_path).name

        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _ = img.shape
        mid = w // 2

        left = img[:, :mid]
        right = img[:, mid:]

        center_w = w // 2
        start = (w - center_w) // 2
        center = img[:, start:start + center_w]

        left_t   = self.transform(image=left)["image"]
        center_t = self.transform(image=center)["image"]
        right_t  = self.transform(image=right)["image"]

        targets = np.array([row[t] for t in self.target_order],
                           dtype=np.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

        ndvi = torch.tensor(row["Pre_GSHH_NDVI"], dtype=torch.float32)
        height = torch.tensor(row["Height_Ave_cm"], dtype=torch.float32)

        return left_t, center_t, right_t, targets, ndvi, height

# ============================================================
# 6. QLORA MODEL WITH GEM POOLING (CORRECTED)
# ============================================================
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
import torch.nn.functional as F

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # Initialize p as a parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: [batch, sequence_len, channels]
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp for numerical stability -> pow -> mean -> inverse pow
        # Note: p must be on the same device as x
        return x.clamp(min=eps).pow(p).mean(dim=1).pow(1./p)

class Dinov3ThreeStreamDiversity(nn.Module):
    def __init__(self, backbone_name, dropout=0.5, pretrained=True, use_lora=True):
        super().__init__()

        print(f"Loading Backbone (4-bit) for Diversity Model: {backbone_name}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                backbone_name,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            config = AutoConfig.from_pretrained(backbone_name, trust_remote_code=True)
            self.backbone = AutoModel.from_config(config)

        self.backbone = prepare_model_for_kbit_training(self.backbone)

        if use_lora:
            print(">>> Applying QLoRA...")
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules="all-linear",
                lora_dropout=0.1,
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, peft_config)
            self.backbone.print_trainable_parameters()

            try:
                self.backbone.base_model.model.gradient_checkpointing_enable()
            except Exception as e:
                print(f"Checkpoint warning: {e}")

        hidden_size = self.backbone.config.hidden_size

        # Diversity strategy: GeM Pooling
        self.pool = GeM(p=3.0)
        self.feat_dim = hidden_size

        self.norm = nn.LayerNorm(self.feat_dim, dtype=torch.float32)

        fused_dim = self.feat_dim * 3

        # Heads
        self.head_targets = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Linear(fused_dim // 2, 5),
        )

        self.head_ndvi = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, 1),
        )

        self.head_height = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, 1),
        )

        # IMPORTANT FIX: move everything (including pooling) to device
        self.head_targets.to(device)
        self.head_ndvi.to(device)
        self.head_height.to(device)
        self.norm.to(device)
        self.pool.to(device) # <--- this line was missing and caused the error

    def forward_one(self, x):
        outputs = self.backbone(x)
        last_hidden_state = outputs.last_hidden_state
        patch_tokens = last_hidden_state[:, 1:, :]

        # GeM Pooling
        pooled = self.pool(patch_tokens)

        return self.norm(pooled.to(dtype=torch.float32))

    def forward(self, img_left, img_center, img_right):
        feat_l = self.forward_one(img_left)
        feat_c = self.forward_one(img_center)
        feat_r = self.forward_one(img_right)
        fused = torch.cat([feat_l, feat_c, feat_r], dim=1)

        pred_targets = self.head_targets(fused)
        pred_ndvi    = self.head_ndvi(fused).squeeze(1)
        pred_height  = self.head_height(fused).squeeze(1)
        return pred_targets, pred_ndvi, pred_height

def get_nuclear_sampler(df, target_col="Dry_Total_g"):
    """
    Aggressive weighted sampler to force the model to see very large plants
    (>120g) much more often.
    """
    targets = df[target_col].values

    # Base equalization (inverse frequency)
    counts, bins = np.histogram(targets, bins=20)
    idxs = np.digitize(targets, bins) - 1
    idxs = np.clip(idxs, 0, len(counts) - 1)
    weights = 1.0 / (counts[idxs] + 1e-6)

    # Nuclear boost multipliers

    # Critical band (>120g): the model should see these very frequently.
    # 15x multiplier
    mask_extreme = targets > 120
    weights[mask_extreme] *= 15.0

    # Bridge band (80g-120g): help the transition from medium to high.
    # 5x multiplier
    mask_high = (targets > 80) & (targets <= 120)
    weights[mask_high] *= 5.0

    # Noise band (<10g): stabilize near-empty samples.
    # 3x multiplier
    mask_low = targets < 10
    weights[mask_low] *= 3.0

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=len(df),
        replacement=True
    )

    print(f"Nuclear sampler created. Max weight: {weights.max():.2f}, Min weight: {weights.min():.2f}")
    return sampler

# ============================================================
# 2. NUCLEAR DATALOADER (FULL DATASET)
# ============================================================

def get_nuclear_full_dataloaders():
    # Use the entire train_wide
    train_df_full = train_wide.copy().reset_index(drop=True)

    if SMOKE_TEST:
        train_df_full = train_df_full.head(SMOKE_MAX_SAMPLES).copy().reset_index(drop=True)

    print(f"Preparing nuclear DataLoader on {len(train_df_full)} samples...")

    # Sampler
    sampler = get_nuclear_sampler(train_df_full, target_col="Dry_Total_g")

    train_ds = BiomassThreeStreamDataset(
        df=train_df_full,
        root_dir=DATA_DIR / "train",
        transform=train_transform,
        target_order=TARGET_ORDER,
    )

    # DataLoader with sampler
    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    return train_loader, train_df_full

# ============================================================
# 9. EVAL ON VALID (competition metric)
# ============================================================

@torch.no_grad()
def evaluate_model(model, valid_loader, valid_df):
    model.eval()
    all_preds = []

    for img_l, img_c, img_r, _, _, _ in valid_loader:
        img_l = img_l.to(device)
        img_c = img_c.to(device)
        img_r = img_r.to(device)

        # Explicit autocast
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            pred_targets, _, _ = model(img_l, img_c, img_r)

        preds = pred_targets.cpu().numpy()
        preds = np.maximum(0.0, preds)
        all_preds.append(preds)

    # ... rest of the function remains the same ...
    all_preds = np.concatenate(all_preds, axis=0)

    preds_wide = pd.DataFrame(
        {
            "image_path": valid_df["image_path"].values,
            "Dry_Green_g":  all_preds[:, 0],
            "Dry_Dead_g":   all_preds[:, 1],
            "Dry_Clover_g": all_preds[:, 2],
            "GDM_g":        all_preds[:, 3],
            "Dry_Total_g":  all_preds[:, 4],
        }
    )

    truth_wide = valid_df[["image_path"] + TARGET_ORDER].copy()

    preds_long = preds_wide.melt(
        id_vars=["image_path"],
        value_vars=TARGET_ORDER,
        var_name="target_name",
        value_name="pred",
    )
    truth_long = truth_wide.melt(
        id_vars=["image_path"],
        value_vars=TARGET_ORDER,
        var_name="target_name",
        value_name="target",
    )

    merged = truth_long.merge(preds_long,
                              on=["image_path", "target_name"],
                              how="left")
    score = weighted_r2_long(merged, y_col="target", yhat_col="pred")
    return score

# ============================================================
# 9. MSE LOSS (MODIFIED)
# ============================================================

# Instantiate standard PyTorch MSELoss
# reduction='none' is required so we can apply per-target weights before averaging
mse_criterion = nn.MSELoss(reduction='none')

loss_weight_tensor = torch.tensor(
    [0.10, 0.20, 0.20, 0.20, 0.30], dtype=torch.float32, device=device
).view(1, 5)

def biomass_mse_loss(pred_targets, y_targets, pred_ndvi, y_ndvi, pred_height, y_height):
    """
    Compute the total loss using weighted MSE.
    """
    # Explicit float32 cast for safety
    pred_targets = pred_targets.float()
    y_targets = y_targets.float()

    # 1. MSE for the 5 main targets
    # shape result: (batch_size, 5)
    mse_targets = mse_criterion(pred_targets, y_targets)

    # Apply weights per target and average
    l_targets = (mse_targets * loss_weight_tensor).mean()

    # 2. MSE for auxiliary variables (NDVI and Height)
    # Casting matters here as well
    l_ndvi = mse_criterion(pred_ndvi.float(), y_ndvi.float()).mean()
    l_height = mse_criterion(pred_height.float(), y_height.float()).mean()

    # 3. Final weighted sum (keeping the original coefficients 0.2 and 0.3)
    loss = l_targets + 0.2 * l_ndvi + 0.3 * l_height

    return loss

# ============================================================
# 10. EMA (MODIFIED FOR QLORA)
# ============================================================

def save_ema_params(ema_model, normal_model, path):
    """
    Save EMA parameters corresponding to trainable parameters in the normal model.
    """
    # Identify keys that are trainable in the normal model (LoRA + heads)
    trainable_keys = [n for n, p in normal_model.named_parameters() if p.requires_grad]

    ema_state_dict = ema_model.state_dict()
    custom_state_dict = {}

    # Extract matching values from the EMA model
    count = 0
    for key in trainable_keys:
        if key in ema_state_dict:
            custom_state_dict[key] = ema_state_dict[key].cpu()
            count += 1

    if count > 0:
        torch.save(custom_state_dict, path)
        print(f"✅ Saved EMA params to {path} ({count} keys)")
    else:
        print("Error: no parameters found to save from EMA.")

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.99):
        # Note: decay=0.99 is more reactive than 0.999 and avoids model lag
        self.decay = decay

        # Create a deep copy of the model for EMA
        self.ema = deepcopy(model)
        self.ema.eval()

        # Freeze the EMA model completely (no gradients)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        # Get state_dicts
        msd = model.state_dict()
        esd = self.ema.state_dict()

        # Iterate over parameters
        for k, v in esd.items():
            # In QLoRA/PEFT, we update only what changes (trainable params)
            # and buffers (e.g., running stats).
            # Backbone int8/nf4 weights are frozen in the original model, so msd[k]
            # does not change. Updating those is effectively a no-op.

            model_v = msd[k].to(v.device)

            if v.dtype.is_floating_point:
                # EMA formula: v_new = v_old * decay + v_current * (1 - decay)
                v.mul_(self.decay).add_(model_v * (1.0 - self.decay))
            else:
                # For integers (or quantized non-floats) copy directly
                v.copy_(model_v)

# ============================================================
# 11. TRAIN ONE FOLD (DIVERSITY VERSION)
# ============================================================

def save_trainable_params(model, path):
    trainable_keys = [n for n, p in model.named_parameters() if p.requires_grad]
    full_state_dict = model.state_dict()
    custom_state_dict = {}
    for key in trainable_keys:
        if key in full_state_dict:
            custom_state_dict[key] = full_state_dict[key].cpu()
    if len(custom_state_dict) > 0:
        torch.save(custom_state_dict, path)

# ============================================================
# 3. NUCLEAR TRAINING FUNCTION (FULL FIT)
# ============================================================

def train_nuclear_full():
    # Output folder for the nuclear model
    save_dir = str(MODEL_DIR / "dinov3-nuclear-768px")
    os.makedirs(save_dir, exist_ok=True)

    final_model_path = os.path.join(save_dir, "dinov3_nuclear_last.pth")

    swa_epochs = 5
    if SMOKE_TEST:
        swa_epochs = 1
    swa_epochs = min(swa_epochs, EPOCHS_STAGE2)
    swa_start_epoch = EPOCHS_STAGE2 - swa_epochs + 1
    swa_snapshots_dir = os.path.join(save_dir, "swa_snapshots")
    os.makedirs(swa_snapshots_dir, exist_ok=True)

    print(f"\n========== NUCLEAR FULL TRAINING (ViT-7B QLoRA) ==========")
    print(f"Strategies: Nuclear Sampler (15x) | GeM Pooling | MSE Loss")

    seed_everything(SEED)

    # 1. Load data with active sampler
    train_loader, train_df_full = get_nuclear_full_dataloaders()

    # 2. Initialize model (same architecture as base)
    model = Dinov3ThreeStreamDiversity(
        backbone_name=BACKBONE_NAME,
        dropout=DROPOUT,
        pretrained=True,
        use_lora=True
    )
    model.to(device)

    # EMA
    ema = ModelEMA(model, decay=0.95)
    scaler = torch.amp.GradScaler('cuda')

    # --- Stage 1: Head Warmup ---
    print("\n--- Stage 1: Head Warmup (Nuclear) ---")
    # Freeze everything except the heads
    for n, p in model.named_parameters(): p.requires_grad = False
    for head in [model.head_targets, model.head_ndvi, model.head_height]:
        for p in head.parameters(): p.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=BASE_LR_HEADS, weight_decay=WEIGHT_DECAY
    )

    for epoch in range(1, EPOCHS_STAGE1 + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Nuc S1 Ep {epoch}", leave=False)

        for i, batch in enumerate(pbar):
            img_l, img_c, img_r, targets, ndvi, height = batch
            img_l, img_c, img_r = img_l.to(device), img_c.to(device), img_r.to(device)
            targets, ndvi, height = targets.to(device), ndvi.to(device), height.to(device)

            with torch.amp.autocast('cuda'):
                pt, pn, ph = model(img_l, img_c, img_r)
                loss = biomass_mse_loss(pt, targets, pn, ndvi, ph, height)
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)

            running_loss += loss.item() * GRAD_ACCUM_STEPS * img_l.size(0)

        print(f"Nuc S1 Ep {epoch:02d} | Loss: {running_loss/len(train_df_full):.4f}")

    # --- Sync EMA ---
    ema.ema.load_state_dict(model.state_dict(), strict=False)

    # --- Stage 2: LoRA Full Training ---
    print("\n--- Stage 2: LoRA Nuclear Training ---")
    for n, p in model.named_parameters():
        if "lora" in n: p.requires_grad = True

    lora_params = [p for n, p in model.named_parameters() if "lora" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "lora" not in n and "backbone" not in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": LORA_LR},
        {"params": head_params, "lr": BASE_LR_HEADS},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_STAGE2, eta_min=1e-6)

    for epoch in range(1, EPOCHS_STAGE2 + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Nuc S2 Ep {epoch}", leave=False)

        for i, batch in enumerate(pbar):
            img_l, img_c, img_r, targets, ndvi, height = batch
            img_l, img_c, img_r = img_l.to(device), img_c.to(device), img_r.to(device)
            targets, ndvi, height = targets.to(device), ndvi.to(device), height.to(device)

            with torch.amp.autocast('cuda'):
                pt, pn, ph = model(img_l, img_c, img_r)
                loss = biomass_mse_loss(pt, targets, pn, ndvi, ph, height)
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)

            running_loss += loss.item() * GRAD_ACCUM_STEPS * img_l.size(0)

        scheduler.step()
        print(f"Nuc S2 Ep {epoch:02d} | Loss: {running_loss/len(train_df_full):.4f}")

        if epoch >= swa_start_epoch:
            snapshot_idx = epoch - swa_start_epoch + 1
            snapshot_path = os.path.join(swa_snapshots_dir, f"swa_snapshot_ep{snapshot_idx}.pth")
            save_ema_params(ema_model=ema.ema, normal_model=model, path=snapshot_path)

    # --- SAVE ---
    print(f"Saving nuclear model to: {final_model_path}")
    save_ema_params(ema_model=ema.ema, normal_model=model, path=final_model_path)

    return final_model_path

# ============================================================
# 4. RUN
# ============================================================

gc.collect()
torch.cuda.empty_cache()

try:
    nuclear_model_path = train_nuclear_full()
    print("Nuclear specialist training completed successfully.")
except Exception as e:
    print(f"Error during nuclear specialist training: {e}")
    import traceback
    traceback.print_exc()

import torch
import os
import copy
from tqdm import tqdm

# ============================================================
# SWA ON TRAINED SPECIALIST (LOCAL)
# ============================================================
SWA_EPOCHS = 5
SWA_LR = 1e-5

if SMOKE_TEST:
    SWA_EPOCHS = 1

source_path = nuclear_model_path
if not os.path.exists(source_path):
    source_path = os.path.join(str(MODEL_DIR / "dinov3-nuclear-768px"), "dinov3_nuclear_last.pth")

swa_output_dir = str(MODEL_DIR / "dinov3-high-specialist-swa")
BASE_DIR = swa_output_dir
SWA_SNAPSHOTS_DIR = os.path.join(BASE_DIR, "swa_snapshots")

os.makedirs(SWA_SNAPSHOTS_DIR, exist_ok=True)

print("--- Starting SWA procedure ---")

if not os.path.exists(source_path):
    raise FileNotFoundError(f"Base checkpoint not found at: {source_path}")

print(f"Loading initial weights from: {source_path}")
print(f"Output snapshots in: {SWA_SNAPSHOTS_DIR}")

SWA_SNAPSHOTS_DIR = os.path.join(str(MODEL_DIR / "dinov3-nuclear-768px"), "swa_snapshots")
snapshot_paths = []
effective_swa_epochs = min(SWA_EPOCHS, EPOCHS_STAGE2)
for epoch in range(1, effective_swa_epochs + 1):
    path = os.path.join(SWA_SNAPSHOTS_DIR, f"swa_snapshot_ep{epoch}.pth")
    if os.path.exists(path):
        snapshot_paths.append(path)

# 5. SWA aggregation
print("\n--- Computing averaged weights (SWA) ---")
if len(snapshot_paths) > 0:
    swa_state_dict = torch.load(snapshot_paths[0], map_location="cpu")

    for path in snapshot_paths[1:]:
        current_sd = torch.load(path, map_location="cpu")
        for key in swa_state_dict:
            swa_state_dict[key] += current_sd[key]

    n_snapshots = len(snapshot_paths)
    for key in swa_state_dict:
        swa_state_dict[key] /= n_snapshots

    # 6. Final save
    swa_save_path = os.path.join(BASE_DIR, "dinov3_high_specialist_swa.pth")
    torch.save(swa_state_dict, swa_save_path)

    print("="*50)
    print(f"SWA model saved to:\n{swa_save_path}")
    print("This is the weight file used for inference.")
    print("="*50)
else:
    print("No snapshots saved; cannot create SWA model.")