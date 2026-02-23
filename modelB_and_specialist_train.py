# ============================================================
# 0. INSTALL, DOWNLOAD & SETUP
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
# 1. IMPORTS AND BASE CONFIGURATION
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
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupKFold

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm.auto import tqdm
from transformers import AutoModel, AutoConfig

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

# The dataset must be available under RAW_DATA_DIR (see SETTINGS.json)
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
# Fold: GroupKFold per Sampling_Date a livello immagine
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

# UNIQUE IMAGE SIZE DEFINITION
# All subsequent sections will use this variable.
IMG_SIZE = 768  # Multiple of 14 (patch size Dinov3): 336/14 = 24

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

EPOCHS_STAGE1 = 3         # Warmup only heads
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

# Ensure IMG_SIZE is defined (if you haven't run Section 3 correctly above)
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
    # FIX: Recent Albumentations wants 'size'=(h, w) instead of separate height/width
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
# 6. QLORA MODEL WITH GeM POOLING (CORRECTED)
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
        # Clamp for numerical stability -> pow -> mean -> pow inverse
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

        # DIVERSITY STRATEGY: GeM Pooling
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

        # === IMPORTANT FIX: MOVE EVERYTHING (INCLUDING POOL) TO DEVICE ===
        self.head_targets.to(device)
        self.head_ndvi.to(device)
        self.head_height.to(device)
        self.norm.to(device)
        self.pool.to(device) # <--- THIS LINE WAS MISSING AND CAUSED THE ERROR

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

# ============================================================
# 8. DATALOADER FULL DATA (ALL DATA)
# ============================================================

def get_full_dataloaders():
    # Use the entire train_wide for training
    train_df_full = train_wide.copy().reset_index(drop=True)

    if SMOKE_TEST:
        train_df_full = train_df_full.head(SMOKE_MAX_SAMPLES).copy().reset_index(drop=True)

    # No validation set in this scenario
    print(f"Full Training Data Size: {len(train_df_full)}")

    train_ds = BiomassThreeStreamDataset(
        df=train_df_full,
        root_dir=DATA_DIR / "train",
        transform=train_transform, # Use augmentations
        target_order=TARGET_ORDER,
    )

    # Single DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True, # Important to shuffle
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, train_df_full

# ============================================================
# 9. EVALUATION ON VALIDATION (official metric)
# ============================================================

@torch.no_grad()
def evaluate_model(model, valid_loader, valid_df):
    model.eval()
    all_preds = []

    for img_l, img_c, img_r, _, _, _ in valid_loader:
        img_l = img_l.to(device)
        img_c = img_c.to(device)
        img_r = img_r.to(device)

        # === FIX: Explicit autocast syntax ===
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
# reduction='none' is crucial to apply weights to individual targets before mean
mse_criterion = nn.MSELoss(reduction='none')

# Loss weights for the 5 main targets (ensure float32)
loss_weight_tensor = torch.tensor(
    [0.10, 0.20, 0.20, 0.20, 0.30], dtype=torch.float32, device=device
).view(1, 5)

def biomass_mse_loss(pred_targets, y_targets, pred_ndvi, y_ndvi, pred_height, y_height):
    """
    Compute total loss using weighted MSE.
    """
    # Explicit float32 cast for safety
    pred_targets = pred_targets.float()
    y_targets = y_targets.float()

    # 1. Compute MSE for the 5 main targets
    # shape result: (batch_size, 5)
    mse_targets = mse_criterion(pred_targets, y_targets)

    # Apply weights for each column (target) and compute mean
    l_targets = (mse_targets * loss_weight_tensor).mean()

    # 2. Compute MSE for auxiliary variables (NDVI and Height)
    # Important to cast here as well
    l_ndvi = mse_criterion(pred_ndvi.float(), y_ndvi.float()).mean()
    l_height = mse_criterion(pred_height.float(), y_height.float()).mean()

    # 3. Weighted sum (keeping original coefficients 0.2 and 0.3)
    loss = l_targets + 0.2 * l_ndvi + 0.3 * l_height

    return loss

# ============================================================
# 10. EMA (MODIFIED FOR QLORA & FIX LAG)
# ============================================================

def save_ema_params(ema_model, normal_model, path):
    """Save EMA parameters corresponding to trainable parameters in the normal model."""
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
        # Note: decay=0.99 is more reactive than 0.999.
        self.decay = decay

        # Create a deep copy for EMA.
        self.ema = deepcopy(model)
        self.ema.eval()

        # Freeze EMA model parameters
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        # Read state_dicts
        msd = model.state_dict()
        esd = self.ema.state_dict()

        # Iterate over parameters
        for k, v in esd.items():
            # In QLoRA/PEFT we mostly update trainable params and buffers.

            model_v = msd[k].to(v.device)

            if v.dtype.is_floating_point:
                # Formula EMA: v_new = v_old * decay + v_current * (1 - decay)
                v.mul_(self.decay).add_(model_v * (1.0 - self.decay))
            else:
                # For non-floats, copy directly
                v.copy_(model_v)

# ============================================================
# 11. TRAIN ONE FOLD (MODIFIED FOR DIVERSITY)
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
# 11. TRAIN FULL DATA (MSE LOSS)
# ============================================================

def train_full_dataset():
    # Output folder for the full-fit model
    save_dir = str(MODEL_DIR / "dinov3-mse-768px")
    os.makedirs(save_dir, exist_ok=True)

    final_model_path = os.path.join(save_dir, "dinov3_full_last_mse.pth")

    swa_epochs = 5
    if SMOKE_TEST:
        swa_epochs = 1
    swa_epochs = min(swa_epochs, EPOCHS_STAGE2)
    swa_start_epoch = EPOCHS_STAGE2 - swa_epochs + 1
    swa_snapshots_dir = os.path.join(save_dir, "swa_snapshots")
    os.makedirs(swa_snapshots_dir, exist_ok=True)

    print(f"\n========== TRAINING FULL DATASET (MSE LOSS) - Img: {IMG_SIZE} ==========")

    # Global seed
    seed_everything(SEED)

    # Load all data
    train_loader, train_df_full = get_full_dataloaders()

    # Initialize model
    model = Dinov3ThreeStreamDiversity(
        backbone_name=BACKBONE_NAME,
        dropout=DROPOUT,
        pretrained=True,
        use_lora=True
    )
    model.to(device) # Ensure model is on device

    # EMA initialization
    ema = ModelEMA(model, decay=0.95)
    print("EMA Initialized")

    scaler = torch.amp.GradScaler('cuda')

    # --- Stage 1: Head Warmup ---
    print("\n--- Stage 1: Head Warmup ---")
    for n, p in model.named_parameters():
        p.requires_grad = False
    for head in [model.head_targets, model.head_ndvi, model.head_height]:
        for p in head.parameters(): p.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=BASE_LR_HEADS, weight_decay=WEIGHT_DECAY
    )

    for epoch in range(1, EPOCHS_STAGE1 + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"S1 Ep {epoch}", leave=False)

        for i, batch in enumerate(pbar):
            img_l, img_c, img_r, targets, ndvi, height = batch
            img_l, img_c, img_r = img_l.to(device), img_c.to(device), img_r.to(device)
            targets, ndvi, height = targets.to(device), ndvi.to(device), height.to(device)

            with torch.amp.autocast('cuda'):
                pt, pn, ph = model(img_l, img_c, img_r)
                # --- MODIFIED HERE: Call to MSE loss ---
                loss = biomass_mse_loss(pt, targets, pn, ndvi, ph, height)
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)

            running_loss += loss.item() * GRAD_ACCUM_STEPS

        epoch_loss = running_loss / len(train_df_full)
        print(f"S1 Ep {epoch:02d} | Train Loss (MSE): {epoch_loss:.4f}")

    # --- Sync EMA before Stage 2 ---
    ema.ema.load_state_dict(model.state_dict(), strict=False)

    # --- Stage 2: LoRA Training ---
    print("\n--- Stage 2: LoRA Full Training ---")
    for n, p in model.named_parameters():
        if "lora" in n: p.requires_grad = True

    lora_params = [p for n, p in model.named_parameters() if "lora" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "lora" not in n and "backbone" not in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": LORA_LR},
        {"params": head_params, "lr": BASE_LR_HEADS},
    ], weight_decay=WEIGHT_DECAY)

    # Scheduler: Cosine Decay until the end
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_STAGE2, eta_min=1e-6)

    for epoch in range(1, EPOCHS_STAGE2 + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"S2 Ep {epoch}", leave=False)
        for i, batch in enumerate(pbar):
            img_l, img_c, img_r, targets, ndvi, height = batch
            img_l, img_c, img_r = img_l.to(device), img_c.to(device), img_r.to(device)
            targets, ndvi, height = targets.to(device), ndvi.to(device), height.to(device)

            with torch.amp.autocast('cuda'):
                pt, pn, ph = model(img_l, img_c, img_r)
                # --- MODIFIED HERE: Call to MSE loss ---
                loss = biomass_mse_loss(pt, targets, pn, ndvi, ph, height)
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)

            running_loss += loss.item() * GRAD_ACCUM_STEPS

        scheduler.step()
        epoch_loss = running_loss / len(train_df_full)
        print(f"S2 Ep {epoch:02d} | Train Loss (MSE): {epoch_loss:.4f}")

        if epoch >= swa_start_epoch:
            snapshot_idx = epoch - swa_start_epoch + 1
            snapshot_path = os.path.join(swa_snapshots_dir, f"swa_snapshot_ep{snapshot_idx}.pth")
            save_ema_params(ema_model=ema.ema, normal_model=model, path=snapshot_path)

    # --- FINAL SAVE ---
    print(f"Saving final model to: {final_model_path}")
    save_ema_params(ema_model=ema.ema, normal_model=model, path=final_model_path)

    return final_model_path

# ============================================================
# 12. RUN FULL-FIT TRAINING
# ============================================================

import gc
gc.collect()
torch.cuda.empty_cache()

try:
    final_model_path = train_full_dataset()
    print("\nFull-fit training completed.")
except Exception as e:
    print(f"Fatal error: {e}")
    import traceback
    traceback.print_exc()

import copy

# ============================================================
# SWA CONFIGURATION (FULL DATA)
# ============================================================
SWA_EPOCHS = 5
SWA_LR = 1e-5
BASE_DIR = str(MODEL_DIR / "dinov3-mse-768px")
LAST_MODEL_PATH = os.path.join(BASE_DIR, "dinov3_full_last_mse.pth")

if SMOKE_TEST:
    SWA_EPOCHS = 1
SWA_SNAPSHOTS_DIR = os.path.join(BASE_DIR, "swa_snapshots")

os.makedirs(SWA_SNAPSHOTS_DIR, exist_ok=True)

print("--- Starting SWA procedure (Full Fit) ---")
print(f"Loading final weights from: {LAST_MODEL_PATH}")

snapshot_paths = []
effective_swa_epochs = min(SWA_EPOCHS, EPOCHS_STAGE2)
for epoch in range(1, effective_swa_epochs + 1):
    path = os.path.join(SWA_SNAPSHOTS_DIR, f"swa_snapshot_ep{epoch}.pth")
    if os.path.exists(path):
        snapshot_paths.append(path)

# 3. SWA aggregation
print("\n--- Computing averaged weights (SWA) ---")
if len(snapshot_paths) == 0:
    raise FileNotFoundError(f"No SWA snapshots found in: {SWA_SNAPSHOTS_DIR}")

swa_state_dict = torch.load(snapshot_paths[0], map_location="cpu")

for path in snapshot_paths[1:]:
    current_sd = torch.load(path, map_location="cpu")
    for key in swa_state_dict:
        swa_state_dict[key] += current_sd[key]

n_snapshots = len(snapshot_paths)
for key in swa_state_dict:
    swa_state_dict[key] /= n_snapshots

# 4. FINAL SAVE
swa_save_path = os.path.join(BASE_DIR, "dinov3_full_SWA.pth")
torch.save(swa_state_dict, swa_save_path)
print("="*50)
print(f"SWA FULL-FIT model saved to: {swa_save_path}")
print("This is the weight file used for inference on the test set.")
print("="*50)

# ============================================================
# HIGH-BIOMASS SPECIALIST CONFIGURATION (FULL DATASET)
# ============================================================

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler, DataLoader
from tqdm.auto import tqdm
from copy import deepcopy

# --- CONFIGURATION ---
SAVE_DIR = str(MODEL_DIR / "dinov3-mse-high-specialist-768px")
os.makedirs(SAVE_DIR, exist_ok=True)

# Path to the previously trained SWA model
PRETRAINED_PATH = os.path.join(str(MODEL_DIR / "dinov3-mse-768px"), "dinov3_full_SWA.pth")

# Specialist parameters
HIGH_BIOMASS_THRESHOLD = 80.0
HIGH_WEIGHT = 10.0      # Oversample 10x for cases > 80g
EPOCHS_SPECIALIST = 6   # Few epochs
LR_SPECIALIST = 1e-6    # Very low LR for safety with MSE
TRAIN_BATCH_SIZE = 2    # Low for handling 768px
GRAD_ACCUM_STEPS = 4    # Gradient accumulation

if SMOKE_TEST:
    EPOCHS_SPECIALIST = 1
    TRAIN_BATCH_SIZE = 1
    GRAD_ACCUM_STEPS = 1

# Ensure IMG_SIZE is set
if 'IMG_SIZE' not in locals():
    IMG_SIZE = 768

print(f"Targeting High Biomass > {HIGH_BIOMASS_THRESHOLD}g with IMG_SIZE {IMG_SIZE}")
print(f"Loss Strategy: Asymmetric MSE with Gradient Clipping (Max Norm 3.0)")

# ============================================================
# 1. UTILITY FUNCTIONS: ASYMMETRIC MSE + MIXUP
# ============================================================

def asymmetric_mse_loss(pred, target, threshold=80.0, penalty_factor=3.0):
    """
    Asymmetric MSE (Mean Squared Error).
    Applies a higher penalty when the target is high (> threshold)
    and the model underestimates (pred < target).
    """
    # Raw element-wise MSE (no reduction yet). Cast to float32 for stability.
    loss = (pred.float() - target.float()) ** 2

    # Identify critical cases: high target + underestimation
    is_high = target > threshold
    is_underestimated = pred < target
    mask_penalty = is_high & is_underestimated

    # Apply penalty
    loss[mask_penalty] *= penalty_factor

    return loss.mean()

def mixup_data(x1, x2, x3, y, alpha=0.4):
    """MixUp for a 3-image input."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size(0)
    index = torch.randperm(batch_size).to(x1.device)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    mixed_x3 = lam * x3 + (1 - lam) * x3[index, :]

    # Target mix
    y_a, y_b = y, y[index]
    mixed_y = lam * y_a + (1 - lam) * y_b

    return mixed_x1, mixed_x2, mixed_x3, mixed_y, lam

# ============================================================
# 2. WEIGHTED DATALOADER (WeightedRandomSampler)
# ============================================================

def get_specialist_loader(df):
    df_full = df.reset_index(drop=True)

    if SMOKE_TEST:
        df_full = df_full.head(SMOKE_MAX_SAMPLES).copy().reset_index(drop=True)

    targets_total = df_full["Dry_Total_g"].values

    # Sampling weights
    sample_weights = []
    count_high = 0
    for t in targets_total:
        if t > HIGH_BIOMASS_THRESHOLD:
            sample_weights.append(HIGH_WEIGHT)
            count_high += 1
        else:
            sample_weights.append(1.0)

    print(f"Full Dataset: {len(df_full)}")
    print(f"High Biomass Samples (> {HIGH_BIOMASS_THRESHOLD}g): {count_high}")

    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # Uses transforms and dataset class defined above
    train_ds = BiomassThreeStreamDataset(
        df=df_full,
        root_dir=DATA_DIR / "train",
        transform=train_transform, # Augmentations active
        target_order=TARGET_ORDER
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    return train_loader

# ============================================================
# 3. TRAINING LOOP SPECIALIST (WITH CLIPPING)
# ============================================================

def train_specialist():
    # 1. Load data
    train_loader = get_specialist_loader(train_wide)

    # 2. Initialize model
    print("\nInitializing QLoRA model...")
    model = Dinov3ThreeStreamDiversity(
        backbone_name=BACKBONE_NAME,
        dropout=0.5,
        pretrained=True,
        use_lora=True
    ).to(device)

    # 3. Load previously trained SWA weights (local)
    if os.path.exists(PRETRAINED_PATH):
        print(f"Loading SWA weights from: {PRETRAINED_PATH}")
        state_dict = torch.load(PRETRAINED_PATH, map_location=device)
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Weights loaded. Missing keys (expected for backbone): {len(msg.missing_keys)}")
    else:
        raise FileNotFoundError(f"Base model weights not found at: {PRETRAINED_PATH}")

    # 4. Enable gradients (LoRA + heads)
    for n, p in model.named_parameters():
        p.requires_grad = False # Reset

    trainable_params = []
    for n, p in model.named_parameters():
        if "lora" in n or "head" in n or "pool" in n:
            p.requires_grad = True
            trainable_params.append(p)

    print(f"Trainable parameter groups: {len(trainable_params)}")

    # 5. Setup optimizer and EMA
    # NOTE: LR reduced to 1e-6 for safety with MSE
    optimizer = torch.optim.AdamW(trainable_params, lr=LR_SPECIALIST, weight_decay=1e-2)
    ema = ModelEMA(model, decay=0.99)
    scaler = torch.amp.GradScaler('cuda')

    # 6. Loop
    print("\n=== START SPECIALIST TRAINING (MSE + CLIPPING) ===")

    for epoch in range(1, EPOCHS_SPECIALIST + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Spec Ep {epoch}", leave=False)

        for i, batch in enumerate(pbar):
            img_l, img_c, img_r, targets, ndvi, height = batch
            img_l, img_c, img_r = img_l.to(device), img_c.to(device), img_r.to(device)
            targets, ndvi, height = targets.to(device), ndvi.to(device), height.to(device)

            # --- MIXUP (50% probability) ---
            do_mixup = np.random.random() < 0.5
            if do_mixup:
                img_l, img_c, img_r, targets_mix, _ = mixup_data(img_l, img_c, img_r, targets)
                target_for_loss = targets_mix
            else:
                target_for_loss = targets

            with torch.amp.autocast('cuda'):
                pred_targets, pred_ndvi, pred_height = model(img_l, img_c, img_r)

                # CAST float32 for loss stability
                pred_targets = pred_targets.float()
                target_for_loss = target_for_loss.float()

                # --- MODIFIED HERE: Asymmetric MSE loss ---
                loss_bio = asymmetric_mse_loss(
                    pred_targets,
                    target_for_loss,
                    threshold=HIGH_BIOMASS_THRESHOLD,
                    penalty_factor=3.0 # Very aggressive with MSE, clipping will save us
                )

                # Aux Loss (reduced)
                loss_aux = 0.1 * F.mse_loss(pred_ndvi.float(), ndvi.float()) + \
                           0.1 * F.mse_loss(pred_height.float(), height.float())

                loss = (loss_bio + loss_aux) / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                # --- MODIFIED HERE: Gradient clipping ---
                # Unscale before clipping is required when using amp.GradScaler
                scaler.unscale_(optimizer)

                # Max norm 3.0: allows strong gradients but clips explosions
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)

            running_loss += loss.item() * GRAD_ACCUM_STEPS

        avg_loss = running_loss / len(train_loader)
        print(f"Ep {epoch:02d} | Specialist Loss (MSE): {avg_loss:.4f}")

    # 7. Save
    final_path = os.path.join(SAVE_DIR, "dinov3_high_specialist.pth")
    save_ema_params(ema.ema, model, final_path)
    print(f"\nSpecialist model saved to: {final_path}")
    return final_path

# Run
try:
    specialist_model_path = train_specialist()
except Exception as e:
    print(f"Fatal error during specialist training: {e}")
    import traceback
    traceback.print_exc()