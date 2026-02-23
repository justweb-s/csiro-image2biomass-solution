import sys
import subprocess
import os
import gc
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm

# =========================================================================
# 0. PULIZIA AMBIENTE (KILL CONFLICTS)
# =========================================================================
print("🧹 Pulizia ambiente dai conflitti (TensorFlow/TensorBoard)...")
pkgs_to_remove = ["tensorflow", "tensorboard", "flax", "keras"]
devnull = open(os.devnull, 'w')
for pkg in pkgs_to_remove:
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", pkg], 
        stdout=devnull, stderr=subprocess.PIPE
    )
devnull.close()

os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# =========================================================================
# 1. SEEDING & SETUP
# =========================================================================
def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

LIB_DIR = Path("/kaggle/input/csiro-inference-libraries")
sys.path.append(str(LIB_DIR))

# =========================================================================
# 2. INSTALLAZIONE CHIRURGICA
# =========================================================================
wheels_to_install = [
    "bitsandbytes-0.49.0-py3-none-manylinux_2_24_x86_64.whl",
    "safetensors-0.7.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "huggingface_hub-0.36.0-py3-none-any.whl",
    "accelerate-1.12.0-py3-none-any.whl",
    "tokenizers-0.22.1-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "peft-0.18.0-py3-none-any.whl",
    "transformers-4.57.3-py3-none-any.whl"
]

print("📦 Inizio installazione chirurgica (No-Deps)...")
devnull = open(os.devnull, 'w')

for whl in wheels_to_install:
    path = LIB_DIR / whl
    if not path.exists():
        print(f"⚠️ ATTENZIONE: {whl} non trovato, salto.")
        continue
    
    cmd = [
        sys.executable, "-m", "pip", "install", 
        str(path), 
        "--no-deps", 
        "--force-reinstall",
        "--quiet"
    ]
    try:
        subprocess.run(cmd, check=True, stdout=devnull, stderr=subprocess.PIPE)
        print(f"✅ Installato: {whl}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore inst {whl}: {e}")

devnull.close()

# =========================================================================
# 3. IMPORT FINALI (SOLO ORA CARICHIAMO TORCH/TRANSFORMERS)
# =========================================================================
print("\n🎉 Importazione librerie...")
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import bitsandbytes
import peft
import transformers
from transformers import AutoModel, AutoConfig, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

seed_everything(123)

print(f"BitsAndBytes version: {bitsandbytes.__version__}")
print(f"Transformers version: {transformers.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device rilevato: {device}")

# ============================================================
# 4. CONFIGURAZIONE DATA & MODELLI
# ============================================================

# --- PERCORSI ---
DATA_DIR = Path("/kaggle/input/csiro-biomass")
BACKBONE_PATH = "/kaggle/input/dinov3-vit7b-backbone" 

# Modelli Base
MODEL_PATH_BASE_1 = Path("/kaggle/input/dinov3-swa-full-fit-768px/dinov3_full_SWA.pth")
MODEL_PATH_BASE_2 = Path("/kaggle/input/dinov3-mse-768px/dinov3_full_SWA.pth")

# Modelli Specialista (High Biomass)
MODEL_PATH_SPEC_1 = Path("/kaggle/input/dinov3-high-specialist-768px/dinov3_high_specialist.pth")
MODEL_PATH_SPEC_2 = Path("/kaggle/input/dinov3-high-specialist-swa/dinov3_high_specialist_swa.pth")
# --- NUOVO MODELLO SPECIALISTA AGGIUNTO ---
MODEL_PATH_SPEC_3 = Path("/kaggle/input/dinov3-mse-high-specialist-768px/dinov3_high_specialist.pth")

IMG_SIZE = 768
BATCH_SIZE = 4 
TARGET_ORDER = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]

# ============================================================
# 5. DATASET & TRANSFORMS
# ============================================================
test_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.ToTensorV2(),
])

class TestBiomassThreeStreamDataset(Dataset):
    def __init__(self, df, root_dir, transform):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = Path(row["image_path"]).name
        img_path = self.root_dir / img_name
        
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

        return left_t, center_t, right_t

# ============================================================
# 6. ARCHITETTURA MODELLO
# ============================================================

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
    def gem(self, x, p=3, eps=1e-6):
        return x.clamp(min=eps).pow(p).mean(dim=1).pow(1./p)

class Dinov3Inference(nn.Module):
    def __init__(self, backbone_path, dropout=0.0):
        super().__init__()
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        config = AutoConfig.from_pretrained(str(backbone_path), trust_remote_code=True)
        
        self.backbone = AutoModel.from_pretrained(
            str(backbone_path),
            config=config,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
            local_files_only=True
        )
        
        peft_config = LoraConfig(
            r=16, lora_alpha=32, target_modules="all-linear",
            lora_dropout=0.1, bias="none",
        )
        self.backbone = get_peft_model(self.backbone, peft_config)
        
        hidden_size = self.backbone.config.hidden_size
        
        self.pool = GeM(p=3.0)
        self.feat_dim = hidden_size
        self.norm = nn.LayerNorm(self.feat_dim, dtype=torch.float32)
        fused_dim = self.feat_dim * 3
        
        self.head_targets = nn.Sequential(
            nn.Linear(fused_dim, fused_dim), 
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Linear(fused_dim // 2, 5),
        )

    def forward_one(self, x):
        outputs = self.backbone(x)
        last_hidden_state = outputs.last_hidden_state
        patch_tokens = last_hidden_state[:, 1:, :]
        pooled = self.pool(patch_tokens)
        return self.norm(pooled.to(dtype=torch.float32))

    def forward(self, img_left, img_center, img_right):
        feat_l = self.forward_one(img_left)
        feat_c = self.forward_one(img_center)
        feat_r = self.forward_one(img_right)
        fused = torch.cat([feat_l, feat_c, feat_r], dim=1)
        return self.head_targets(fused)

# ============================================================
# 7. FUNZIONE DI INFERENZA SEQUENZIALE
# ============================================================

def get_predictions(model_path, dataloader, model_name="Model"):
    print(f"\n--- Elaborazione: {model_name} ---")
    print(f"📂 Caricamento pesi da: {model_path.name}")
    
    if not model_path.exists():
        print(f"⚠️ PATH NON TROVATO: {model_path}")
        return None

    model = Dinov3Inference(BACKBONE_PATH)
    
    state_dict = torch.load(model_path, map_location="cpu")
    msg = model.load_state_dict(state_dict, strict=False)
    
    del state_dict
    gc.collect()
    
    model.head_targets.to(device)
    model.norm.to(device)
    model.pool.to(device)
    model.eval()
    
    preds_list = []
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for img_l, img_c, img_r in tqdm(dataloader, desc=f"Inferenza {model_name}"):
                img_l = img_l.to(device)
                img_c = img_c.to(device)
                img_r = img_r.to(device)
                
                out = model(img_l, img_c, img_r)
                preds_list.append(out.float().cpu().numpy())
    
    del model
    del img_l, img_c, img_r, out
    gc.collect()
    torch.cuda.empty_cache()
    
    return np.concatenate(preds_list, axis=0)

# ============================================================
# 8. ESECUZIONE PRINCIPALE & LOGICA ENSEMBLE
# ============================================================

try:
    # --- SETUP DATI ---
    test_long = pd.read_csv(DATA_DIR / "test.csv")
    test_unique = test_long.drop_duplicates("image_path").reset_index(drop=True)
    TEST_IMG_DIR = DATA_DIR / "test"
    
    ds = TestBiomassThreeStreamDataset(test_unique, TEST_IMG_DIR, test_transform)
    
    # IMPORTANTE: num_workers=0 evita il deadlock dei thread e gli errori OSError
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- 1. INFERENZA MODELLI BASE ---
    print("\n>>> Avvio Ensemble Base Models...")
    preds_b1 = get_predictions(MODEL_PATH_BASE_1, dl, model_name="BASE 1 (Original)")
    if preds_b1 is None: raise FileNotFoundError(f"Missing {MODEL_PATH_BASE_1}")
    
    preds_b2 = get_predictions(MODEL_PATH_BASE_2, dl, model_name="BASE 2 (New NR)")
    if preds_b2 is None: raise FileNotFoundError(f"Missing {MODEL_PATH_BASE_2}")

    print("\n>>> Creazione 'Base Ensemble' (Media Base 1 + Base 2)...")
    preds_base = (preds_b1 + preds_b2) / 2.0
    preds_base = np.maximum(0, preds_base) 

    # --- 2. INFERENZA MODELLI SPECIALISTA ---
    print("\n>>> Avvio Ensemble Specialist Models (3 Modelli)...")
    preds_spec_1 = get_predictions(MODEL_PATH_SPEC_1, dl, model_name="SPEC 1 (High Original)")
    preds_spec_2 = get_predictions(MODEL_PATH_SPEC_2, dl, model_name="SPEC 2 (High MSE)")
    preds_spec_3 = get_predictions(MODEL_PATH_SPEC_3, dl, model_name="SPEC 3 (High MSE New)") # Nuovo Modello

    # Logica Ensemble Specialisti aggiornata per 3 modelli
    spec_preds_list = []
    if preds_spec_1 is not None: spec_preds_list.append(preds_spec_1)
    if preds_spec_2 is not None: spec_preds_list.append(preds_spec_2)
    if preds_spec_3 is not None: spec_preds_list.append(preds_spec_3)

    if len(spec_preds_list) > 0:
        print(f"\n>>> Creazione 'Specialist Ensemble' (Media di {len(spec_preds_list)} modelli)...")
        preds_spec = np.mean(spec_preds_list, axis=0)
    else:
        print("⚠️ NESSUN SPECIALISTA! Uso Base.")
        preds_spec = preds_base.copy()

    preds_spec = np.maximum(0, preds_spec)

    # ============================================================
    # NUOVE REGOLE DI POST-PROCESSING
    # ============================================================
    print("\n🛠️ Applicazione regole di business agli Specialisti...")
    
    # 1. Soglia rumore
    mask_noise = preds_spec < 0.3
    n_clipped = np.sum(mask_noise)
    preds_spec[mask_noise] = 0.0
    print(f"   -> Valori < 0.3 azzerati: {n_clipped} celle")

    # 2. Cap a 200g per Dry_Total_g
    IDX_TOTAL = 4
    mask_cap = preds_spec[:, IDX_TOTAL] > 200.0
    n_capped = np.sum(mask_cap)
    preds_spec[mask_cap, IDX_TOTAL] = 200.0
    print(f"   -> Dry_Total_g > 200 limitati a 200: {n_capped} campioni")
    
    # ============================================================
    # LOGICA DI MERGE
    # ============================================================
    print("\n--- Applicazione Logica Ensemble Condizionale ---")
    
    THRESHOLD = 55.0 
    WEIGHT_BASE = 0.0
    WEIGHT_SPEC = 1.0
    
    final_preds = preds_base.copy()
    
    mask_high = preds_base[:, IDX_TOTAL] > THRESHOLD
    count_high = np.sum(mask_high)
    
    print(f"Campioni totali: {len(final_preds)}")
    print(f"Campioni sopra soglia {THRESHOLD}g: {count_high}")
    
    if count_high > 0:
        final_preds[mask_high] = (preds_base[mask_high] * WEIGHT_BASE) + (preds_spec[mask_high] * WEIGHT_SPEC)
    else:
        print(">>> Nessun campione sopra soglia. Uso solo predizioni Base Ensemble.")
        
    # ============================================================
    # 7. EXPORT FINALE
    # ============================================================
    
    dry_green  = final_preds[:, 0]
    dry_dead   = final_preds[:, 1]
    dry_clover = final_preds[:, 2]
    gdm        = final_preds[:, 3]
    dry_total  = final_preds[:, 4]

    gdm_fixed = np.maximum(gdm, dry_green + dry_clover)
    total_fixed = np.maximum(dry_total, gdm_fixed + dry_dead)

    preds_df = pd.DataFrame({
        "image_path": test_unique["image_path"],
        "Dry_Green_g": dry_green,
        "Dry_Dead_g": dry_dead,
        "Dry_Clover_g": dry_clover,
        "GDM_g": gdm_fixed,
        "Dry_Total_g": total_fixed
    })

    preds_long = preds_df.melt(
        id_vars=["image_path"],
        value_vars=TARGET_ORDER,
        var_name="target_name",
        value_name="pred_target"
    )

    submission = test_long.merge(
        preds_long,
        on=["image_path", "target_name"],
        how="left"
    )

    out_df = submission[["sample_id", "pred_target"]].rename(columns={"pred_target": "target"})
    out_df["target"] = out_df["target"].fillna(0)
    out_df.to_csv("submission.csv", index=False)
    
    print(f"\n🚀 Submission Ensemble creata con successo!")
    print(out_df.head())

except Exception as e:
    print(f"⚠️ Errore critico: {e}")
    import traceback
    traceback.print_exc()