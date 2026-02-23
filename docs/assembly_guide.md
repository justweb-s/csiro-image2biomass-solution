# Assembly guide (winner zip)

This file explains how to assemble the final **single ZIP archive** required by the competition (training code + trained models + inference notebook).

In this repository, documentation is kept under `docs/` for cleanliness.
When building the competition ZIP, copy the required documentation files to the ZIP root using the expected filenames.

## 1) Required files at the top level
Ensure the ZIP contains these files at the top level:
- `README.md`
- `requirements.txt` (exact pinned versions; replace placeholder with your real `pip freeze`)
- `SETTINGS.json` (single source of I/O paths; create it from `SETTINGS.example.json`)
- `entry_points.md` (copy from `docs/entry_points.md`)
- `directory_structure.txt` (copy from `docs/directory_structure.txt`)
- Training scripts:
  - `modelA_and_specialist_train.py`
  - `modelB_and_specialist_train.py`
  - `model_high_specialist_train.py`
- `inference.py` (reference; the official inference notebook must also be included in the ZIP as required)

## 2) Folder structure (expected)
Create/populate these folders (already present as empty placeholders in the repo):
- `data/raw/` (Kaggle competition data extracted here)
- `data/processed/` (intermediate artifacts like `train_wide_*.csv`)
- `models/` (ALL trained weights used for the final solution)
- `backbone/dinov3-vit7b-backbone/` (local Hugging Face snapshot of the backbone)
- `artifacts/` (optional staging dirs; Kaggle upload/download helpers write here)
- `logs/`
- `submissions/`

## 3) Where to copy the trained model weights
The exact weight filenames are taken from `inference.py`.

Copy the following files into `models/` preserving the subfolders (recommended):

- `models/dinov3-swa-full-fit-768px/dinov3_full_SWA.pth`
- `models/dinov3-mse-768px/dinov3_full_SWA.pth`
- `models/dinov3-high-specialist-768px/dinov3_high_specialist.pth`
- `models/dinov3-high-specialist-swa/dinov3_high_specialist_swa.pth`
- `models/dinov3-mse-high-specialist-768px/dinov3_high_specialist.pth`

If you have additional checkpoints that are needed to reproduce the final score, keep them under:
- `models/checkpoints/`

## 4) Backbone: include vs download
For **maximum reproducibility / conformity**, include the backbone snapshot in the ZIP.

### Recommended (offline, self-contained)
Populate `backbone/dinov3-vit7b-backbone/` with the local Hugging Face snapshot of:
- `facebook/dinov3-vit7b16-pretrain-lvd1689m`

The training scripts prefer `BACKBONE_DIR` if the folder exists.

### Alternative (requires internet)
If you do NOT include the backbone, set in `SETTINGS.json`:
- `BACKBONE_DIR` to a non-existing path, and keep `BACKBONE_NAME_OR_PATH` as the HF model id.

## 5) Data placement
Extract the Kaggle competition dataset so that `RAW_DATA_DIR` contains:
- `train.csv`
- `test.csv`
- `train/`
- `test/`

Default `SETTINGS.json` expects:
- `data/raw/train.csv`
- `data/raw/test.csv`
- `data/raw/train/`
- `data/raw/test/`

## 6) Running training (entry points)
See `entry_points.md`.

Important notes:
- Colab/Kaggle helper actions (pip installs, Kaggle download/upload) are **disabled by default** behind flags.
- Standard runs should rely only on the local folders defined in `SETTINGS.json`.

## 7) Final ZIP creation checklist
- [ ] Replace placeholder `requirements.txt` with exact pinned versions (`pip freeze` from the training environment)
- [ ] Copy all model weights into `models/` (paths above)
- [ ] Include the inference notebook (as required by the competition)
- [ ] Verify `SETTINGS.json` paths are correct for the target machine
- [ ] Re-generate `directory_structure.txt` after adding weights/backbone
