# CSIRO - Image2Biomass Prediction — Competition Solution

This repository contains the code for my final solution to the **CSIRO - Image2Biomass Prediction** Kaggle competition.

Results:
- **Public leaderboard**: 1st
- **Private leaderboard**: 38th

The repository is structured to be both:
- portfolio-friendly (clean, readable, English-only docs)
- competition-ready (reproducibility notes and packaging instructions)

## Repository contents
- `modelA_and_specialist_train.py`: base model training + high-biomass specialist training
- `modelB_and_specialist_train.py`: alternative base model training (MSE) + MSE high-biomass specialist training
- `model_high_specialist_train.py`: nuclear specialist training + SWA utility
- `inference.py`: reference inference / ensembling script (Kaggle-oriented)
- `requirements.txt`: pinned minimal requirements used by the training scripts
- `SETTINGS.example.json`: configuration template (paths)
- `docs/`: competition/reproducibility documentation

Notes:
- Trained weights are **not** committed to this repository. The expected filenames are referenced in `inference.py`.
- Local configuration is intentionally not committed: create your own `SETTINGS.json` from the template below.

## Quick start
1) Create `SETTINGS.json`

Copy the template and adjust paths as needed:

```bash
cp SETTINGS.example.json SETTINGS.json
```

On Windows PowerShell:

```powershell
Copy-Item SETTINGS.example.json SETTINGS.json
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) Data layout

Download the competition dataset via Kaggle and extract it to the folder specified by `RAW_DATA_DIR` in `SETTINGS.json`.
Expected structure:
- `<RAW_DATA_DIR>/train.csv`
- `<RAW_DATA_DIR>/test.csv`
- `<RAW_DATA_DIR>/train/`
- `<RAW_DATA_DIR>/test/`

## Training
See `docs/entry_points.md`.

## Competition packaging / reproducibility
See:
- `docs/assembly_guide.md`
- `docs/directory_structure.txt`
- `docs/requirements_freeze.txt`

## Colab smoke test (ZIP workflow)
If you package a single ZIP archive for Colab, the helper script is available as `colab_quickstart.sh`.
