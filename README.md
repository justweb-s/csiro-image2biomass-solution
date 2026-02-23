# CSIRO - Image2Biomass Prediction — Competition Solution
 
Codebase for my final solution to the **CSIRO - Image2Biomass Prediction** Kaggle competition.
 
**Results**
- **Public leaderboard**: 1st
- **Private leaderboard**: 38th
 
## Why this repo matters (portfolio view)
This project showcases end-to-end applied ML work under real competition constraints:
- model architecture adaptation and parameter-efficient fine-tuning (LoRA / PEFT)
- reproducible training pipeline (config-driven I/O via `SETTINGS.json`)
- ensembling + specialist modeling for tail performance
- pragmatic engineering decisions for GPU memory and runtime
 
## Solution overview (high level)
- **Backbone**: DINOv3 ViT-7B
- **Input**: three-stream crop strategy (left / center / right)
- **Heads**: regression head over 5 biomass targets
- **Training**: EMA model + SWA computed by averaging the last epochs (snapshots)
- **Ensemble**: base models + specialist models + conditional merge rules
 
## Compute / runtime note
Full training is **not** meant to run locally.
 
- **Recommended environment**: Google Colab
- **GPU**: 1x A100 40GB
- **Wall time**: ~**20 hours** end-to-end for the full training pipeline
 
For quick validation, use the **smoke tests** (reduced image size + fewer samples).
 
## Repository contents
- `modelA_and_specialist_train.py`: base model training + high-biomass specialist training
- `modelB_and_specialist_train.py`: alternative base model training (MSE) + MSE high-biomass specialist training
- `model_high_specialist_train.py`: nuclear specialist training + SWA utility
- `inference.py`: reference inference / ensembling script (Kaggle-oriented)
- `requirements.txt`: pinned minimal requirements for the training scripts
- `SETTINGS.example.json`: configuration template (paths)
- `docs/`: reproducibility + competition packaging notes
 
Notes:
- Trained weights are **not** committed to this repository.
- `SETTINGS.json` is intentionally **not** committed. Create it from the template.
 
## Quick start (recommended on Colab)
In a fresh Colab notebook:
 
```bash
git clone https://github.com/justweb-s/csiro-image2biomass-solution.git
cd csiro-image2biomass-solution
cp SETTINGS.example.json SETTINGS.json
pip install -r requirements.txt
```
 
Then edit `SETTINGS.json` to point `RAW_DATA_DIR` to the extracted Kaggle dataset.
 
### Smoke test (fast sanity check)
```bash
SMOKE_TEST=1 SMOKE_IMG_SIZE=224 SMOKE_MAX_SAMPLES=64 python modelA_and_specialist_train.py
SMOKE_TEST=1 SMOKE_IMG_SIZE=224 SMOKE_MAX_SAMPLES=64 python modelB_and_specialist_train.py
SMOKE_TEST=1 SMOKE_IMG_SIZE=224 SMOKE_MAX_SAMPLES=64 python model_high_specialist_train.py
```
 
## Full training
Entry points and commands are listed in:
- `docs/entry_points.md`
 
## Competition packaging / reproducibility
See:
- `docs/assembly_guide.md`
- `docs/directory_structure.txt`
- `docs/requirements_freeze.txt`
