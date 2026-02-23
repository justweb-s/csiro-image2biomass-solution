# CSIRO - Image2Biomass Prediction (Winning Solution)

This archive contains the code and trained model weights required to reproduce our final solution for the **CSIRO - Image2Biomass Prediction** competition.

## Hardware used (original training)
- **GPU**: Google Colab A100 40GB
- **CPU/RAM**: (fill in)

## Hardware used (inference)
- **GPU**: Kaggle 2x NVIDIA T4 (as used for the final submission notebook)

## Software
- **OS**: Google Colab (Linux)
- **Python**: 3.12.12
- **CUDA/cuDNN/Driver**: NVIDIA driver 550.54.15 (CUDA 12.4 reported by `nvidia-smi`); PyTorch build: CUDA 12.1 (`torch 2.3.0+cu121`)
- **Python packages**: pinned versions are in `requirements.txt`

## Archive contents
- `modelA_and_specialist_train.py`: base model training (SWA) + high-biomass specialist training
- `modelB_and_specialist_train.py`: MSE base model training (SWA) + MSE high-biomass specialist training
- `model_high_specialist_train.py`: nuclear specialist training + SWA utility
- `inference.py`: reference inference script (the official inference notebook is provided separately per competition requirements)
- `SETTINGS.json`: the only place where input/output paths are configured
- `entry_points.md`: commands to reproduce training and produce predictions
- `models/`: trained model weights used in the final ensemble (must be present in this archive)
- `backbone/`: local Hugging Face snapshot for the DINOv3 backbone (optional but recommended for offline reproducibility)

## Data setup
- Download the competition dataset via Kaggle.
- Extract it to the folder specified by `RAW_DATA_DIR` in `SETTINGS.json`.
- The expected structure is:
  - `<RAW_DATA_DIR>/train.csv`
  - `<RAW_DATA_DIR>/test.csv`
  - `<RAW_DATA_DIR>/train/` (images)
  - `<RAW_DATA_DIR>/test/` (images)

## Colab smoke test (zip workflow)
Follow these steps from a fresh Colab session.
 
### Essential commands
Prerequisites:
- Upload the solution archive to Colab as: `/content/final-solution.zip`
- Upload your Kaggle token to Colab as: `/content/kaggle.json`
 
Setup:
```bash
!rm -rf /content/solution && mkdir -p /content/solution
!unzip -q /content/final-solution.zip -d /content/solution
!bash /content/solution/colab_quickstart.sh setup
```
 
Smoke run:
```bash
%env SMOKE_IMG_SIZE=224
%env SMOKE_MAX_SAMPLES=64
!bash /content/solution/colab_quickstart.sh smoke
```
 
## Environment specs and correct requirements
Use these commands in the same environment where you run the scripts (after `setup`):
 
```bash
!python -V
!nvidia-smi
!python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
!pip freeze > /content/requirements_freeze.txt
```

Use `/content/requirements_freeze.txt` as an audit/debug reference and copy the reported versions into the **Software** section above. The shipped `requirements.txt` is a minimal pinned set for the packages actually used by the scripts.

## Training
See `entry_points.md`.

## Model weights
The exact weight filenames used by the final ensemble are defined in `inference.py` and expected under `MODEL_DIR` as configured in `SETTINGS.json`.

## Important side effects / assumptions
- Training scripts will create output folders (model checkpoints, logs) as configured in `SETTINGS.json`.
- The original Colab/Kaggle helper commands (pip installs, Kaggle uploads/downloads) are preserved but disabled by default behind boolean flags.
