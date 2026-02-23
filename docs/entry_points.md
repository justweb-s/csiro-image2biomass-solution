# Entry points

All scripts must use paths from `SETTINGS.json`. Create it by copying `SETTINGS.example.json` and adjusting paths.

## 1) (Optional) Prepare data
If you want to materialize intermediate files (e.g. train-wide pivot), you can run the training scripts directly; they already perform the required preprocessing.

## 2) Train models
### Base 1 (SWA) + High specialist
```bash
python modelA_and_specialist_train.py
```

#### Smoke test (fast)
```bash
SMOKE_TEST=1 SMOKE_IMG_SIZE=224 SMOKE_MAX_SAMPLES=64 python modelA_and_specialist_train.py
```

### Base 2 (MSE SWA) + High specialist (MSE)
```bash
python modelB_and_specialist_train.py
```

#### Smoke test (fast)
```bash
SMOKE_TEST=1 SMOKE_IMG_SIZE=224 SMOKE_MAX_SAMPLES=64 python modelB_and_specialist_train.py
```

### Nuclear specialist + SWA utility
```bash
python model_high_specialist_train.py
```

#### Smoke test (fast)
```bash
SMOKE_TEST=1 SMOKE_IMG_SIZE=224 SMOKE_MAX_SAMPLES=64 python model_high_specialist_train.py
```

## 3) Predict / generate submission
The official inference notebook is provided separately (per competition requirement).
A reference script is available as `inference.py`.
