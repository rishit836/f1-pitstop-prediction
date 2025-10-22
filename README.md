# F1 Pit Stop Prediction

Predicting if/when an F1 car will pit during a session using lap telemetry and session data. This project pulls data via FastF1, engineers lap-level features, balances the dataset, and trains a logistic regression model in PyTorch to classify whether a pit occurs on a given lap.


## Highlights

- Data ingestion from FastF1 (schedule, laps, weather, results)
- Lap-level dataset assembly for a selected driver (default: `VER`)
- Feature engineering from time and speed metrics
- Class balancing to reduce bias (undersampling majority class)
- Model training with PyTorch Logistic Regression
- Simple visualization of pit vs. speed for intuition


## Repository structure

```
.
├─ main.ipynb                # End-to-end notebook: data, features, model, evaluation
├─ requirements.txt          # Python dependencies
└─ data/
   ├─ laps/                  # Raw lap CSVs saved per event and session
   ├─ processed/             # Engineered datasets (e.g., lap_data.csv)
   ├─ result/                # Session results CSVs
   └─ weather/               # Session weather CSVs
```

> Note: The notebook creates the above directories as needed. Ensure you run the data acquisition cells before training.


## Getting started

### 1) Environment

- Python 3.10+ recommended
- Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

- Install dependencies:

```powershell
pip install -r requirements.txt
```

If you don’t have a complete `requirements.txt`, install the core libraries:

```powershell
pip install fastf1 torch pandas numpy scikit-learn matplotlib
```

### 2) FastF1 cache (recommended)

FastF1 benefits from a cache to speed up repeated downloads:

```python
import fastf1
fastf1.Cache.enable_cache("./fastf1_cache")  # run once per session/notebook
```

Add the above near the top of the notebook before calling any FastF1 API.


## Usage (from the notebook)

Open `main.ipynb` and run the cells in order. The notebook performs these steps:

1. Configure imports and (optionally) install missing packages.
2. Pull event schedule and define the driver-of-interest via `WIN_DRIVER` (default `"VER"`).
3. Fetch sessions per event, then save:
   - Laps → `data/laps/*.csv`
   - Weather → `data/weather/*.csv`
   - Results → `data/result/*.csv`
4. Engineer features and build a consolidated lap-level dataset → `data/processed/lap_data.csv`.
5. Balance the dataset (undersample non-pit laps to match pit laps).
6. Train a PyTorch logistic regression classifier.
7. Evaluate predictions and visualize simple relationships (e.g., pit vs. `SpeedST`).


## Data and features

- Input lap columns of interest (subset):
  - `LapNumber`, `SpeedI1`, `SpeedI2`, `SpeedFL`, `SpeedST`, `IsPersonalBest`, `Compound`, `TyreLife`, `FreshTyre`
  - Time-based columns transformed into “improvement” flags: `improve_LapTime`, `improve_Sector1Time`, `improve_Sector2Time`, `improve_Sector3Time`
  - Target label: `did_pit` (1 if `PitInTime` not null, else 0)

- Time handling
  - Raw time strings are converted to `datetime` and compared lap-to-lap to mark “improvement” (1 if next lap is faster; last lap defaults to 0).

- Scaling and encoding
  - Numeric speed channels are z-scored (mean 0, unit variance when possible).
  - Categorical columns (`FreshTyre`, `IsPersonalBest`, `Compound`) are intended to be encoded. See “Known issues” below for a note on one-hot handling.


## Model

- Architecture: Logistic Regression (single linear layer + sigmoid)
- Loss: Binary Cross-Entropy (BCE)
- Optimizer: Adam (lr=1e-3)
- Epochs: 10,000 (adjust as needed)
- Train/test split: 90/10
- Features used:
  - `['LapNumber', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'IsPersonalBest', 'Compound', 'TyreLife', 'FreshTyre', 'improve_LapTime', 'improve_Sector1Time', 'improve_Sector2Time', 'improve_Sector3Time']`

Evaluation prints prediction errors and a simple accuracy percentage. Consider using scikit-learn metrics (precision/recall/F1, ROC-AUC) for a fuller view.


## Example visualization

The notebook includes a quick scatter to show `did_pit` vs. `SpeedST` bucketed on the x-axis for a visual sense of separation.


## Known issues and gotchas

- Hard-coded driver
  - `WIN_DRIVER = "VER"` is hard-coded. Update this variable to train/evaluate on other drivers.

- Schedule/year mismatch
  - The notebook fetches schedule for 2024 but passes 2021 to `get_race_session_data(...)`. Ensure schedule and year align or document intent.

- Weather output path
  - Weather CSVs are saved under `models/weather/...` in one cell. This is likely a typo. Change to `data/weather/...` to be consistent with the repo layout.

- One-hot encoding shape
  - `OneHotEncoder` returns multiple columns; assigning the entire array back to a single column (e.g., `data[col] = enc.fit_transform(...).toarray()`) yields a column of arrays or unintended structure. Prefer:
    - `pd.get_dummies(data, columns=["FreshTyre", "IsPersonalBest", "Compound"], drop_first=True)`
    - Or, with `OneHotEncoder`, create a DataFrame from the array and join it back with proper column names.

- NaN handling
  - Speed column NaNs are filled with mean values later. Consider imputation earlier and document rationale, especially if mixing multiple sessions.

- Class balancing
  - Current approach undersamples negatives to match positives. This may discard signal. Consider class weighting or SMOTE as alternatives.


## Extending the project

- Add richer features (track status, stint transitions, compound sequences, gaps to cars ahead/behind, weather deltas).
- Try additional models: XGBoost/LightGBM, Logistic Regression (scikit-learn), calibrated classifiers.
- Perform time-aware splits (by session) to avoid leakage and better simulate race-day generalization.
- Hyperparameter tuning with cross-validation.
- Model interpretability: feature importance, permutation importance, SHAP.
- Calibrate probabilities and evaluate cost-sensitive metrics.


## Reproducibility checklist

- Enable FastF1 cache and run data-fetch cells first.
- Verify the event year(s) match the files you intend to generate.
- Confirm `data/processed/lap_data.csv` is created before training.
- Ensure consistent encoding and scaling between train and test subsets.


## Troubleshooting

- FastF1 requests are slow or fail: enable cache, try a different event, or re-run later; ensure internet connectivity.
- Time parsing errors: check that time strings are valid; ensure the `convert_timestring_to_timestamp` helper is applied before `to_datetime`.
- Shape errors after encoding: use `pd.get_dummies(...)` or rebuild the encoded columns as a DataFrame before concatenation.
- GPU not used: the current model is small and CPU is fine; for larger models, move tensors and model to CUDA if available.


## Acknowledgments

- Data access and APIs by [FastF1](https://theoehrly.github.io/Fast-F1/). Please review their documentation and usage guidelines.


## License

This repository has no explicit license file. Add a license (e.g., MIT) if you plan to publish or share.
