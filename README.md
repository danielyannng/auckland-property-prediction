# Auckland Properties Price Prediction

A Streamlit app for estimating Auckland property prices. Provide bedrooms, bathrooms, property type, and suburb, and the app combines suburb-level reference stats with a trained KNN model to predict the price.

Chinese version: see `README_zh.md`.

## Features
- Interactive inputs: Bedrooms, Bathrooms, Property Type, Suburb
- Auto-load suburb reference data (average score and average price)
- Prediction using a trained KNN Regressor (k=4)

## Project structure
```
Submission 2/
├── app.py                # Streamlit app entry
├── knn_best_model.pkl    # Trained KNN regressor
├── scaler.pkl            # Feature scaler used during training (scikit-learn)
├── suburb_ref.csv        # Suburb reference data (expects Suburb, Score_avg, avg_price)
├── all_data.csv          # Source/analysis data (not read by the app directly)
├── report.ipynb          # Analysis/Modelling notebook
├── requirements.txt      # Python dependencies
├── README.md             # This document (EN)
└── README_zh.md          # Chinese documentation
```

## Requirements
- Python 3.9+ (recommended 3.9–3.11)
- See `requirements.txt`

## Quickstart
1) Install dependencies
   - Optionally create a venv and install:
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.txt
     ```
2) Run the app
   ```bash
   streamlit run app.py
   ```
3) Open the URL printed in the terminal, fill the form, and click “Predict Property Price”.

Note: Ensure `knn_best_model.pkl`, `scaler.pkl`, and `suburb_ref.csv` sit in the same directory as `app.py`.

## Inputs and features
- Bedrooms (integer 1–10)
- Bathrooms (integer 1–10)
- Property Type (House/Apartment/Townhouse; internally mapped to 1/2/3)
- Suburb (options loaded from `suburb_ref.csv`)

From `suburb_ref.csv` the app uses:
- `Suburb`: suburb name (for selection)
- `Score_avg`: average score for the suburb (relative reference)
- `avg_price`: average price for the suburb (numeric)

Final model features: Bedrooms, Bathrooms, Score_avg, avg_price, PropertyTypeID.

## Model
- Regressor: KNN (k=4)
- Preprocessing: `StandardScaler` consistent with training (loaded from `scaler.pkl`)

## How it works
1) You select inputs in the UI. The app looks up suburb-level features (`Score_avg`, `avg_price`) from `suburb_ref.csv`.
2) The property type is mapped to an integer ID: House=1, Apartment=2, Townhouse=3.
3) A feature vector is built: `[Bedrooms, Bathrooms, Score_avg, avg_price, PropertyTypeID]`.
4) Features are transformed with the saved scaler from `scaler.pkl`.
5) The trained KNN model in `knn_best_model.pkl` predicts the price.

Caching: The app uses Streamlit caching (`st.cache_resource` for model/scaler, `st.cache_data` for suburb data) to reduce load times. If you change files, choose “Rerun” in Streamlit to refresh.

## Required data files
- `suburb_ref.csv` must include at least the following columns:
   - `Suburb` (string): suburb name shown in the dropdown
   - `Score_avg` (float): the suburb’s average score (relative metric)
   - `avg_price` (float): the suburb’s average price

Format notes:
- CSV should be UTF-8 encoded.
- Column names must match exactly (case sensitive).
- `Score_avg` and `avg_price` should be numeric (no currency symbols); if needed, clean the data before use.

## Development
- Python: 3.9–3.11 recommended. Create a virtual environment to avoid version conflicts.
- Live reload: Streamlit auto-reloads when you save `app.py`.
- Linting/formatting: optional, e.g., `ruff` and `black`.
- Updating the model: if you retrain a model, keep the feature order and preprocessing consistent, and overwrite `knn_best_model.pkl` and `scaler.pkl` together.

## Deployment
Minimal options:
- Streamlit Community Cloud: push this folder to a repo, set the app entry to `app.py`, and ensure `requirements.txt` is present. Upload `knn_best_model.pkl`, `scaler.pkl`, and `suburb_ref.csv` to the repo.
- Any server/VPS: install dependencies from `requirements.txt` and run `streamlit run app.py`. For public access, run behind a reverse proxy (e.g., Nginx) and set the correct port with Streamlit config or CLI flags.
