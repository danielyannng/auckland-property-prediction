# Auckland Properties Price Prediction

A Streamlit app for estimating Auckland property prices. Provide bedrooms, bathrooms, property type, and suburb, and the app combines suburb-level reference stats with a trained KNN model to predict the price.

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
└── README.md             # This document
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
