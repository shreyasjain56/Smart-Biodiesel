# Smart Biodiesel — ML-Powered Biodiesel Yield Prediction

A Random Forest regression model that predicts FAME (Fatty Acid Methyl Ester) 
biodiesel yield from transesterification reaction parameters.

**Presented at MNIT Industry-Institute Conclave 2026**

## Live App
https://smart-biodiesel-je8ejrjxjkz29jrrfn9fva.streamlit.app/

## Problem
Optimising biodiesel yield through trial-and-error experimentation is 
time-consuming and resource-intensive. This model predicts yield from 4 
input parameters, enabling faster process optimisation.

## Input Parameters
- Temperature (°C)
- Methanol:Oil ratio
- Catalyst concentration (%)
- Reaction time (minutes)

## Model Performance
- Algorithm: Random Forest Regression (scikit-learn)
- R² Score: 0.89 on held-out test set
- Validation: Cross-validation + GridSearchCV hyperparameter tuning
- Key finding: Methanol:oil ratio identified as dominant yield predictor

## Tech Stack
Python · scikit-learn · pandas · NumPy · Matplotlib · Streamlit

## Repository Structure
- `biodiesel_dataset.csv` — training dataset
- `biodiesel_model.pkl` — trained model
- `phase4_streamlit_app.py` — Streamlit web application
- `requirements.txt` — dependencies

## Author
Shreyas Jain — B.Tech Chemical Technology, MNIT Jaipur (2024–2028)
