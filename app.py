# ----------------------------------------------------------
# app.py
# ----------------------------------------------------------
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Load model + metadata
model = joblib.load("disease_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
symptoms = joblib.load("symptom_columns.pkl")  # list of symptom names (lowercased)

# Try loading optional files (if present) for extra info
desc_df = None
prec_df = None
if Path("symptom_Description.csv").exists():
    try:
        desc_df = pd.read_csv("symptom_Description.csv", encoding="latin1")
        desc_df.columns = [c.strip() for c in desc_df.columns]
    except Exception:
        desc_df = None

if Path("symptom_precaution.csv").exists():
    try:
        prec_df = pd.read_csv("symptom_precaution.csv", encoding="latin1")
        prec_df.columns = [c.strip() for c in prec_df.columns]
    except Exception:
        prec_df = None

st.title("🩺 Disease Prediction from Symptoms")
st.write("Select your symptoms and click **Predict**.")

# Make labels a bit nicer for UI (title case)
nice_to_raw = {s.title().replace("_", " "): s for s in symptoms}
choices = sorted(nice_to_raw.keys())

selected = st.multiselect("Choose symptoms", choices)

if st.button("Predict Disease"):
    x = np.zeros((1, len(symptoms)), dtype=np.uint8)
    for nice in selected:
        raw = nice_to_raw[nice]  # back to lowercase/raw
        idx = symptoms.index(raw)
        x[0, idx] = 1

    pred_idx = model.predict(x)[0]
    disease = label_encoder.inverse_transform([pred_idx])[0]
    prob = model.predict_proba(x)[0].max() * 100

    st.success(f"**Predicted Disease:** {disease}")
    st.caption(f"Confidence: {prob:.2f}%")

    # Extra info if CSVs exist
    if desc_df is not None:
        # look for columns like 'Disease' and 'Description'
        dcol = next((c for c in desc_df.columns if c.lower().startswith("disease")), None)
        ccol = next((c for c in desc_df.columns if "description" in c.lower()), None)
        if dcol and ccol:
            row = desc_df[desc_df[dcol].astype(str).str.strip().str.lower() ==
                          disease.strip().lower()]
            if not row.empty:
                st.markdown("**Description**")
                st.write(row.iloc[0][ccol])

    if prec_df is not None:
        dcol = next((c for c in prec_df.columns if c.lower().startswith("disease")), None)
        pres = [c for c in prec_df.columns if "precaution" in c.lower()]
        if dcol and pres:
            row = prec_df[prec_df[dcol].astype(str).str.strip().str.lower() ==
                          disease.strip().lower()]
            if not row.empty:
                st.markdown("**Recommended Precautions**")
                items = [str(row.iloc[0][c]) for c in pres if pd.notna(row.iloc[0][c])]
                for it in items:
                    st.write("•", it)
