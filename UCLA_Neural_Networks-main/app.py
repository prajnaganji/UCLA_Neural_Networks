import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ─── Load scaler & model once ────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    here = os.getcwd()
    with open(os.path.join(here, "ucla_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(here, "ucla_mlp_model.pkl"), "rb") as f:
        model = pickle.load(f)
    return scaler, model

scaler, model = load_artifacts()

# Grab the exact feature order the scaler was fitted on
feature_names = list(scaler.feature_names_in_)

st.title("🎓 UCLA Admission Predictor")
st.markdown("Enter the applicant’s details below and click **Predict Admission**.")

# ─── Collect raw inputs ──────────────────────────────────────────────────
gre       = st.number_input("GRE Score",       min_value=0,   max_value=340, value=320, step=1)
toefl     = st.number_input("TOEFL Score",     min_value=0,   max_value=120, value=105, step=1)
unirating = st.selectbox("University Rating", [1, 2, 3, 4, 5])
sop       = st.slider("SOP Strength (1–5)",       1, 5, 3)
lor       = st.slider("LOR Strength (1–5)",       1, 5, 3)
cgpa      = st.number_input("CGPA (0.0–10.0)", min_value=0.0, max_value=10.0, value=8.5, step=0.1, format="%.1f")
research  = st.selectbox("Research Experience", ["Yes", "No"])

if st.button("Predict Admission"):
    
    row = dict.fromkeys(feature_names, 0)

    # ─── Fill in the 7 raw values
    row["GRE Score"]        = gre
    row["TOEFL Score"]      = toefl
    row["SOP"]              = sop
    row["LOR "]             = lor       
    row["CGPA"]             = cgpa

    for i in [1, 2, 3, 4, 5]:
        key = f"University_Rating_{i}"
        row[key] = 1 if unirating == i else 0

    # ─── One-hot encode Research
    row["Research_1"] = 1 if research == "Yes" else 0
    row["Research_0"] = 1 if research == "No"  else 0

    # ─── Create a DataFrame in the exact column order
    X_df = pd.DataFrame([row], columns=feature_names)

    # ─── Scale & predict
    X_scaled = scaler.transform(X_df)
    prob = model.predict_proba(X_scaled)[0, 1]
    pred = model.predict(X_scaled)[0]

    # ─── Display results
    st.write("**Admit?**", "Yes" if pred == 1 else "No")
    st.write(f"**Chance of Admission:** {prob:.1%}")

    # ─── Bar chart of admit vs reject
    st.bar_chart({
        "Admit Probability": [prob],
        "Reject Probability": [1 - prob]
    })
