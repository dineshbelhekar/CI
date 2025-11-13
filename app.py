# =======================================
# 🩺 NephroPredict: CKD Prediction Web App
# =======================================

import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="🩺 NephroPredict", layout="centered", page_icon="💊")

# -------------------------------
# Custom Styling (CSS)
# -------------------------------
st.markdown("""
    <style>
        .main {
            background-color: #f2f7ff;
            padding: 25px;
            border-radius: 12px;
        }
        h1, h2, h3 {
            color: #003366;
            text-align: center;
        }
        .stButton>button {
            background-color: #0066cc;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 100%;
            font-size: 18px;
        }
        .stButton>button:hover {
            background-color: #004c99;
        }
        .stSuccess {
            background-color: #d9fdd3;
        }
        .stError {
            background-color: #ffe5e5;
        }
        .footer {
            text-align: center;
            color: #666;
            margin-top: 50px;
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model and Preprocessing Tools
# -------------------------------
model = joblib.load("nephropredict_best_model.pkl")
scaler = joblib.load("nephropredict_scaler.pkl")
feature_list = joblib.load("nephropredict_feature_list.pkl")

# -------------------------------
# Page Header
# -------------------------------
st.title("🩺 NephroPredict")
st.markdown("### 🌿 Early Detection of Chronic Kidney Disease using Machine Learning")
st.write("### Enter your health details below to predict the likelihood of CKD.")

st.markdown("---")

# -------------------------------
# Input Section
# -------------------------------
st.subheader("📋 Patient Health Information")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=45)
    bp = st.number_input("Blood Pressure (mmHg)", min_value=0, value=80)
    sg = st.number_input("Specific Gravity", min_value=1.0, max_value=1.03, value=1.02)
    al = st.number_input("Albumin", min_value=0, max_value=5, value=1)
    su = st.number_input("Sugar", min_value=0, max_value=5, value=0)
    htn = st.selectbox("Hypertension", ["yes", "no"])

with col2:
    bgr = st.number_input("Blood Glucose Random (mg/dl)", min_value=0, value=120)
    bu = st.number_input("Blood Urea (mg/dl)", min_value=0, value=40)
    sc = st.number_input("Serum Creatinine (mg/dl)", min_value=0.0, value=1.2)
    sod = st.number_input("Sodium (mEq/L)", min_value=0.0, value=135.0)
    pot = st.number_input("Potassium (mEq/L)", min_value=0.0, value=4.5)
    hemo = st.number_input("Hemoglobin (g/dl)", min_value=0.0, value=14.0)
    dm = st.selectbox("Diabetes Mellitus", ["yes", "no"])
    appet = st.selectbox("Appetite", ["good", "poor"])

st.markdown("---")

# -------------------------------
# Predict Button
# -------------------------------
if st.button("🔍 Predict CKD Status"):
    try:
        # Prepare input data
        input_dict = {
            "age": age, "bp": bp, "sg": sg, "al": al, "su": su, "bgr": bgr,
            "bu": bu, "sc": sc, "sod": sod, "pot": pot, "hemo": hemo,
            "htn": 1 if htn == "yes" else 0,
            "dm": 1 if dm == "yes" else 0,
            "appet": 1 if appet == "good" else 0
        }

        input_df = pd.DataFrame([input_dict])

        # Add missing columns if needed
        for col in feature_list:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns
        input_df = input_df[feature_list]

        # Scale & Predict
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]

        # Display result
        st.markdown("---")
        st.subheader("📊 Prediction Result")

        if pred == 1:
            st.error("🩸 **Chronic Kidney Disease Detected**")
            st.warning("⚠️ Please consult a nephrologist for further diagnosis and management.")
        else:
            st.success("✅ **No CKD Detected**")
            st.balloons()
            st.info("Your kidney parameters appear healthy. Keep maintaining a balanced lifestyle!")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown('<p class="footer">© 2025 NephroPredict | Developed for Early CKD Detection</p>', unsafe_allow_html=True)
