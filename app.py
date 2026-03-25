import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Load model
model = load_model("model/heart_failure_model.keras")

st.title("❤️ Heart Failure Prediction App")

st.write("Enter patient details to predict risk")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=60)
anaemia = st.selectbox("Anaemia (0 = No, 1 = Yes)", [0, 1])
creatinine_phosphokinase = st.number_input("CPK Level", value=200)
diabetes = st.selectbox("Diabetes (0 = No, 1 = Yes)", [0, 1])
ejection_fraction = st.number_input("Ejection Fraction", value=20)
high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1])
platelets = st.number_input("Platelets", value=250000)
serum_creatinine = st.number_input("Serum Creatinine", value=1.0)
serum_sodium = st.number_input("Serum Sodium", value=130)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
smoking = st.selectbox("Smoking (0 = No, 1 = Yes)", [0, 1])
time = st.number_input("Follow-up Time", value=4)

# Prediction button
if st.button("Predict"):

    input_data = pd.DataFrame([[
        age, anaemia, creatinine_phosphokinase, diabetes,
        ejection_fraction, high_blood_pressure, platelets,
        serum_creatinine, serum_sodium, sex, smoking, time
    ]], columns=[
        'age','anaemia','creatinine_phosphokinase','diabetes',
        'ejection_fraction','high_blood_pressure','platelets',
        'serum_creatinine','serum_sodium','sex','smoking','time'
    ])

    prediction = model.predict(input_data)

    if prediction > 0.5:
        st.error("⚠️ High Risk of Heart Failure")
    else:
        st.success("✅ Low Risk")
