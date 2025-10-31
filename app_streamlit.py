import numpy as np
import pandas as pd
import streamlit as st
import joblib

model = joblib.load("models/xgb.joblib")
scaler = joblib.load("models/scaler.joblib")

st.title("💳 German Credit Risk Prediction App")
st.write("Predict whether a loan applicant has **Good Credit** or **Bad Credit**")

st.header("📋 Applicant Information")
duration = st.number_input("Duration (in months)", 6, 72, 24)
credit_amount = st.number_input("Credit Amount", 500, 50000, 3500)
installment_rate = st.number_input("Installment Rate (1–4)", 1, 4, 2)
age = st.number_input("Age", 18, 75, 35)
existing_credits = st.number_input("Existing Credits", 1, 4, 1)
num_dependents = st.number_input("Number of Dependents", 1, 2, 1)

checking_account_status = st.selectbox("Checking Account Status", ["A11","A12","A13","A14"])
savings_account_status = st.selectbox("Savings Account Status", ["A61","A62","A63","A64","A65"])
credit_history = st.selectbox("Credit History", ["A30","A31","A32","A33","A34"])
purpose = st.selectbox("Purpose", ["A40","A41","A42","A43","A44","A45","A46","A47","A48","A49"])
employment = st.selectbox("Employment Duration", ["A71","A72","A73","A74","A75"])
personal_status = st.selectbox("Personal Status", ["A91","A92","A93","A94"])
other_debtors = st.selectbox("Other Debtors / Guarantors", ["A101","A102","A103"])
property = st.selectbox("Property", ["A121","A122","A123","A124"])
other_installment_plans = st.selectbox("Other Installment Plans", ["A141","A142","A143"])
housing = st.selectbox("Housing", ["A151","A152","A153"])
job = st.selectbox("Job", ["A171","A172","A173","A174"])
telephone = st.selectbox("Telephone", ["A191","A192"])
foreign_worker = st.selectbox("Foreign Worker", ["A201","A202"])
present_residence_since = st.number_input("Present Residence Since (years)", 1, 4, 3)

input_data = pd.DataFrame([[
    checking_account_status, duration, credit_history, purpose, credit_amount,
    savings_account_status, employment, installment_rate, personal_status,
    other_debtors, present_residence_since, property, age, other_installment_plans,
    housing, existing_credits, job, num_dependents, telephone, foreign_worker
]], columns=[
    'Checking Account', 'Duration', 'Credit History', 'Purpose', 'Credit Amount',
    'Savings Account', 'Present Employment Since', 'Installment Rate', 'Personal Status and Sex',
    'Other Debtors', 'Present Residence Since', 'Property', 'Age', 'Other Installment Plans',
    'Housing', 'Existing Credits', 'Job', 'Liable Maintaince Provider', 'Telephone', 'Foreign_Worker'
])

if st.button("🔍 Predict Credit Risk"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    confidence = np.max(model.predict_proba(input_scaled)) * 100

    result = "✅ Good Credit" if prediction == 1 else "❌ Bad Credit"
    st.subheader(f"Prediction: {result}")
    st.write(f"Confidence: **{confidence:.2f}%**")
