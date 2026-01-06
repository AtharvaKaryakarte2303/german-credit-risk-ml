import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Load saved model, scaler, and label encoders (dictionary)
model = joblib.load("models/xgb.joblib")
scaler = joblib.load("models/scaler.joblib")
label_encoders = joblib.load("models/LabelEncoders.joblib")  # <- dict of encoders

UI_LABELS = {
    "Duration": "Loan Tenure (Months)",
    "Credit Amount": "Loan Amount (₹)",
    "Installment Rate": "EMI Burden Category",
    "Age": "Applicant Age",
    "Existing Credits": "Active Loans (Current EMIs)",
    "Liable Maintaince Provider": "Number of Financial Dependents",

    "Checking Account": "Salary / Primary Bank Account",
    "Savings Account": "Savings & Bank Balance Behaviour",
    "Credit History": "Credit Bureau Status",
    "Purpose": "Loan Purpose",
    "Present Employment Since": "Employment Stability",
    "Personal Status and Sex": "Marital Status",
    "Other Debtors": "Co-applicant / Guarantor",
    "Property": "Asset Ownership",
    "Other Installment Plans": "Other Loan Plans",
    "Housing": "Residence Type",
    "Job": "Occupation Type",
    "Telephone": "Phone Availability",
    "Foreign_Worker": "Resident Status",
    "Present Residence Since": "Residence Stability (Years)"
}


st.title("💳 German Credit Risk Prediction App")
st.write("Predict whether a loan applicant has **Good Credit** or **Bad Credit**")

st.header("📋 Applicant Information")

duration = st.number_input(UI_LABELS["Duration"], 6, 72, 24)
credit_amount = st.number_input(UI_LABELS["Credit Amount"], 500, 50000, 3500)
installment_rate = st.number_input(UI_LABELS["Installment Rate"], 1, 4, 2)
age = st.number_input(UI_LABELS["Age"], 18, 75, 35)
existing_credits = st.number_input(UI_LABELS["Existing Credits"], 1, 4, 1)
num_dependents = st.number_input(UI_LABELS["Liable Maintaince Provider"], 1, 2, 1)

checking_account_status = st.selectbox(UI_LABELS["Checking Account"], ["A11", "A12", "A13", "A14"])
savings_account_status = st.selectbox(UI_LABELS["Savings Account"], ["A61", "A62", "A63", "A64", "A65"])
credit_history = st.selectbox(UI_LABELS["Credit History"], ["A30", "A31", "A32", "A33", "A34"])
purpose = st.selectbox(UI_LABELS["Purpose"], ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49"])
employment = st.selectbox(UI_LABELS["Present Employment Since"], ["A71", "A72", "A73", "A74", "A75"])
personal_status = st.selectbox(UI_LABELS["Personal Status and Sex"], ["A91", "A92", "A93", "A94"])
other_debtors = st.selectbox(UI_LABELS["Other Debtors"], ["A101", "A102", "A103"])
property = st.selectbox(UI_LABELS["Property"], ["A121", "A122", "A123", "A124"])
other_installment_plans = st.selectbox(UI_LABELS["Other Installment Plans"], ["A141", "A142", "A143"])
housing = st.selectbox(UI_LABELS["Housing"], ["A151", "A152", "A153"])
job = st.selectbox(UI_LABELS["Job"], ["A171", "A172", "A173", "A174"])
telephone = st.selectbox(UI_LABELS["Telephone"], ["A191", "A192"])
foreign_worker = st.selectbox(UI_LABELS["Foreign_Worker"], ["A201", "A202"])
present_residence_since = st.number_input(UI_LABELS["Present Residence Since"], 1, 4, 3)

# Create input dataframe
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
    # Encode categorical columns using the saved label encoders
    for col in input_data.columns:
        if col in label_encoders:  # only for categorical features
            encoder = label_encoders[col]
            val = input_data.at[0, col]
            if val in encoder.classes_:
                input_data.at[0, col] = encoder.transform([val])[0]
            else:
                # handle unseen category
                input_data.at[0, col] = -1

    # Scale numeric data
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction_proba = model.predict_proba(input_scaled)[0]
    prediction = model.predict(input_scaled)[0]

    good_prob = prediction_proba[1]
    bad_prob = prediction_proba[0]

    result = "✅ Good Credit" if prediction == 1 else "❌ Bad Credit"
    confidence = max(good_prob, bad_prob) * 100

    st.subheader(f"Prediction: {result}")
    st.write(f"**Good Credit Probability:** {good_prob:.2%}")
    st.write(f"**Bad Credit Probability:** {bad_prob:.2%}")
    st.write(f"**Confidence:** {confidence:.2f}%")
