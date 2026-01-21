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

employment_ui = {
    "Unemployed / < 1 year": "A71",
    "1 – 3 years": "A72",
    "3 – 5 years": "A73",
    "5 – 10 years": "A74",
    "10+ years": "A75"
}

credit_history_ui = {
    "New to credit / No previous loans": "A30",
    "All loans paid on time": "A31",
    "Existing loans, regular repayment": "A32",
    "Past delays, but settled": "A33",
    "Serious past defaults": "A34"
}

purpose_ui = {
    "Two-wheeler / Personal use": "A40",
    "Four-wheeler / Vehicle purchase": "A41",
    "Education loan": "A42",
    "Business expansion / Working capital": "A43",
    "Home improvement / Renovation": "A44",
    "Consumer durables (TV, fridge, etc.)": "A45",
    "Medical expenses": "A46",
    "Travel / Lifestyle": "A47",
    "Agriculture / Allied activity": "A48",
    "Other personal needs": "A49"
}

savings_ui = {
    "No savings / Low balance": "A61",
    "< ₹50k balance": "A62",
    "₹50k – ₹2L balance": "A63",
    "> ₹2L balance": "A64",
    "Unknown / Not reported": "A65"
}

checking_ui = {
    "No active salary account": "A11",
    "Low balance / irregular credits": "A12",
    "Regular salary credits": "A13",
    "High balance & strong banking history": "A14"
}

job_ui = {
    "Salaried (Private / Govt)": "A171",
    "Self-Employed Professional": "A172",
    "Business Owner / MSME": "A173",
    "Daily Wage / Gig / Contract": "A174"
}

housing_ui = {
    "Own House / Flat": "A151",
    "Rented (Registered / Informal)": "A152",
    "Company / Family Provided": "A153"
}

property_ui = {
    "Real Estate (House / Land)": "A121",
    "Vehicle / Gold / Fixed Assets": "A122",
    "Savings / Investments / Insurance": "A123",
    "No Major Assets": "A124"
}

debtor_ui = {
    "No Co-applicant / Guarantor": "A101",
    "Co-applicant Present": "A102",
    "Guarantor Present": "A103"
}

installment_ui = {
    "Low EMI Burden": 1,
    "Moderate EMI Burden": 2,
    "High EMI Burden": 3,
    "Very High EMI Burden": 4
}

personal_ui = {
    "Single": "A91",
    "Married": "A92",
    "Divorced / Separated": "A93",
    "Widowed": "A94"
}

installment_ui = {
    "None": "A141",
    "Bank / NBFC Loans": "A142",
    "Informal / Employer / Others": "A143"
}

telephone_ui = {
    "No": "A191",
    "Yes (Mobile / Landline)": "A192"
}

resident_ui = {
    "Indian Resident": "A201",
    "Non-Resident (NRI / Foreign)": "A202"
}

st.set_page_config(
    page_title="Bharat Credit Risk Assessment System",
    page_icon="🇮🇳",
    layout="wide"
)

st.title("🇮🇳 Bharat Credit Risk Assessment System")
st.caption(
    "RBI-aligned credit risk evaluation demo for Indian NBFCs, FinTechs, and lending institutions"
)

st.warning(
    "⚠️ This is a prototype system for educational and demonstration purposes only. "
    "It is not an official credit decision engine and should not be used for live lending decisions."
)

st.write("Predict whether a loan applicant has **Good Credit** or **Bad Credit**")

st.header("📋 Applicant Information")

duration = st.number_input(UI_LABELS["Duration"], 6, 72, 24)
credit_amount = st.number_input(UI_LABELS["Credit Amount"], 500, 50000, 3500)
installment_label = st.selectbox(
    "EMI Burden Category",
    list(installment_ui.keys())
)
installment_rate = installment_ui[installment_label]
age = st.number_input(UI_LABELS["Age"], 18, 75, 35)
existing_credits = st.number_input(UI_LABELS["Existing Credits"], 0, 5, 1)
num_dependents = st.number_input(UI_LABELS["Liable Maintaince Provider"], 0, 5, 1)

checking_label = st.selectbox(
    UI_LABELS["Checking Account"],
    list(checking_ui.keys())
)

checking_account_status = checking_ui[checking_label]
savings_label = st.selectbox(
    UI_LABELS["Savings Account"],
    list(savings_ui.keys())
)

savings_account_status = savings_ui[savings_label]
credit_history_label = st.selectbox(
    UI_LABELS["Credit History"],
    list(credit_history_ui.keys())
)
credit_history = credit_history_ui[credit_history_label]
purpose_label = st.selectbox(
    UI_LABELS["Purpose"],
    list(purpose_ui.keys())
)

purpose = purpose_ui[purpose_label]
employment_label = st.selectbox(
    "Employment Stability",
    list(employment_ui.keys())
)

employment = employment_ui[employment_label]
ersonal_label = st.selectbox(
    UI_LABELS["Personal Status and Sex"],
    list(personal_ui.keys())
)

personal_status = personal_ui[personal_label]
other_debtors = debtor_ui[
    st.selectbox(UI_LABELS["Other Debtors"], debtor_ui.keys())
]
other_debtors = other_debtors_ui[other_debtors_label]
property_label = st.selectbox(
    UI_LABELS["Property"],
    list(property_ui.keys())
)

property = property_ui[property_label]
other_installment_plans_label = installment_ui[
    st.selectbox(UI_LABELS["Other Installment Plans"], installment_ui.keys())
]
other_installment_plans = installment_ui[other_installment_plans_label]

housing_label = st.selectbox(
    UI_LABELS["Housing"],
    list(housing_ui.keys())
)

housing = housing_ui[housing_label]
job_label = st.selectbox(
    UI_LABELS["Job"],
    list(job_ui.keys())
)

job = job_ui[job_label]
telephoneLabel = telephone_ui[
    st.selectbox(UI_LABELS["Telephone"], telephone_ui.keys())
]
telephone = telephone_ui[telephoneLabel]
foreign_worker_label = resident_ui[
    st.selectbox("Resident Status", resident_ui.keys())
]
foreign_worker = resident_ui[foreign_worker_label]
present_residence_since = st.selectbox(
    "Years at Current Residence",
    ["< 1 year", "1 – 3 years", "3 – 5 years", "5+ years"]
)

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

    st.subheader("📊 Credit Risk Evaluation Result")
    st.write(f"**Good Credit Probability:** {good_prob:.2%}")
    st.write(f"**Bad Credit Probability:** {bad_prob:.2%}")
    st.write(f"**Confidence:** {confidence:.2f}%")
