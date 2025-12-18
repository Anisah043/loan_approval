import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_models():
    model = joblib.load("loan_status_predictor.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

def predict_loan_status(applicant_data, model, preprocessor):
    processed_data = preprocessor.transform(applicant_data)
    prediction = model.predict(processed_data)
    probabilities = model.predict_proba(processed_data)
    
    return {
        'status': prediction[0],
        'probability_approved': probabilities[0][1],
        'probability_rejected': probabilities[0][0]
    }

st.title("💰 Loan Approval Predictor")
st.markdown("---")

try:
    model, preprocessor = load_models()
except:
    st.error("Error to load the model.")
    st.stop()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Borrower informations")
    person_age = st.number_input("Âge", min_value=18, max_value=100, value=30)
    person_income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
    person_education = st.selectbox(
        "Education",
        ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']
    )

with col2:
    st.subheader("Housing Situation and Credit")
    person_home_ownership = st.selectbox(
        "Home Ownership",
        ['RENT', 'OWN', 'MORTGAGE', 'OTHER']
    )
    credit_score = st.number_input("Credit score", min_value=300, max_value=850, value=700)
    cb_person_cred_hist_length = st.number_input(
        "Person credit history",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5
    )
    previous_loan_defaults = st.selectbox(
        "Previous loan defaults on file",
        ['No', 'Yes']
    )

with col3:
    st.subheader("Loan Details")
    loan_intent = st.selectbox(
        "Loan Intent",
        ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']
    )
    loan_amnt = st.number_input("Loan Amount ($)", min_value=0, value=10000, step=500)
    loan_int_rate = st.number_input("Loan Rate (%)", min_value=0.0, max_value=100.0, value=8.5, step=0.1)
    loan_percent_income = st.number_input(
        "Loan Percent of Income (%)",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.01
    )

st.markdown("---")

if st.button("Predict", type="primary", use_container_width=True):

    new_applicant = pd.DataFrame({
        'person_age': [person_age],
        'person_income': [person_income],
        'person_home_ownership': [person_home_ownership],
        'loan_intent': [loan_intent],
        'loan_amnt': [loan_amnt],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'credit_score': [credit_score],
        'person_education': [person_education],
        'previous_loan_defaults_on_file': [previous_loan_defaults],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length]
    })

    result = predict_loan_status(new_applicant, model, preprocessor)

    st.header("Prediction Result")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        status = "Approved" if result['status'] == 1 else "Rejected"
        st.metric("Loan Status", status)
    with col2:
        st.metric("Probability Approved", f"{result['probability_approved']*100:.2f}%")
    with col3:
        st.metric("Probability Rejected", f"{result['probability_rejected']*100:.2f}%")