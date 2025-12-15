from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load("loan_status_predictor.pkl")

preprocessor = joblib.load("preprocessor.pkl")

class LoanApproval(BaseModel):
    person_age:float
    person_gender:str
    person_education:str
    person_income:float
    person_emp_exp:int
    person_home_ownership:str
    loan_amnt:float
    loan_intent:str
    loan_int_rate:float
    loan_percent_income:float
    cb_person_cred_hist_length:float
    credit_score:int
    previous_loan_defaults_on_file:str

app = FastAPI()

@app.post("/api/predict")
async def predict_loan_status(applicant_data: LoanApproval):
    input_data = pd.DataFrame([applicant_data.dict()])

    processed_data = preprocessor.transform(input_data)
    
    prediction = model.predict(processed_data)
    probabilities = model.predict_proba(processed_data)
    
    return {
        'status': 200,
        'message': 'Prediction successful',
        'data':{
            'status': int(prediction[0]),
            'probability_approved': float(probabilities[0][1]),
            'probability_rejected': float(probabilities[0][0])
        }
    }