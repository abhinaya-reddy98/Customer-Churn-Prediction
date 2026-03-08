import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load saved files
model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))
columns = pickle.load(open("columns.pkl","rb"))

st.title("Customer Churn Prediction App")

st.write("Enter customer information")

# Inputs
gender = st.selectbox("Gender",["Male","Female"])
SeniorCitizen = st.selectbox("Senior Citizen",[0,1])
Partner = st.selectbox("Partner",["Yes","No"])
Dependents = st.selectbox("Dependents",["Yes","No"])
tenure = st.slider("Tenure",0,72,12)

PhoneService = st.selectbox("Phone Service",["Yes","No"])
MultipleLines = st.selectbox("Multiple Lines",
["No","Yes","No phone service"])

InternetService = st.selectbox(
"Internet Service",
["DSL","Fiber optic","No"]
)

OnlineSecurity = st.selectbox("Online Security",
["Yes","No","No internet service"])

OnlineBackup = st.selectbox("Online Backup",
["Yes","No","No internet service"])

DeviceProtection = st.selectbox("Device Protection",
["Yes","No","No internet service"])

TechSupport = st.selectbox("Tech Support",
["Yes","No","No internet service"])

StreamingTV = st.selectbox("Streaming TV",
["Yes","No","No internet service"])

StreamingMovies = st.selectbox("Streaming Movies",
["Yes","No","No internet service"])

Contract = st.selectbox(
"Contract",
["Month-to-month","One year","Two year"]
)

PaperlessBilling = st.selectbox(
"Paperless Billing",
["Yes","No"]
)

PaymentMethod = st.selectbox(
"Payment Method",
[
"Electronic check",
"Mailed check",
"Bank transfer (automatic)",
"Credit card (automatic)"
]
)

MonthlyCharges = st.number_input("Monthly Charges",0.0,200.0,70.0)
TotalCharges = st.number_input("Total Charges",0.0,10000.0,1000.0)

# Predict button
if st.button("Predict Churn"):

    data = {
        "gender":gender,
        "SeniorCitizen":SeniorCitizen,
        "Partner":Partner,
        "Dependents":Dependents,
        "tenure":tenure,
        "PhoneService":PhoneService,
        "MultipleLines":MultipleLines,
        "InternetService":InternetService,
        "OnlineSecurity":OnlineSecurity,
        "OnlineBackup":OnlineBackup,
        "DeviceProtection":DeviceProtection,
        "TechSupport":TechSupport,
        "StreamingTV":StreamingTV,
        "StreamingMovies":StreamingMovies,
        "Contract":Contract,
        "PaperlessBilling":PaperlessBilling,
        "PaymentMethod":PaymentMethod,
        "MonthlyCharges":MonthlyCharges,
        "TotalCharges":TotalCharges
    }

    df = pd.DataFrame([data])

    df = pd.get_dummies(df)

    df = df.reindex(columns=columns,fill_value=0)

    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    if prediction == 1:
        st.error(f"Customer likely to churn (Probability {probability:.2f})")
    else:
        st.success(f"Customer likely to stay (Probability {probability:.2f})")