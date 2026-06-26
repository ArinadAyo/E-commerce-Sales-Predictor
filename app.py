
import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('linear_regression_model.pkl')

st.title("E-Commerce Sales Predictor")
st.write("Enter customer details below to predict yearly amount spent.")

# Input fields
session_length     = st.number_input("Avg. Session Length (minutes)", min_value=0.0, value=30.0)
time_on_app        = st.number_input("Time on App (minutes)", min_value=0.0, value=12.0)
time_on_website    = st.number_input("Time on Website (minutes)", min_value=0.0, value=37.0)
membership_length  = st.number_input("Length of Membership (years)", min_value=0.0, value=3.0)

# Predict button
if st.button("Predict"):
    features = np.array([[session_length, time_on_app, time_on_website, membership_length]])
    prediction = model.predict(features)
    st.success(f"Predicted Yearly Amount Spent: ${prediction[0]:,.2f}")
