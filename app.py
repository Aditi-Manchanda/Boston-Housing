import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open("boston_housing_model.pkl", "rb") as file:
    model = pickle.load(file)
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)
with open("feature_names.pkl", "rb") as file:
    feature_names = pickle.load(file)

st.title("Housing Price Prediction")
st.write("Enter details")

user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"{feature}", value=0.0)

input_df = pd.DataFrame([user_input])

input_scaled = scaler.transform(input_df)

if st.button("Predict Price"):
    predicted_price = model.predict(input_scaled)[0]
    st.success(f"Predicted House Price: ${predicted_price * 1000:.2f}")
