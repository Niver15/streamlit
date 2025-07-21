import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd

# Initialize session state to store predictions
if 'history' not in st.session_state:
    st.session_state['history'] = []

st.title("Ebola Case Reduction Predictor (SDG 3)")

# Check if model exists
if not os.path.exists('model/my_model.pkl'):
    st.error("❌ Model file not found at 'model/my_model.pkl'. Please train and save the model first.")
    st.stop()

# Load model
model = joblib.load('model/my_model.pkl')
st.success("✅ Model loaded successfully!")

# Input sliders for NPIs
npi1 = st.slider('School closing (S1)', 0, 3, 1)
npi2 = st.slider('Workplace closing (S2)', 0, 3, 1)
npi3 = st.slider('Cancel public events (S3)', 0, 2, 1)
npi4 = st.slider('Restrictions on gatherings (S4)', 0, 4, 1)
npi5 = st.slider('Public information campaigns (S5)', 0, 2, 1)

# Combine into feature array
features = np.array([[npi1, npi2, npi3, npi4, npi5]])

# Predict
if st.button("Predict"):
    prediction = model.predict(features)[0]

    # Save to history
    st.session_state.history.append({
        'S1': npi1,
        'S2': npi2,
        'S3': npi3,
        'S4': npi4,
        'S5': npi5,
        'Prediction (%)': prediction
    })

    st.write(f"🧮 Predicted reduction in new cases next week: {prediction:.2f}%")
    st.success(f"✅ Predicted Disease Case Reduction: {prediction:.2f}%")

    # Show historical predictions chart
    history_df = pd.DataFrame(st.session_state.history)
    st.line_chart(history_df['Prediction (%)'])

