import streamlit as st
import pandas as pd
import joblib

# Title
st.title("Fraud Detection using Generative AI")

st.write("Upload transaction dataset for prediction.")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    # Read data
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(data.head())

    # Load model
    model = joblib.load("fraud_model.pkl")

    # Prediction
    predictions = model.predict(data)

    # Add results
    data["Prediction"] = predictions

    st.subheader("Prediction Results")
    st.dataframe(data)

    # Download results
    csv = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Results",
        data=csv,
        file_name="prediction_results.csv",
        mime="text/csv"
    )