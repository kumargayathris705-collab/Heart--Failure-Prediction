import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Failure Prediction")

st.title("❤️ Heart Failure Prediction using Machine Learning")

st.write("Upload a CSV file to predict heart disease.")

model = joblib.load("model.pkl")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.write(data.head())

    try:
        data_encoded = pd.get_dummies(data)

        predictions = model.predict(data_encoded)

        data["Prediction"] = predictions
        data["Prediction"] = data["Prediction"].map({
            0: "No Heart Disease",
            1: "Heart Disease"
        })

        st.subheader("Prediction Results")
        st.write(data)

    except Exception as e:
        st.error("Error during prediction.")
        st.write(e)