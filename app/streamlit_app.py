import streamlit as st
import pandas as pd
import pickle

# Load saved models
rf_model = pickle.load(open('models/rf_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
imputer = pickle.load(open('models/imputer.pkl', 'rb'))
label_encoders = pickle.load(open('models/label_encoders.pkl', 'rb'))

st.title("🚀 Drop-Off Prediction Application")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    input_data = pd.read_csv(uploaded_file)

    # Preprocess input
    for col in input_data.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            input_data[col] = label_encoders[col].transform(input_data[col].astype(str))

    input_data_imputed = imputer.transform(input_data)
    input_data_scaled = scaler.transform(input_data_imputed)

    # Predict
    predictions = rf_model.predict(input_data_scaled)

    # Show results
    result_df = pd.DataFrame({'Prediction': predictions})
    st.write("### 📋 Prediction Results", result_df)

    # Download predictions
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

    # Feature Importance Visualization
    st.write("### 📊 Feature Importance")
    feature_importances = rf_model.feature_importances_
    features = input_data.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    st.bar_chart(importance_df.sort_values(by="Importance", ascending=False).head(10))
