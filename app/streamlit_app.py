import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load models and encoders
rf_model = pickle.load(open('models/rf_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
imputer = pickle.load(open('models/imputer.pkl', 'rb'))
label_encoders = pickle.load(open('models/label_encoders.pkl', 'rb'))

# Feature names used during model training (must match exactly)
model_features = [
    'Learner SignUp DateTime',
    'Opportunity Id',
    'Opportunity Name',
    'Opportunity Category',
    'Opportunity End Date',
    'First Name',
    'Date of Birth',
    'Gender',
    'Country',
    'Institution Name',
    'Current/Intended Major',
    'Entry created at',
    'Status Code',
    'Apply Date',
    'Opportunity Start Date',
    'Age',
    'Opportunity Duration(in days)',
    'Age_Normalized',
    'Opportunity_Duration_Normalized',
    'Gender_Encoded',
    'SignUp_Month',
    'SignUp_Year',
    'Engagement_Days (Neg: Apply > Start)',
    'Engagement_Score',
    'Learner_ID',
    'Repeat_Opportunities'
]

# Streamlit app UI
st.title("🚀 Drop-Off Prediction App")

uploaded_file = st.file_uploader("📤 Upload Excel File", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    st.subheader("📄 Uploaded Data Preview")
    st.write(df)

    # Add any missing columns required for model input
    for col in model_features:
        if col not in df.columns:
            df[col] = None

    # Subset and copy only model-related columns
    input_data = df[model_features].copy()

    # Encode categorical/text columns using label encoders
    for col in input_data.select_dtypes(include='object').columns:
        if col in label_encoders:
            try:
                input_data[col] = label_encoders[col].transform(input_data[col].astype(str))
            except Exception as e:
                st.warning(f"⚠️ Encoding failed for {col}: {e}")
                input_data[col] = 0  # fallback if unseen category

    # Impute missing values
    try:
        imputed_data = imputer.transform(input_data)
    except Exception as e:
        st.error(f"❌ Imputation failed: {e}")
        st.stop()

    # Scale data
    try:
        scaled_data = scaler.transform(imputed_data)
    except Exception as e:
        st.error(f"❌ Scaling failed: {e}")
        st.stop()

    # Predict
    try:
        predictions = rf_model.predict(scaled_data)
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    # Results
    result_df = pd.DataFrame({
        'Prediction': predictions
    })

    # Include record_id if available
    if 'record_id' in df.columns:
        result_df['record_id'] = df['record_id']

    st.subheader("🔮 Prediction Results")
    st.write(result_df)

    # Download predictions
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")

    # Feature importance chart
    st.subheader("📊 Feature Importance")
    try:
        importance_df = pd.DataFrame({
            'Feature': model_features,
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
        ax.set_title('Feature Importance')
        ax.set_xlabel('Importance')
        st.pyplot(fig)
    except AttributeError:
        st.warning("⚠️ Feature importance not available for this model.")
