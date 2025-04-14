import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Drop-Off Prediction App", layout="wide")
st.title("🚀 Drop-Off Prediction App")

# Upload Excel File
st.sidebar.header("📤 Upload Excel File")
uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    # Read the uploaded file
    df = pd.read_excel(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # Ensure 'Status Description' is cleaned up and check unique values
    df['Status Description'] = df['Status Description'].astype(str).str.strip()
    st.write("Unique values in 'Status Description' column:")
    st.write(df['Status Description'].unique())

    # Create DropOff column based on 'Status Description'
    df['DropOff'] = df['Status Description'].str.strip().str.lower().apply(lambda x: 'dropped out' in x or 'withdrawn' in x)

    # Filters
    st.sidebar.subheader("🔍 Filters")
    country_options = df['Country'].dropna().unique().tolist()
    selected_country = st.sidebar.selectbox("Filter by Country", ['All'] + country_options)

    gender_options = df['Gender'].dropna().unique().tolist()
    selected_gender = st.sidebar.selectbox("Filter by Gender", ['All'] + gender_options)

    filtered_df = df.copy()
    if selected_country != 'All':
        filtered_df = filtered_df[filtered_df['Country'] == selected_country]
    if selected_gender != 'All':
        filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]

    # Prediction Summary
    st.subheader("🔮 Prediction Results")
    st.write("Total Records:", len(filtered_df))
    st.write("Drop-Off Count:", filtered_df['DropOff'].sum())
    st.write("Drop-Off Rate (%):", round(filtered_df['DropOff'].mean() * 100, 2))

    # Drop-Off Summary (Pie Chart)
    st.subheader("📊 Drop-Off Summary")
    pie_data = filtered_df['DropOff'].value_counts().rename(index={0: 'Not Dropped', 1: 'Dropped'})
    st.plotly_chart(px.pie(names=pie_data.index, values=pie_data.values, title="Drop-Off Distribution"))

    # Drop-Off by Category
    st.subheader("🗺️ Drop-Off by Category")
    for category in ['Gender', 'Country']:  
        if category in filtered_df.columns:
            try:
                category_counts = (
                    filtered_df.groupby([category, 'DropOff'])
                    .size()
                    .unstack(fill_value=0)
                    .rename(columns={0: 'Not Dropped', 1: 'Dropped'})
                )
                st.write(f"### Drop-Off Count by {category}")
                st.bar_chart(category_counts)
            except Exception as e:
                st.warning(f"⚠️ Could not plot drop-off by {category}: {e}")

    # Time-Based Drop-Off Trends
    st.subheader("📅 Time-Based Drop-Off Trends")

    if 'SignUp_Month' in filtered_df.columns:
        try:
            time_counts = (
                filtered_df.groupby(['SignUp_Month', 'DropOff'])
                .size()
                .unstack(fill_value=0)
                .rename(columns={0: 'Not Dropped', 1: 'Dropped'})
            )
            st.write("### Sign-Up Month vs Drop-Off Count")
            st.line_chart(time_counts)
        except Exception as e:
            st.warning(f"⚠️ Could not show sign-up trends: {e}")

    if 'Opportunity Start Date' in filtered_df.columns:
        try:
            filtered_df['Opportunity Start Month'] = pd.to_datetime(filtered_df['Opportunity Start Date'], errors='coerce').dt.to_period('M').astype(str)
            start_month_counts = (
                filtered_df.groupby(['Opportunity Start Month', 'DropOff'])
                .size()
                .unstack(fill_value=0)
                .rename(columns={0: 'Not Dropped', 1: 'Dropped'})
            )
            st.write("### Opportunity Start Month vs Drop-Off Count")
            st.line_chart(start_month_counts)
        except Exception as e:
            st.warning(f"⚠️ Could not show opportunity start trends: {e}")

    # Feature Distributions
    st.sidebar.subheader("📌 Feature Distribution Filter")
    st.subheader("📈 Feature Distributions by Drop-Off")

    numeric_columns = filtered_df.select_dtypes(include='number').columns.tolist()
    selected_features = st.sidebar.multiselect("Select numeric features to visualize", numeric_columns)

    for feature in selected_features:
        try:
            fig = px.histogram(filtered_df, x=feature, color=filtered_df['DropOff'].map({True: 'Dropped', False: 'Not Dropped'}),
                               marginal="box", nbins=30, barmode='overlay', opacity=0.7)
            fig.update_layout(title=f"{feature} Distribution by Drop-Off Status")
            st.plotly_chart(fig)
        except Exception as e:
            st.warning(f"⚠️ Could not plot {feature}: {e}")

    # 💡 Recommendations
    st.subheader("💡 Recommendations to Reduce Drop-Offs")
    st.markdown("""
    - **Identify High-Risk Demographics**: Focus support efforts on countries or genders with higher drop-off rates.
    - **Improve Onboarding for Early Sign-Ups**: If early sign-up months show high drop-off, improve onboarding in those periods.
    - **Monitor Opportunity Start Timing**: Look for patterns where drop-offs increase and align support accordingly.
    - **Engage Regularly**: Send periodic nudges or check-ins to users during key inactivity windows.
    - **Personalized Follow-ups**: Tailor interventions based on past behavior data or regional insights.
    """)

    # 📈 Predictive Modeling (Experimental)
    st.subheader("🤖 Predictive Modeling (Experimental)")

    try:
        model_df = filtered_df.select_dtypes(include=['number', 'bool']).copy()
        model_df = model_df.dropna()
        X = model_df.drop(columns=['DropOff'])
        y = model_df['DropOff']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        st.success(f"Model Accuracy on Current Data: {round(accuracy * 100, 2)}%")

        # Feature Importance
        feature_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        # Create a more informative bar chart for Feature Importance
        fig = px.bar(
            feature_imp,
            orientation='h',
            title='Feature Importance (Random Forest)',
            labels={'index': 'Features', 'value': 'Importance'},
            color=feature_imp,
            color_continuous_scale='Viridis'
        )
        
        # Add feature importance values as annotations on the bars
        fig.update_traces(text=feature_imp.round(3), textposition='outside', insidetextanchor='start')

        # Customize layout for readability
        fig.update_layout(
            title="Feature Importance in Predicting Drop-Off",
            xaxis_title="Importance",
            yaxis_title="Features",
            template="plotly_white",
            margin=dict(l=200),  # Adjust the left margin to avoid clipping
            showlegend=False
        )
        
        # Display the plot
        st.plotly_chart(fig)

    except Exception as e:
        st.warning(f"⚠️ Could not train model: {e}")
