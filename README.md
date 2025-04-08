Opportunity Status Prediction 🧠
This repository contains a machine learning project aimed at predicting the Status Description of opportunities based on various features.

This code was developed as part of an internship project to practice and apply skills in data preprocessing, model training, model evaluation, and feature interpretation.

📚 Libraries Used
pandas

numpy

matplotlib

seaborn

scikit-learn

🛠️ Project Workflow
Load Dataset

Load a cleaned Excel dataset (Cleaned_Opportunity_Data_With_Record_ID.xlsx) containing opportunity data.

Define Target Variable

The target variable is Status Description.

Data Preprocessing

Drop irrelevant columns like Record ID and Status Description itself (during feature preparation).

Encode categorical features using LabelEncoder.

Handle missing values using SimpleImputer.

Scale the numerical features using StandardScaler.

Model Training

Logistic Regression: A simple linear model to classify the data.

Random Forest Classifier: An ensemble method using decision trees.

Model Evaluation

Accuracy scores for both models.

Detailed classification report (precision, recall, f1-score).

Confusion matrix visualization.

Feature importance plotting for Random Forest.

Visualization

Heatmaps for confusion matrix and classification report.

Bar plot for feature importances.

🎨 Visual Outputs
Confusion Matrix for Random Forest Classifier.

Classification Report Heatmap.

Feature Importance bar chart highlighting top predictors.
