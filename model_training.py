import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_excel('data/Cleaned_Opportunity_Data_With_Record_ID.xlsx')

# Prepare Features and Target
y = df['Status Description']
X = df.drop(columns=['record_id', 'Status Description'])

# Encode categorical features
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)

# Save model and preprocessors
with open('models/rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('models/imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)

with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("✅ Models and preprocessors saved successfully!")
