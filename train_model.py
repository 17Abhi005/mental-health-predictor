# train_model.py

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

DATA_PATH = "data/survey.csv"
MODEL_PATH = "model/model.pkl"
CLEANED_DATA_PATH = "data/clean_survey.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

df = df[df['Age'].between(18, 60)]

# Clean gender column
def clean_gender(val):
    val = str(val).strip().lower()
    if 'm' in val and not 'f' in val:
        return 'Male'
    elif 'f' in val:
        return 'Female'
    else:
        return 'Other'
df['Gender'] = df['Gender'].apply(clean_gender)

# Map treatment target
df['treatment'] = df['treatment'].map({'Yes': 1, 'No': 0})

# Fill missing with known good defaults or mode
df['self_employed'].fillna('No', inplace=True)
df['family_history'].fillna('No', inplace=True)
df['work_interfere'].fillna('Never', inplace=True)

# Replace unexpected values in categorical features
def safe_map(col, valid_values):
    return col.apply(lambda x: x if x in valid_values else valid_values[0])

df['self_employed'] = safe_map(df['self_employed'].astype(str), ['No', 'Yes'])
df['family_history'] = safe_map(df['family_history'].astype(str), ['No', 'Yes'])
df['work_interfere'] = safe_map(df['work_interfere'].astype(str), ['Never', 'Rarely', 'Sometimes', 'Often'])

categorical_cols = ['self_employed', 'family_history', 'work_interfere']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  
features = ['Age', 'self_employed', 'family_history', 'work_interfere']
target = 'treatment'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model Evaluation:\n")
print(classification_report(y_test, y_pred))

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

# Save cleaned dataset
df.to_csv(CLEANED_DATA_PATH, index=False)

print(f"\nModel saved to {MODEL_PATH}")
print(f"Cleaned data saved to {CLEANED_DATA_PATH}")
