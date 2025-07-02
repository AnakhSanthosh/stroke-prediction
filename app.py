import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("stroke.csv")
df.dropna(inplace=True)
df.drop('id', axis=1, inplace=True)

# Save original values for UI
gender_options = df['gender'].unique()
married_options = df['ever_married'].unique()
work_options = df['work_type'].unique()
residence_options = df['Residence_type'].unique()
smoke_options = df['smoking_status'].unique()

# Encode and scale
le_gender = LabelEncoder().fit(df['gender'])
le_married = LabelEncoder().fit(df['ever_married'])
le_work = LabelEncoder().fit(df['work_type'])
le_residence = LabelEncoder().fit(df['Residence_type'])
le_smoke = LabelEncoder().fit(df['smoking_status'])

df['gender'] = le_gender.transform(df['gender'])
df['ever_married'] = le_married.transform(df['ever_married'])
df['work_type'] = le_work.transform(df['work_type'])
df['Residence_type'] = le_residence.transform(df['Residence_type'])
df['smoking_status'] = le_smoke.transform(df['smoking_status'])

X = df.drop("stroke", axis=1)
y = df["stroke"]

sc = StandardScaler()
X_scaled = sc.fit_transform(X)

model = RandomForestClassifier()
model.fit(X_scaled, y)

# Streamlit UI
st.title("🧠 Stroke Prediction App")

st.sidebar.header("Patient Information")

def user_input():
    gender = st.sidebar.selectbox("Gender", gender_options)
    age = st.sidebar.slider("Age", 0, 100, 25)
    hypertension = st.sidebar.selectbox("Hypertension (0=No, 1=Yes)", [0, 1])
    heart_disease = st.sidebar.selectbox("Heart Disease (0=No, 1=Yes)", [0, 1])
    ever_married = st.sidebar.selectbox("Ever Married", married_options)
    work_type = st.sidebar.selectbox("Work Type", work_options)
    Residence_type = st.sidebar.selectbox("Residence Type", residence_options)
    avg_glucose_level = st.sidebar.slider("Avg Glucose Level", 50, 300, 100)
    bmi = st.sidebar.slider("BMI", 10.0, 60.0, 20.0)
    smoking_status = st.sidebar.selectbox("Smoking Status", smoke_options)

    data = {
        'gender': le_gender.transform([gender])[0],
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': le_married.transform([ever_married])[0],
        'work_type': le_work.transform([work_type])[0],
        'Residence_type': le_residence.transform([Residence_type])[0],
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': le_smoke.transform([smoking_status])[0]
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input()
scaled_input = sc.transform(input_df)
prediction = model.predict(scaled_input)[0]
st.subheader("Prediction:")
st.write("✅ Likely No Stroke" if prediction == 0 else "⚠️ High Risk of Stroke")
