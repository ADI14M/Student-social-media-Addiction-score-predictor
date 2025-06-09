import streamlit as st
import pandas as pd
import numpy as np

# Load dataset (used only to get dummies and normalization)
df = pd.read_csv("Students Social Media Addiction.csv")
df.drop(columns=["Student_ID"], inplace=True)
df_encoded = pd.get_dummies(df, drop_first=True)

X_df = df_encoded.drop("Addicted_Score", axis=1)
X = X_df.to_numpy().astype(np.float64)
y = df_encoded["Addicted_Score"].to_numpy().astype(np.float64)

# Normalization
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / X_std
X_final = np.c_[np.ones(X_norm.shape[0]), X_norm]

# Train model
split = int(0.8 * len(X_final))
X_train = X_final[:split]
y_train = y[:split]
theta = np.linalg.pinv(X_train) @ y_train

# Prediction function
def predict(new_input_dict):
    new_df = pd.DataFrame([new_input_dict])
    new_df_encoded = pd.get_dummies(new_df)

    for col in df_encoded.drop("Addicted_Score", axis=1).columns:
        if col not in new_df_encoded.columns:
            new_df_encoded[col] = 0
    new_df_encoded = new_df_encoded[df_encoded.drop("Addicted_Score", axis=1).columns]

    new_X = (new_df_encoded.values - X_mean) / X_std
    new_X = np.c_[np.ones(new_X.shape[0]), new_X]
    prediction = new_X @ theta
    return prediction[0]

# ----------- Streamlit Interface -----------

st.title("📱 Student Social Media Addiction Score Predictor")

with st.form("prediction_form"):
    age = st.slider("Age", 13, 30, 21)
    gender = st.selectbox("Gender", ["Male", "Female"])
    academic_level = st.selectbox("Academic Level", ["Undergraduate", "Graduate", "Postgraduate"])
    country = st.selectbox("Country", ["India", "Other"])
    avg_usage = st.slider("Avg Daily Usage (hours)", 0.0, 12.0, 4.5, step=0.5)
    platform = st.selectbox("Most Used Platform", ["Instagram", "Facebook", "Twitter", "Snapchat", "Other"])
    affects_performance = st.radio("Affects Academic Performance?", ["Yes", "No"])
    sleep_hours = st.slider("Sleep Hours per Night", 0.0, 12.0, 6.0, step=0.5)
    mental_score = st.slider("Mental Health Score (1–10)", 1, 10, 7)
    relationship = st.selectbox("Relationship Status", ["Single", "In a Relationship", "Married"])
    conflicts = st.slider("Conflicts Over Social Media (per week)", 0, 10, 2)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = {
        "Age": age,
        "Gender": gender,
        "Academic_Level": academic_level,
        "Country": country,
        "Avg_Daily_Usage_Hours": avg_usage,
        "Most_Used_Platform": platform,
        "Affects_Academic_Performance": affects_performance,
        "Sleep_Hours_Per_Night": sleep_hours,
        "Mental_Health_Score": mental_score,
        "Relationship_Status": relationship,
        "Conflicts_Over_Social_Media": conflicts
    }

    result = predict(input_data)
    st.success(f"🎯 Predicted Addiction Score: **{result:.2f}**")