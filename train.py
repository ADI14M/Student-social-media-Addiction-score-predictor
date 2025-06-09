import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("Students Social Media Addiction.csv")
df.drop(columns=["Student_ID"], inplace=True)

# One-hot encode categorical columns
df_encoded = pd.get_dummies(df, drop_first=True)

# Separate features and target
X_df = df_encoded.drop("Addicted_Score", axis=1)
X = X_df.to_numpy().astype(np.float64)
y = df_encoded["Addicted_Score"].to_numpy().astype(np.float64)

# Normalize features
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / X_std

# Add bias term
X_final = np.c_[np.ones(X_norm.shape[0]), X_norm]

# Split data manually (80% train, 20% test)
split = int(0.8 * len(X_final))
X_train, X_test = X_final[:split], X_final[split:]
y_train, y_test = y[:split], y[split:]

# Train using pseudo-inverse
theta = np.linalg.pinv(X_train) @ y_train

# Predict function
def predict(new_input_dict):
    # Convert input dict to dataframe and one-hot encode
    new_df = pd.DataFrame([new_input_dict])
    new_df_encoded = pd.get_dummies(new_df)

    # Align new data to match training feature columns
    for col in df_encoded.drop("Addicted_Score", axis=1).columns:
        if col not in new_df_encoded.columns:
            new_df_encoded[col] = 0
    new_df_encoded = new_df_encoded[df_encoded.drop("Addicted_Score", axis=1).columns]

    # Normalize
    new_X = (new_df_encoded.values - X_mean) / X_std

    # Add bias term
    new_X = np.c_[np.ones(new_X.shape[0]), new_X]

    # Predict
    prediction = new_X @ theta
    return prediction[0]

# Evaluate on test set
y_pred = X_test @ theta
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print("Model trained successfully. RMSE on test set:", rmse)

# Example usage
example_input = {
    "Age": 21,
    "Gender": "Male",
    "Academic_Level": "Graduate",
    "Country": "India",
    "Avg_Daily_Usage_Hours": 4.5,
    "Most_Used_Platform": "Instagram",
    "Affects_Academic_Performance": "Yes",
    "Sleep_Hours_Per_Night": 6.0,
    "Mental_Health_Score": 7,
    "Relationship_Status": "Single",
    "Conflicts_Over_Social_Media": 2
}

predicted_score = predict(example_input)
print("Predicted Addicted Score:", predicted_score)