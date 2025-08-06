import pandas as pd
import numpy as np
from datetime import timedelta
import joblib # For loading models and scaler

# Data/end_result_processed_data_local/version1/fact_events_transformed.csv // large dataset with user events for better feature engineering
# ML/best_churn_model.pkl // path to save the best churn model
# Data/end_result_processed_data_local/version4/user_churn_predictions.csv // path to save predictions
# ML/scaler.pkl // path to save the scaler used for feature scaling

# --- Configuration ---
FACT_EVENTS_FILE = 'Data/end_result_processed_data_local/version1/fact_events_transformed.csv' # Needed to re-engineer features for new data
MODEL_LOAD_PATH = 'ML/best_churn_model.pkl'
SCALER_LOAD_PATH = 'ML/scaler.pkl'
PREDICTED_OUTPUT_FILE = 'Data/end_result_processed_data_local/version4/user_churn_predictions.csv'

# --- Step 1: Load the Best Model and Scaler ---
try:
    print(f"Loading best churn model from '{MODEL_LOAD_PATH}'...")
    best_model = joblib.load(MODEL_LOAD_PATH)
    print(f"Loading scaler from '{SCALER_LOAD_PATH}'...")
    scaler = joblib.load(SCALER_LOAD_PATH)
    print("Model and scaler loaded successfully!")
except FileNotFoundError:
    print(f"Error: Model or scaler file not found. Please run 'churn_model_comparison.py' first.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the model or scaler: {e}")
    exit()

# --- Step 2: Simulate New Data and Engineer Features ---
# In a real-world scenario, you would load new, unseen user event data here.
# For this example, we'll re-load the original event data and simulate
# new users by taking a subset or by pretending it's "new" data.
# It's crucial that the feature engineering process is IDENTICAL to the training script.

try:
    print(f"\nLoading data from '{FACT_EVENTS_FILE}' to simulate new user data...")
    df_events_new = pd.read_csv(FACT_EVENTS_FILE)
    df_events_new['date'] = pd.to_datetime(df_events_new['date'])
    print("Simulated new data loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file '{FACT_EVENTS_FILE}' was not found. Please ensure it is in the correct directory.")
    exit()

# --- Re-apply the same feature engineering steps as in the training script ---
# Determine the latest date in the dataset (our "snapshot" date)
snapshot_date = df_events_new['date'].max()
CHURN_THRESHOLD_DAYS = 30 # Must be the same as used in training

last_login_df_new = df_events_new.groupby('user_id')['date'].max().reset_index()
last_login_df_new.rename(columns={'date': 'last_login_date'}, inplace=True)
last_login_df_new['days_since_last_login'] = (snapshot_date - last_login_df_new['last_login_date']).dt.days

user_features_new = df_events_new.groupby('user_id').agg(
    total_session_length_minutes=('session_length', 'sum'),
    average_session_length_minutes=('session_length', 'mean'),
    login_frequency=('date', pd.Series.nunique),
    unique_features_used=('feature_used', pd.Series.nunique),
    unique_categories_used=('category_used', pd.Series.nunique),
    total_events=('user_id', 'size')
).reset_index()

platform_counts_new = df_events_new.groupby(['user_id', 'platform']).size().unstack(fill_value=0).reset_index()
platform_counts_new.rename(columns=lambda col: f'platform_{col}_count' if col != 'user_id' else col, inplace=True)

user_df_for_prediction = user_features_new.merge(last_login_df_new, on='user_id', how='left')
user_df_for_prediction = user_df_for_prediction.merge(platform_counts_new, on='user_id', how='left')

platform_cols = [col for col in user_df_for_prediction.columns if 'platform_' in col and '_count' in col]
user_df_for_prediction[platform_cols] = user_df_for_prediction[platform_cols].fillna(0)

# Select the same features used during training
# Ensure the order of columns is the same as during training!
# We get the feature names from the scaler's fitted features if available, or from the training script's X columns.
# For simplicity here, we'll re-list them.
features_for_prediction = [
    'total_session_length_minutes', 'average_session_length_minutes',
    'login_frequency', 'unique_features_used', 'unique_categories_used',
    'total_events', 'days_since_last_login'
]
# Add platform columns dynamically, ensuring they are present and in order
for col in scaler.feature_names_in_: # Use feature_names_in_ from scaler for exact order
    if col not in features_for_prediction:
        features_for_prediction.append(col)

X_predict = user_df_for_prediction[features_for_prediction]

# Fill any NaN values that might arise from platform_counts merge (e.g., if a user never used a specific platform)
# Use 0 for counts, as it means they didn't use that platform.
X_predict = X_predict.fillna(0)


print("\nFirst 5 rows of data prepared for prediction:")
print(X_predict.head())

# --- Step 3: Scale the new data using the loaded scaler ---
print("\nScaling new data for prediction...")
X_predict_scaled = scaler.transform(X_predict)
print("Data scaled successfully!")

# --- Step 4: Make Predictions ---
print("\nMaking churn predictions...")
# Predict churn probability (0: no churn, 1: churn)
churn_probabilities = best_model.predict_proba(X_predict_scaled)[:, 1] # Probability of class 1 (churn)
churn_predictions = best_model.predict(X_predict_scaled) # Binary prediction (0 or 1)

user_df_for_prediction['churn_probability'] = churn_probabilities
user_df_for_prediction['predicted_churn'] = churn_predictions

print("\nFirst 10 users with their predicted churn probability and status:")
print(user_df_for_prediction[['user_id', 'last_login_date', 'days_since_last_login', 'churn_probability', 'predicted_churn']].head(10))

# --- Step 5: Save Predictions ---
try:
    user_df_for_prediction.to_csv(PREDICTED_OUTPUT_FILE, index=False)
    print(f"\nChurn predictions saved to '{PREDICTED_OUTPUT_FILE}'")
except Exception as e:
    print(f"Error saving prediction file: {e}")

