import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
import joblib # For saving and loading models

'''
Churn Definition: We'll define a user as "churned" if their last activity date is older than a specified CHURN_THRESHOLD_DAYS (e.g., 30 days) from the latest date observed in your dataset.

Feature Engineering: We'll create features like days_since_last_login in addition to the aggregated metrics from the previous K-Means task.

Class Imbalance Handling: Churn datasets are often imbalanced (many more non-churned users than churned users). We'll use SMOTE (Synthetic Minority Over-sampling Technique) to balance the training data, which helps models learn better from the minority (churned) class.

Model Comparison: We'll train three common classification models and compare their performance using the F1-score, which is a good metric for imbalanced datasets as it balances precision and recall.
'''

# Data/end_result_processed_data_local/version1/fact_events_transformed.csv // large dataset with user events for better feature engineering
# ML/best_churn_model.pkl // path to save the best churn model
# --- Configuration ---
FACT_EVENTS_FILE = 'Data/end_result_processed_data_local/version1/fact_events_transformed.csv'
CHURN_THRESHOLD_DAYS = 30 # Define churn: no login for this many days from the latest date
TEST_SIZE = 0.2           # Percentage of data for testing
RANDOM_STATE = 42         # For reproducibility
MODEL_SAVE_PATH = 'ML/best_churn_model.pkl' # Path to save the best model

# --- Step 1: Load and Pre-process Data for Churn Definition ---
try:
    print(f"Loading data from '{FACT_EVENTS_FILE}'...")
    df_events = pd.read_csv(FACT_EVENTS_FILE)
    print("Data loaded successfully!")
    print(f"Initial data shape: {df_events.shape}")
    print("\nFirst 5 rows of the raw event data:")
    print(df_events.head())
except FileNotFoundError:
    print(f"Error: The file '{FACT_EVENTS_FILE}' was not found. Please ensure it is in the correct directory.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the file: {e}")
    exit()

# Convert 'date' column to datetime objects
df_events['date'] = pd.to_datetime(df_events['date'])

# Determine the latest date in the dataset (our "snapshot" date)
snapshot_date = df_events['date'].max()
print(f"\nLatest activity date in the dataset (snapshot date): {snapshot_date.strftime('%Y-%m-%d')}")

# --- Step 2: Feature Engineering for User-Level Data with Churn Label ---
print("\nEngineering user-level features and defining churn status...")

# Calculate last login date for each user
last_login_df = df_events.groupby('user_id')['date'].max().reset_index()
last_login_df.rename(columns={'date': 'last_login_date'}, inplace=True)

# Calculate days since last login from the snapshot date
last_login_df['days_since_last_login'] = (snapshot_date - last_login_df['last_login_date']).dt.days

# Define churn status: 1 if churned, 0 otherwise
# A user is churned if their last login date is older than CHURN_THRESHOLD_DAYS from the snapshot date.
last_login_df['churn_status'] = (last_login_df['days_since_last_login'] > CHURN_THRESHOLD_DAYS).astype(int)

# Aggregate other user-level features from df_events
user_features = df_events.groupby('user_id').agg(
    total_session_length_minutes=('session_length', 'sum'),
    average_session_length_minutes=('session_length', 'mean'),
    login_frequency=('date', pd.Series.nunique),
    unique_features_used=('feature_used', pd.Series.nunique),
    unique_categories_used=('category_used', pd.Series.nunique),
    total_events=('user_id', 'size')
).reset_index()

# Merge platform usage counts (similar to K-Means script)
platform_counts = df_events.groupby(['user_id', 'platform']).size().unstack(fill_value=0).reset_index()
# Rename platform columns for clarity
platform_counts.rename(columns=lambda col: f'platform_{col}_count' if col != 'user_id' else col, inplace=True)

# Merge all user-level data
user_df = user_features.merge(last_login_df, on='user_id', how='left')
user_df = user_df.merge(platform_counts, on='user_id', how='left')

# Fill any NaN values that might arise from platform_counts merge (e.g., if a user never used a specific platform)
# Use 0 for counts, as it means they didn't use that platform.
platform_cols = [col for col in user_df.columns if 'platform_' in col and '_count' in col]
user_df[platform_cols] = user_df[platform_cols].fillna(0)


print("\nFirst 5 rows of the engineered user-level data with churn status:")
print(user_df.head())
print(f"\nChurn status distribution:\n{user_df['churn_status'].value_counts()}")

# --- Step 3: Prepare Data for Classification ---
# Define features (X) and target (y)
X = user_df.drop(columns=['user_id', 'last_login_date', 'churn_status'])
y = user_df['churn_status']

# Handle potential NaN values in features (e.g., from new platform columns if not all platforms exist for all users)
X = X.fillna(0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
print(f"\nData split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled successfully!")

# Handle class imbalance using SMOTE
print("Checking for class imbalance and applying SMOTE if necessary...")
if y_train.value_counts()[0] / y_train.value_counts()[1] > 1.5: # Simple check for imbalance
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"Original training set shape: {X_train_scaled.shape}")
    print(f"Resampled training set shape: {X_train_resampled.shape}")
    print(f"Resampled churn status distribution:\n{pd.Series(y_train_resampled).value_counts()}")
else:
    X_train_resampled, y_train_resampled = X_train_scaled, y_train
    print("No significant class imbalance detected or SMOTE not applied.")


# --- Step 4: Train and Evaluate Multiple Classification Models ---
models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, solver='liblinear'),
    'Random Forest Classifier': RandomForestClassifier(random_state=RANDOM_STATE),
    'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=RANDOM_STATE)
}

f1_scores = {}
best_f1_score = -1
best_model_name = None
best_model = None

print("\n--- Training and Evaluating Models ---")
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test_scaled)
    
    f1 = f1_score(y_test, y_pred)
    f1_scores[name] = f1
    
    print(f"{name} F1-Score: {f1:.4f}")
    print(f"Classification Report for {name}:\n{classification_report(y_test, y_pred)}")

    if f1 > best_f1_score:
        best_f1_score = f1
        best_model_name = name
        best_model = model

print("\n--- Model Comparison Results (F1-Score) ---")
for name, f1 in f1_scores.items():
    print(f"{name}: {f1:.4f}")

print(f"\nBest model based on F1-Score: {best_model_name} (F1-Score: {best_f1_score:.4f})")

# --- Step 5: Save the Best Model and Scaler ---
try:
    joblib.dump(best_model, MODEL_SAVE_PATH)
    joblib.dump(scaler, 'scaler.pkl') # Save the scaler as well, it's needed for prediction
    print(f"\nBest model ('{best_model_name}') and scaler saved to '{MODEL_SAVE_PATH}' and 'scaler.pkl'.")
except Exception as e:
    print(f"Error saving model or scaler: {e}")

