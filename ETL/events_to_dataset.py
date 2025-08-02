import pandas as pd
import numpy as np

# --- Configuration ---
# Assuming the CSV file is in the same directory as this script.
# If not, provide the full path to the file.
# Data/trial_processed_data_local/version1/fact_events_transformed.csv
FILE_NAME = 'Data/end_result_processed_data_local/version1/fact_events_transformed.csv'
OUTPUT_FILE_NAME = 'Data/end_result_processed_data_local/version2/user_level_aggregated_data.csv'

# --- Step 1: Data Loading ---
try:
    print(f"Loading data from '{FILE_NAME}'...")
    df_events = pd.read_csv(FILE_NAME)
    print("Data loaded successfully!")
    print(f"Initial data shape: {df_events.shape}")
    print("\nFirst 5 rows of the raw data:")
    print(df_events.head())
except FileNotFoundError:
    print(f"Error: The file '{FILE_NAME}' was not found. Please ensure the file is in the correct directory.")
    # Exit the script gracefully if the file is not found
    exit()
except Exception as e:
    print(f"An error occurred while loading the file: {e}")
    exit()

# --- Step 2: Feature Engineering - Grouping and Aggregating Data ---
print("\nAggregating data to create a user-level dataset...")

# We will group by 'user_id' and apply various aggregation functions.
# The .agg() function allows us to apply multiple functions to different columns simultaneously.
user_data = df_events.groupby('user_id').agg(
    # Calculate total and average session length
    total_session_length_minutes=('session_length', 'sum'),
    average_session_length_minutes=('session_length', 'mean'),

    # Calculate login frequency (by counting unique dates)
    login_frequency=('date', pd.Series.nunique),

    # Calculate the number of unique features and categories used
    unique_features_used=('feature_used', pd.Series.nunique),
    unique_categories_used=('category_used', pd.Series.nunique),

    # Count the number of events per user
    total_events=('user_id', 'size')
)

# Reset the index to make 'user_id' a column again, which is good practice
user_data = user_data.reset_index()

# --- Step 3: Additional Feature Engineering - Platform Usage ---
# We can also create features for which platforms a user has used.
# This uses a pivot table to count the number of events per platform for each user.
# Then, we merge this back into our main user_data DataFrame.
print("Counting platform usage for each user...")
platform_counts = df_events.groupby(['user_id', 'platform']).size().unstack(fill_value=0)
user_data = user_data.merge(platform_counts, on='user_id', how='left')
user_data.rename(columns=lambda col: f'platform_{col}_count' if col not in ['user_id', 'total_session_length_minutes', 'average_session_length_minutes', 'login_frequency', 'unique_features_used', 'unique_categories_used', 'total_events'] else col, inplace=True)
user_data.fillna(0, inplace=True)

# --- Step 4: Displaying the Final User-Level Data ---
print("\nUser-level data aggregation complete!")
print(f"Final user-level data shape: {user_data.shape}")
print("\nFirst 5 rows of the aggregated user-level data:")
print(user_data.head())

# --- Step 5: Save the Aggregated Data to a New CSV File ---
try:
    user_data.to_csv(OUTPUT_FILE_NAME, index=False)
    print(f"\nAggregated data successfully saved to '{OUTPUT_FILE_NAME}'")
except Exception as e:
    print(f"Error saving file: {e}")

