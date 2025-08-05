import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# --- Configuration ---
# Data/trial_processed_data_local/version2/user_level_aggregated_data.csv
# Data/trial_processed_data_local/version3/user_segments_with_clusters.csv
INPUT_FILE_NAME = 'Data/end_result_processed_data_local/version2/user_level_aggregated_data.csv'
OUTPUT_FILE_NAME = 'Data/end_result_processed_data_local/version3/user_segments_with_clusters.csv'

# --- Step 1: Load the user-level data ---
try:
    print(f"Loading data from '{INPUT_FILE_NAME}'...")
    user_df = pd.read_csv(INPUT_FILE_NAME)
    print("Data loaded successfully!")
    print(f"Initial data shape: {user_df.shape}")
    print("\nFirst 5 rows of the user-level data:")
    print(user_df.head())
except FileNotFoundError:
    print(f"Error: The file '{INPUT_FILE_NAME}' was not found. Please ensure it is in the same directory as this script.")
    exit()

# --- Step 2: Feature Selection and Scaling ---
# We select the numerical features for clustering and exclude the 'user_id' identifier.
# Scaling is crucial for K-Means as it's sensitive to the magnitude of values.
features = user_df[['total_session_length_minutes', 'average_session_length_minutes',
                    'login_frequency', 'unique_features_used', 'unique_categories_used',
                    'total_events']]

print("\nScaling the selected features...")
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
print("Features scaled successfully!")

# --- Step 3: Run K-Means with a chosen number of clusters ---
# It's recommended to use the Elbow Method (from the previous script) to find the optimal 'k'.
# For this example, we will proceed with 4 clusters.
n_clusters = 4
print(f"\nRunning K-Means with {n_clusters} clusters...")
kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_model.fit(scaled_features)

# Add the numerical cluster labels to the DataFrame
user_df['Segment_Numerical'] = kmeans_model.labels_

# --- Step 4: Interpret the Clusters ---
# Group by the new 'Segment_Numerical' column to see the average values of each feature for each segment.
print("\n--- Cluster Interpretation: Mean values for each numerical segment ---")
segment_profiles = user_df.groupby('Segment_Numerical')[features.columns].mean().round(2)
print(segment_profiles)

# --- Step 5: Map Numerical Segments to Descriptive Names ---
# Based on the 'segment_profiles' table above, we can assign meaningful names.
# For example, if Segment 0 has high averages across all metrics, we can call it 'Power Users'.
# You MUST modify this dictionary based on your specific results from the 'segment_profiles' table.
segment_name_map = {
    0: 'Power Users',
    1: 'Regular Users',
    2: 'At-Risk Users',
    3: 'Feature Explorers'
}

# Apply the mapping to create the new 'Segment' column with descriptive names
user_df['Segment'] = user_df['Segment_Numerical'].map(segment_name_map)

# Drop the temporary numerical segment column if you no longer need it
user_df.drop('Segment_Numerical', axis=1, inplace=True)

# --- Step 6: Display the Final Result ---
print("\nFirst 5 rows with the new descriptive segment names:")
print(user_df[['user_id', 'Segment', 'login_frequency', 'total_events']].head())

# --- Step 7: Save the final segmented data to a new CSV file ---
try:
    user_df.to_csv(OUTPUT_FILE_NAME, index=False)
    print(f"\nFinal segmented data successfully saved to '{OUTPUT_FILE_NAME}'")
except Exception as e:
    print(f"Error saving file: {e}")