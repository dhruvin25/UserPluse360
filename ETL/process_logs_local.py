import pandas as pd
import os

# --- Configuration ---
# Set the name of the input JSON file you want to process.
# You can choose 'trial_data.json' or 'end_result_data.json'
INPUT_JSON_FILE = 'end_result_data.json' 

# Define a directory to save the output CSV files
OUTPUT_DIR = 'end_result_processed_data_local'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")


# --- Helper Functions ---
def load_and_transform_data(file_path):
    """
    Reads a JSON file into a pandas DataFrame and performs initial transformations.
    
    Args:
        file_path (str): The path to the input JSON log file.
    
    Returns:
        pd.DataFrame: A DataFrame with the 'timestamp' column in datetime format.
    """
    print(f"1. Reading raw data from: {file_path}")
    try:
        # The orient='records' parameter is crucial because our JSON is a list of objects.
        df = pd.read_json(file_path, orient='records')
        print(f"   - Loaded {len(df)} records.")
        
        # Initial Transformations
        print("2. Performing initial transformations...")
        # Convert the 'timestamp' column to a proper datetime object.
        # This is the most critical step for any time-based analysis.
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # In a real-world scenario, we'd add checks here for data integrity.
        # For our simulated data, we can assume it's clean.
        # Example check:
        # df.dropna(subset=['timestamp'], inplace=True) 
        # print(f"   - Cleaned up data, now have {len(df)} records.")

        return df
    
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred during data loading or transformation: {e}")
        return None

def compute_metrics(df):
    """
    Computes key metrics like DAU, WAU, and top features from the DataFrame.
    
    Args:
        df (pd.DataFrame): The transformed DataFrame.
        
    Returns:
        tuple: A tuple containing DAU_df, WAU_df, top_features_df, and features_over_time_df.
    """
    print("3. Computing key metrics...")
    
    # 3.1 Calculate Daily Active Users (DAU)
    # First, we need to extract the date part from the timestamp.
    df['date'] = df['timestamp'].dt.date
    # Then, we group by date and count the number of unique user_ids.
    dau_df = df.groupby('date')['user_id'].nunique().reset_index()
    dau_df.columns = ['date', 'active_users']
    print(f"   - Calculated DAU, with data from {dau_df['date'].min()} to {dau_df['date'].max()}.")

    # 3.2 Calculate Weekly Active Users (WAU)
    # A simple way is to use isocalendar() to get the week number.
    # Note: isocalendar() gives year, week, and weekday.
    df['week'] = df['timestamp'].dt.isocalendar().week
    # Group by both year and week to handle multi-year data correctly
    df['year'] = df['timestamp'].dt.year
    wau_df = df.groupby(['year', 'week'])['user_id'].nunique().reset_index()
    wau_df.columns = ['year', 'week', 'active_users']
    print(f"   - Calculated WAU for {len(wau_df)} weeks.")

    # 3.3 Determine Top Used Features
    # The value_counts() method is perfect for this.
    top_features_df = df['feature_used'].value_counts().reset_index()
    top_features_df.columns = ['feature_used', 'usage_count']
    print(f"   - Found {len(top_features_df)} distinct features. Top 5 are:\n{top_features_df.head(5)}")

    # 3.4 Analyze Feature Usage Over Time
    # Group by both date and feature_used to see how each feature's usage evolves.
    features_over_time_df = df.groupby(['date', 'feature_used'])['feature_used'].count().reset_index(name='usage_count')
    print("   - Analyzed feature usage over time.")
    
    return dau_df, wau_df, top_features_df, features_over_time_df


def store_data_locally(df, dau_df, wau_df, top_features_df, features_over_time_df, output_dir):
    """
    Stores the raw and computed dataframes as CSVs in the specified directory.
    
    Args:
        df (pd.DataFrame): The main, transformed DataFrame (fact_events).
        dau_df (pd.DataFrame): The DAU aggregation.
        wau_df (pd.DataFrame): The WAU aggregation.
        top_features_df (pd.DataFrame): The top features aggregation.
        features_over_time_df (pd.DataFrame): Feature usage over time.
        output_dir (str): The directory to save the files.
    """
    print(f"\n4. Storing data locally in the '{output_dir}' directory...")
    
    # Save the main transformed DataFrame (Simulating a `fact_events` table)
    # We set index=False to prevent pandas from writing its own index column.
    df.to_csv(os.path.join(output_dir, 'fact_events_transformed.csv'), index=False)
    print("   - Saved 'fact_events_transformed.csv'")
    
    # Save the DAU metrics
    dau_df.to_csv(os.path.join(output_dir, 'metrics_dau.csv'), index=False)
    print("   - Saved 'metrics_dau.csv'")
    
    # Save the WAU metrics
    wau_df.to_csv(os.path.join(output_dir, 'metrics_wau.csv'), index=False)
    print("   - Saved 'metrics_wau.csv'")
    
    # Save the top features
    top_features_df.to_csv(os.path.join(output_dir, 'metrics_top_features.csv'), index=False)
    print("   - Saved 'metrics_top_features.csv'")

    # Save feature usage over time
    features_over_time_df.to_csv(os.path.join(output_dir, 'metrics_feature_usage_over_time.csv'), index=False)
    print("   - Saved 'metrics_feature_usage_over_time.csv'")

    print("\nLocal ETL and analysis complete!")


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- E: Extract ---
    # Load the raw data and perform initial transformations.
    events_df = load_and_transform_data(INPUT_JSON_FILE)
    
    if events_df is not None:
        # --- T: Transform ---
        # Compute the DAU, WAU, and other metrics from the DataFrame.
        dau, wau, top_features, features_over_time = compute_metrics(events_df)
        
        # --- L: Load (to local storage) ---
        # Store the transformed and aggregated data locally as CSV files.
        store_data_locally(events_df, dau, wau, top_features, features_over_time, OUTPUT_DIR)
