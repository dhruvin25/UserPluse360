import pandas as pd

# Data/end_result_processed_data_local/version3/user_segments_with_clusters.csv // user segments with clusters
# Data/end_result_processed_data_local/version4/user_churn_predictions.csv // user churn predictions
# Merging these two files to check for discrepancies in user data 
# Define the file paths and column names as provided by you
file1_name = 'Data/end_result_processed_data_local/version3/user_segments_with_clusters.csv'
file1_columns = ['user_id', 'total_session_length_minutes', 'average_session_length_minutes', 'login_frequency', 'unique_features_used', 'unique_categories_used', 'total_events', 'platform_desktop_app_count', 'platform_mobile_android_count', 'platform_mobile_ios_count', 'platform_web_count', 'Segment']

file2_name = 'Data/end_result_processed_data_local/version4/user_churn_predictions.csv'
file2_columns = ['user_id', 'total_session_length_minutes', 'average_session_length_minutes', 'login_frequency', 'unique_features_used', 'unique_categories_used', 'total_events', 'last_login_date', 'days_since_last_login', 'platform_desktop_app_count', 'platform_mobile_android_count', 'platform_mobile_ios_count', 'platform_web_count', 'churn_probability', 'predicted_churn']

# Read the CSV files into DataFrames
try:
    df1 = pd.read_csv(file1_name, usecols=file1_columns)
    df2 = pd.read_csv(file2_name, usecols=file2_columns)
except FileNotFoundError as e:
    print(f"Error: One of the files was not found. Please check the file names and try again. Error: {e}")
    exit()

# Identify common and unique columns
common_columns = list(set(file1_columns) & set(file2_columns))
unique_to_df1 = list(set(file1_columns) - set(common_columns))
unique_to_df2 = list(set(file2_columns) - set(common_columns))

# The key for merging is 'user_id'
merge_key = 'user_id'

# Merge the two DataFrames on the common key
merged_df = pd.merge(df1, df2, on=merge_key, how='outer', suffixes=('_file1', '_file2'))

# Create a DataFrame to store discrepancies
discrepancies_df = pd.DataFrame()

# Check for discrepancies in common columns
discrepancy_found = False
for col in common_columns:
    if col != merge_key:
        col1 = f'{col}_file1'
        col2 = f'{col}_file2'

        # Find rows where the values for the common column are different
        diff_mask = merged_df[col1].ne(merged_df[col2]) & merged_df[col1].notna() & merged_df[col2].notna()

        if diff_mask.any():
            discrepancy_found = True
            temp_df = merged_df[diff_mask][[merge_key, col1, col2]].copy()
            temp_df.rename(columns={col1: f'{col}_file1', col2: f'{col}_file2'}, inplace=True)
            if discrepancies_df.empty:
                discrepancies_df = temp_df
            else:
                # To merge discrepancies on the same user_id for multiple columns
                discrepancies_df = pd.merge(discrepancies_df, temp_df, on=merge_key, how='outer')

# Remove the duplicated common columns from the merged_df to avoid redundancy in the final merged file
for col in common_columns:
    if col != merge_key:
        merged_df[col] = merged_df[f'{col}_file1'].fillna(merged_df[f'{col}_file2'])
        merged_df.drop(columns=[f'{col}_file1', f'{col}_file2'], inplace=True)

# Reorder columns for better readability
final_merged_columns = [merge_key] + sorted([col for col in merged_df.columns if col != merge_key])
merged_df = merged_df[final_merged_columns]

# Data/end_result_processed_data_local/version5/merged_data.csv // final merged data
# Data/end_result_processed_data_local/version5/discrepancies.csv // discrepancies found in the common columns
# Save the final merged DataFrame and discrepancies DataFrame to CSV files
merged_output_file = 'Data/end_result_processed_data_local/version5/merged_data.csv'
discrepancies_output_file = 'Data/end_result_processed_data_local/version5/discrepancies.csv'

merged_df.to_csv(merged_output_file, index=False)

if discrepancy_found:
    discrepancies_df.to_csv(discrepancies_output_file, index=False)
    print(f"Successfully merged '{file1_name}' and '{file2_name}'.")
    print(f"Final merged data saved to '{merged_output_file}'.")
    print(f"Discrepancies found and saved to '{discrepancies_output_file}'.")
else:
    print(f"Successfully merged '{file1_name}' and '{file2_name}'.")
    print(f"Final merged data saved to '{merged_output_file}'.")
    print("No data discrepancies were found in the common columns.")