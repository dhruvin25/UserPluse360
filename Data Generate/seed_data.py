import json
import random
from datetime import datetime, timedelta
from faker import Faker
import pandas as pd
import uuid # For unique user_ids

# --- Configuration ---
# Define the file path for your features JSON
FEATURES_FILE = 'features.json'

# --- Data Generation Settings (based on your research plan) ---
# Trial Data
TRIAL_NUM_RECORDS = 5000 # Max 5000 logs
TRIAL_NUM_UNIQUE_USERS = 200 # Max 200 unique users
TRIAL_TIME_SPAN_DAYS = 7 # 1 week
TRIAL_NUM_FEATURES = (5, 10) # Range of distinct features to use

# End Result Data
END_RESULT_NUM_RECORDS = 1000000 # Example: 1 million logs (can be 500k-2M)
END_RESULT_NUM_UNIQUE_USERS = 15000 # Example: 15k unique users (can be 5k-20k)
END_RESULT_TIME_SPAN_MONTHS = 12 # Example: 12 months (can be 3-6 months)
END_RESULT_NUM_FEATURES = (10, 20) # Range of distinct features to use

# Session length in seconds (min, max)
SESSION_LENGTH_RANGE_SECONDS = (5, 3600) # 5 seconds to 1 hour

# Platforms
PLATFORMS = ['web', 'mobile_ios', 'mobile_android', 'desktop_app']
PLATFORM_WEIGHTS = [0.4, 0.3, 0.2, 0.1] # Web is most common, desktop least


# --- Initialize Faker ---
fake = Faker('en_CA') # Using 'en_CA' locale for Canadian context


# --- Helper Function to Load Features from JSON ---
def load_features_from_json(file_path):
    """Loads features and their categories from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        all_features_with_categories = []
        for category_entry in data.get('features', []):
            category = category_entry.get('category')
            sub_features = category_entry.get('sub_features', [])
            for sub_feature in sub_features:
                all_features_with_categories.append({
                    'category': category,
                    'feature_name': sub_feature
                })
        return all_features_with_categories
    except FileNotFoundError:
        print(f"Error: Features JSON file not found at {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Check file format.")
        return []

# Load features once
FEATURES_DATA = load_features_from_json(FEATURES_FILE)
if not FEATURES_DATA:
    raise SystemExit("Exiting: Could not load features data. Please check 'financial_features.json'.")

# Separate features and categories for easier random selection
ALL_FEATURE_NAMES = [f['feature_name'] for f in FEATURES_DATA]
FEATURE_CATEGORY_MAP = {f['feature_name']: f['category'] for f in FEATURES_DATA}


# --- Data Generation Function ---
def generate_dummy_data(
    num_records: int,
    num_unique_users: int,
    time_span_days: int,
    num_features_range: tuple,
    output_filename: str
):
    """
    Generates dummy financial application data and saves it to a JSON file.

    Args:
        num_records (int): Total number of log entries to generate.
        num_unique_users (int): Number of distinct users.
        time_span_days (int): The total duration in days for the timestamps.
        num_features_range (tuple): (min, max) range for distinct features to use.
        output_filename (str): Name of the output JSON file.
    """
    print(f"\n--- Generating {num_records} records for {output_filename} ---")

    data_records = []

    # Generate unique user_ids
    # Using UUID4 for robust unique IDs
    user_ids = [str(uuid.uuid4()) for _ in range(num_unique_users)]

    # Define the start date for the data generation (current date minus time span)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=time_span_days)

    # Determine features to pick from (a subset of all features for diversity)
    num_distinct_features_for_this_run = random.randint(*num_features_range)
    selected_features = random.sample(ALL_FEATURE_NAMES, min(num_distinct_features_for_this_run, len(ALL_FEATURE_NAMES)))

    # Optional: Assign varying activity levels to users (e.g., some users more active than others)
    # Higher weight means more records for that user
    user_activity_weights = [random.uniform(0.1, 2.0) for _ in range(num_unique_users)]
    total_activity_weight = sum(user_activity_weights)
    user_probabilities = [w / total_activity_weight for w in user_activity_weights]

    for _ in range(num_records):
        # 1. user_id: Pick a user based on activity probability
        user_id = random.choices(user_ids, weights=user_probabilities, k=1)[0]

        # 2. timestamp: Random timestamp within the defined range
        time_delta_seconds = random.randint(0, int(time_span_days * 24 * 3600))
        timestamp = start_date + timedelta(seconds=time_delta_seconds)

        # 3. feature_used & category_used:
        # Pick a feature, ensuring category_used is correctly fetched
        feature_used = random.choice(selected_features)
        category_used = FEATURE_CATEGORY_MAP.get(feature_used, "Unknown") # Fallback for safety

        # 4. session_length: Random within plausible range, skewed towards shorter sessions
        # Using exponential distribution to skew towards shorter sessions
        # Adjust lambda for different skewness. Smaller lambda -> more shorter sessions.
        lam = 0.0005 # Adjust this value to control skewness (smaller lambda = more shorter sessions)
        session_len = round(random.expovariate(lam))
        # Clamp session length to the defined range
        session_length = max(SESSION_LENGTH_RANGE_SECONDS[0], min(session_len, SESSION_LENGTH_RANGE_SECONDS[1]))

        # 5. platform: Choose platform based on weights
        platform = random.choices(PLATFORMS, weights=PLATFORM_WEIGHTS, k=1)[0]

        record = {
            "user_id": user_id,
            "timestamp": timestamp.isoformat(), # ISO format for easy parsing later
            "category_used": category_used,
            "feature_used": feature_used,
            "session_length": session_length, # in seconds
            "platform": platform
        }
        data_records.append(record)

    # Convert to Pandas DataFrame for easy manipulation and export
    df = pd.DataFrame(data_records)

    # Save to JSON file
    # orient='records' creates a list of JSON objects (one per row)
    # indent=4 for pretty printing
    df.to_json(output_filename, orient='records', indent=4, date_format='iso')
    print(f"Generated {len(df)} records and saved to {output_filename}")


# --- Generate Data ---
# Generate Trial Data
generate_dummy_data(
    num_records=TRIAL_NUM_RECORDS,
    num_unique_users=TRIAL_NUM_UNIQUE_USERS,
    time_span_days=TRIAL_TIME_SPAN_DAYS,
    num_features_range=TRIAL_NUM_FEATURES,
    output_filename="trial_data.json"
)

# Generate End Result Data
# Convert months to days for the time_span_days parameter
end_result_time_span_days = END_RESULT_TIME_SPAN_MONTHS * 30 # Approximation
generate_dummy_data(
    num_records=END_RESULT_NUM_RECORDS,
    num_unique_users=END_RESULT_NUM_UNIQUE_USERS,
    time_span_days=end_result_time_span_days,
    num_features_range=END_RESULT_NUM_FEATURES,
    output_filename="end_result_data.json"
)

print("\nData generation complete!")
