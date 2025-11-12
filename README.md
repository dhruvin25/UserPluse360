# SaaS User Engagement & Churn Analysis Pipeline

**Python | Pandas | Scikit-learn | AWS S3 | Snowflake | Looker Studio**

> Have you ever wondered how your favorite apps know what features you use the most, or how they might predict if you’re about to stop using their service? It all starts with a well-designed data pipeline. This project was my journey to build just that: a comprehensive, end-to-end data pipeline that transforms raw user logs into actionable business intelligence. My goal was simple but powerful: to track feature usage and user engagement within a simulated SaaS application and analyze it to drive smart business decisions.

---

## 🏁 Final Dashboard: Executive Overview



This project transforms millions of raw, simulated user logs into a powerful, interactive dashboard that drives business decisions.

## 💡 Key Features

* **End-to-End ETL:** A complete pipeline from raw data generation to cloud data warehousing.
* **Predictive Modeling:** A machine learning model that predicts user churn probability.
* **User Segmentation:** K-Means clustering to automatically group users into behavioral segments (e.g., "Power Users," "At-Risk").
* **Business Intelligence:** A final dashboard in Looker Studio tracking KPIs like DAU/WAU, top features, and retention.
* **Cloud Scalability:** Built using a modern data stack (AWS S3, Snowflake) designed for massive scale.

## 🎯 Target Audience

This repository demonstrates skills relevant to several key data roles:

* **For Data Analysts & BI Developers:** See the `Phase 2` ETL scripts for metric calculation (DAU/WAU) and `Phase 5` for the final Looker Studio dashboard design and KPI tracking.
* **For Data Engineers:** See `Phase 1` for data simulation, `Phase 4` for the cloud architecture, S3 ingestion, and the Snowflake DDL/DML scripts for building the data warehouse.
* **For Business Analysts:** See the `Phase 3` machine learning implementation (churn/segmentation) and the `Phase 5` dashboards, which translate complex data into actionable business insights.

---

## ⚙️ The 5-Phase Project Pipeline

Here’s a breakdown of the journey, from data generation to the final dashboard.

### Phase 1: Simulating Reality with Python

Every great data project needs great data. I built a Python script using **Faker** and **Pandas** to generate a massive, high-fidelity JSON dataset (1M+ records) that mimics user interactions with a financial app. This provided a realistic and scalable foundation for the entire pipeline.

<details>
  <summary>Click to see Python data generation snippet</summary>
  
  ```python
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
  lam = 0.0005 # Adjust this value to control skewness
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
```
</details>

Phase 2: The ETL Engine
With the raw data ready, I built a local ETL (Extract, Transform, Load) script using Pandas to turn our raw JSON logs into a structured, tabular format. This script computed foundational business metrics such as:

```
DAU (Daily Active Users)

WAU (Weekly Active Users)

Top Features
```
The output of this phase was a set of clean CSV files, perfectly prepared for the next step.

Phase 3: Unlocking Deeper Insights with Machine Learning
This is where the project moved from simple metrics to predictive insights. I used Scikit-learn to answer more complex business questions:
```
User Segmentation: Leveraged K-Means Clustering to automatically segment users into groups based on their behavior, helping to identify “Power Users,” “At-Risk Users,” or “Casual Browsers.”

Churn Prediction: Built a classification model to identify which users were most likely to leave the service, outputting a churn_probability score for each user.
```
<details> <summary>Click to see Python churn prediction snippet</summary>

```Python

# Select the same features used during training
# Ensure the order of columns is the same as during training!
features_for_prediction = [
    'total_session_length_minutes', 'average_session_length_minutes',
    'login_frequency', 'unique_features_used', 'unique_categories_used',
    'total_events', 'days_since_last_login'
]
# Add platform columns dynamically, ensuring they are present and in order
for col in scaler.feature_names_in_: # Use feature_names_in_ from scaler
    if col not in features_for_prediction:
        features_for_prediction.append(col)

X_predict = user_df_for_prediction[features_for_prediction]

# Fill any NaN values that might arise from platform_counts merge
X_predict = X_predict.fillna(0)

# --- Step 3: Scale the new data using the loaded scaler ---
X_predict_scaled = scaler.transform(X_predict)

# --- Step 4: Make Predictions ---
# Predict churn probability (0: no churn, 1: churn)
churn_probabilities = best_model.predict_proba(X_predict_scaled)[:, 1] # Probability of class 1
churn_predictions = best_model.predict(X_predict_scaled) # Binary prediction

user_df_for_prediction['churn_probability'] = churn_probabilities
user_df_for_prediction['predicted_churn'] = churn_predictions
```
</details>

Phase 4: Scaling Up with the Cloud Data Warehouse
The local CSVs were proof of concept. To make this production-ready, I used a modern cloud stack.
```
Storage: Deployed all clean, transformed, and enriched CSV files to an AWS S3 bucket.

Data Warehouse: Used Snowflake to manage and query the data at scale. I created a robust data model with a logical structure of tables (e.g., DIM_USERS, FACT_EVENTS) and ingested the data from S3 using Snowflake's COPY INTO command.
```
<details> <summary>Click to see Snowflake SQL (DDL & Ingestion) snippet</summary>

```SQL

-- Use the new warehouse, database, and schema
USE WAREHOUSE UserPulse360;
USE DATABASE UserPulse360_DB;
USE SCHEMA RAW_DATA;

CREATE OR REPLACE STAGE s3_ingest_stage
  URL = 's3://userpulse360/loadingdata/csv/'
  STORAGE_INTEGRATION = AWS_S3_INT
  FILE_FORMAT = (TYPE = 'CSV', SKIP_HEADER = 1);
  
-- Copy data from S3 into the DIM_USERS table
COPY INTO DIM_USERS
FROM @s3_ingest_stage/merged_data.csv
FILE_FORMAT = (TYPE = 'CSV', SKIP_HEADER = 1)
ON_ERROR = 'SKIP_FILE';

-- Copy data from S3 into the FACT_EVENTS table
COPY INTO FACT_EVENTS (user_id, timestamp, category_used, feature_used, session_length_sec, platform, date,week,year)
FROM @s3_ingest_stage/fact_events_transformed.csv
FILE_FORMAT = (TYPE = 'CSV', SKIP_HEADER = 1)
ON_ERROR = 'SKIP_FILE';

-- Copy data from S3
COPY INTO FACT_DAILY_ACTIVE_USERS
FROM @s3_ingest_stage/metrics_dau.csv
FILE_FORMAT = (TYPE = 'CSV', SKIP_HEADER = 1)
ON_ERROR = 'SKIP_FILE';

-- Copy data from S3
COPY INTO FACT_WEEKLY_ACTIVE_USERS
FROM @s3_ingest_stage/metrics_wau.csv
FILE_FORMAT = (TYPE = 'CSV', SKIP_HEADER = 1)
ON_ERROR = 'SKIP_FILE';

UPDATE FACT_WEEKLY_ACTIVE_USERS
SET week_start_date = 
    DATE_ADD(
        DATE(CONCAT(year, '-01-01')), 
        INTERVAL (week - 1) * 7 - WEEKDAY(DATE(CONCAT(year, '-01-01'))) DAY
    );

-- Copy data from S3
COPY INTO FACT_FEATURE_USAGE
FROM @s3_ingest_stage/metrics_feature_usage_over_time.csv
FILE_FORMAT = (TYPE = 'CSV', SKIP_HEADER = 1)
ON_ERROR = 'SKIP_FILE';

-- Copy data from S3
COPY INTO FACT_FEATURE_SUMMARY
FROM @s3_ingest_stage/metrics_top_features.csv
FILE_FORMAT = (TYPE = 'CSV', SKIP_HEADER = 1)
ON_ERROR = 'SKIP_FILE';
```
</details>

Phase 5: The Business Analysis & Dashboard
The final and most rewarding phase was connecting Looker Studio to Snowflake. This is where the entire pipeline comes to life and delivers real business value. I built several dashboards, each designed to answer specific business questions:
```
Executive Overview: High-level metrics like DAU/WAU and a summary of churn probability.

Engagement & Retention: A deeper dive into user behavior and at-risk segments.

Feature Performance: Insights into which features are most popular and who is using them.
```
By interacting with this dashboard, a business analyst could now easily identify which user segments are most engaged, track the popularity of new features, and proactively target at-risk users to prevent churn.

(Project management for this entire build was tracked in Notion)
