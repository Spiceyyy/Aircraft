import pandas as pd
import numpy as np

# 1. Load the Data 
# Replace these filenames with the actual names of the CSVs you downloaded
print("Loading datasets...")
flights_df = pd.read_csv('bts_flights_ord_jan.csv')
weather_df = pd.read_csv('noaa_weather_ord_jan.csv', low_memory=False)

# 2. Clean and Prepare Flight Data
print("Cleaning flight data...")
# Ensure the flight date is a proper datetime object
flights_df['FL_DATE'] = pd.to_datetime(flights_df['FL_DATE'])

# BTS provides departure time as a float (e.g., 1430.0 for 2:30 PM). 
# We extract just the hour (14) to match with NOAA's hourly weather reports.
flights_df['DEP_HOUR'] = flights_df['DEP_TIME'].fillna(0).astype(int) // 100

# 3. Clean and Prepare Weather Data
print("Cleaning weather data...")
# NOAA data has a 'DATE' column containing both date and time (e.g., 2024-01-01 14:53:00)
weather_df['DATE'] = pd.to_datetime(weather_df['DATE'])
weather_df['FL_DATE'] = weather_df['DATE'].dt.normalize() # Extracts just the YYYY-MM-DD
weather_df['WEATHER_HOUR'] = weather_df['DATE'].dt.hour

# NOAA sometimes records multiple observations per hour; keep only the first one to avoid duplicates
weather_df = weather_df.drop_duplicates(subset=['FL_DATE', 'WEATHER_HOUR'], keep='first')

# 4. Merge the Datasets
print("Joining flights and weather...")
# We perform a left join so we don't lose any flights, matching on BOTH the day and the hour
merged_df = pd.merge(
    flights_df,
    weather_df,
    left_on=['FL_DATE', 'DEP_HOUR'],
    right_on=['FL_DATE', 'WEATHER_HOUR'],
    how='left'
)

# 5. Handle Missing Values (Imputation)
# Convert 'T' (Trace amounts of rain) to a small number, and fill missing rain data with 0
merged_df['HourlyPrecipitation'] = merged_df['HourlyPrecipitation'].replace('T', '0.01')
merged_df['HourlyPrecipitation'] = pd.to_numeric(merged_df['HourlyPrecipitation'], errors='coerce').fillna(0)

# 6. Select Relevant Columns for Power BI
# We only want to export what we need for the dashboard to keep it lightweight
columns_to_keep = [
    'FL_DATE', 'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'DEP_DELAY',
    'TAXI_OUT', 'HourlyPrecipitation', 'HourlyWindSpeed', 'HourlyVisibility'
]
final_df = merged_df[columns_to_keep]

# 7. Export the Clean Data
final_df.to_csv('skyops_powerbi_ready.csv', index=False)
print("Pipeline complete! 'skyops_powerbi_ready.csv' is ready for your dashboard.")