import pandas as pd
import sqlite3
import glob

print("Starting SkyOps ETL Pipeline...")

# 1. CREATE / CONNECT TO DATABASE
# This will automatically create 'skyops.db' in your folder if it doesn't exist
conn = sqlite3.connect('skyops.db')

# ==========================================
# 2. PROCESS FLIGHT DATA (EXTRACT & TRANSFORM)
# ==========================================
# glob finds every file that starts with 'bts_flight_' and ends with '.csv'
flight_files = glob.glob('bts_flights_ord_*.csv') 
print(f"\nFound {len(flight_files)} flight files to process.")

for file in flight_files:
    print(f" -> Loading & Cleaning: {file}")
    flights_df = pd.read_csv(file, low_memory=False)
    
    # THE FIX: Filter the dataset strictly for ORD departures
    if 'ORIGIN' in flights_df.columns:
        flights_df = flights_df[flights_df['ORIGIN'] == 'ORD']
    
    # Standardize formats (Keeping our previous Windows bug fixes!)
    flights_df['FL_DATE'] = pd.to_datetime(flights_df['FL_DATE'], format='mixed').dt.strftime('%Y-%m-%d')
    flights_df['DEP_HOUR'] = (flights_df['DEP_TIME'].fillna(0).astype('int64') // 100)
    
    # Filter columns to keep the database lightweight
    columns_to_keep = ['FL_DATE', 'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'ORIGIN', 'DEP_DELAY', 'TAXI_OUT', 'DEP_HOUR']
    # List comprehension to prevent KeyErrors if a column is missing in a specific month
    flights_df = flights_df[[c for c in columns_to_keep if c in flights_df.columns]]
    
    # LOAD: Append this month's data into the 'flights' table in SQLite
    flights_df.to_sql('flights', conn, if_exists='append', index=False)

# ==========================================
# 3. PROCESS WEATHER DATA
# ==========================================
# Assuming you have one massive 12-month weather file, or you can loop it like the flights above
print("\nProcessing Weather Data...")
weather_df = pd.read_csv('noaa_weather_ord.csv', low_memory=False) # Update with your filename

# Clean Weather
weather_df['DATE'] = pd.to_datetime(weather_df['DATE'])
weather_df['FL_DATE'] = weather_df['DATE'].dt.normalize().dt.strftime('%Y-%m-%d')
weather_df['WEATHER_HOUR'] = weather_df['DATE'].dt.hour.astype('int64')

# Drop duplicate hourly reports
weather_df = weather_df.drop_duplicates(subset=['FL_DATE', 'WEATHER_HOUR'], keep='first')

# Handle Precipitation Imputation
weather_df['HourlyPrecipitation'] = weather_df['HourlyPrecipitation'].replace('T', '0.01')
weather_df['HourlyPrecipitation'] = pd.to_numeric(weather_df['HourlyPrecipitation'], errors='coerce').fillna(0)

# Filter columns
weather_cols = ['FL_DATE', 'WEATHER_HOUR', 'HourlyPrecipitation', 'HourlyWindSpeed', 'HourlyVisibility']
weather_df = weather_df[[c for c in weather_cols if c in weather_df.columns]]

# LOAD: Replace/Create the weather table
weather_df.to_sql('weather', conn, if_exists='replace', index=False)

# 4. CLOSE CONNECTION
conn.close()
print("\nPipeline Complete! Data successfully loaded into 'skyops.db'.")