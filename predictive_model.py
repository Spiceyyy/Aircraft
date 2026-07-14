import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

print("Step 1: Extracting data from SQLite Database...")
# Connect to your existing SQLite database
conn = sqlite3.connect('skyops.db')

# We only pull the features (X) we need for prediction, and the target (y) which is TAXI_OUT
query = """
SELECT 
    DEP_HOUR, 
    HourlyPrecipitation, 
    HourlyWindSpeed, 
    HourlyVisibility, 
    OP_UNIQUE_CARRIER, 
    TAXI_OUT 
FROM v_skyops_master
WHERE TAXI_OUT IS NOT NULL
"""
df = pd.read_sql(query, conn)
conn.close()

# Drop any rows where weather data might be missing from the join
df = df.dropna()

print(f"Loaded {len(df)} flight records for training.")

print("\nStep 2: Preprocessing the Data (Feature Engineering)...")
# Machine Learning models only understand numbers, not text.
# We must convert the airline codes (e.g., 'AA', 'DL') into numerical columns using One-Hot Encoding.
# This creates a column for every airline with a 1 or 0 (True/False).
df_encoded = pd.get_dummies(df, columns=['OP_UNIQUE_CARRIER'], drop_first=True)

# Define our Features (X) and our Target (y)
X = df_encoded.drop('TAXI_OUT', axis=1) # Everything EXCEPT the taxi time
y = df_encoded['TAXI_OUT']              # ONLY the taxi time

print("\nStep 3: Splitting data into Training and Testing sets...")
# We train the model on 80% of the data, and hide 20% to test it later so it doesn't "cheat"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nStep 4: Training the Random Forest Regression Model...")
# A Random Forest creates hundreds of "decision trees" and averages their predictions.
# It is excellent at finding non-linear patterns (e.g., rain + rush hour = massive delay).
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("\nStep 5: Evaluating the Model...")
# Now we ask the model to predict the taxi times for the 20% of data we hid from it
predictions = model.predict(X_test)

# Calculate how far off our predictions were from the actual taxi times
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"Mean Absolute Error (MAE): {mae:.2f} minutes")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} minutes")
print(f"-> Business Translation: On average, our model's taxi-out prediction is off by just {mae:.2f} minutes.")

print("\nStep 6: Feature Importance (What actually causes delays?)...")
# Let's ask the model which features it relied on the most to make its predictions
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(5)

print(feature_importance_df.to_string(index=False))