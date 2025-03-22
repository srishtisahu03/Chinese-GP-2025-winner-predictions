import os
import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

cache_dir = "f1_cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)  

fastf1.Cache.enable_cache(cache_dir)

session_2024 = fastf1.get_session(2024, 5, "R")
session_2024.load();

#2024 lap timings

laps_2024 = session_2024.laps[["Driver", "LapTime"]].copy()
laps_2024.dropna(subset=["LapTime"], inplace=True)
laps_2024["LapTime (s)"] = laps_2024["LapTime"].dt.total_seconds()      

# 2025 quali results

qualifying_2025 = pd.DataFrame({
    "Driver": ["Oscar Piastri", "George Russell", "Lando Norris", "Max Verstappen", "Lewis Hamilton", "Charles Leclerc", "Isack Hadjar", "Andrea Kimi Antonelli", "Yuki Tsunoda", "Alex Albon"],
    "QualifyingTime (s)": ["90.641", "90.723", "90.793", "90.817", "90.927", "91.021", "91.079", "91.103", "91.638", "91.706"]      
})

driver_mapping = {
    "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", "Max Verstappen": "VER", "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC", "Isack Hadjar": "HAD", "Andrea Kimi Antonelli": "ANT", "Yuki Tsunoda": "TSU", "Alex Albon": "ALB"
}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)
qualifying_2025

merged_data = qualifying_2025.merge(laps_2024, left_on="DriverCode", right_on="Driver")

X = merged_data[["QualifyingTime (s)"]]
y = merged_data["LapTime (s)"]

if X.shape[0] == 0:
    raise ValueError("Dataset is empty!")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=40)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=40)
model.fit(X_train, y_train)

