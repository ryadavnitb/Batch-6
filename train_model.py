import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Generate Synthetic "Real-World" Data for Training
# (In a real scenario, you would load a CSV here)
data_size = 5000
np.random.seed(42)

data = pd.DataFrame({
    'pm25': np.random.randint(10, 300, data_size),
    'no2': np.random.randint(5, 100, data_size),
    'co': np.random.randint(1, 20, data_size),
    'ozone': np.random.randint(10, 150, data_size),
    'dist_road': np.random.randint(50, 10000, data_size),
    'dist_ind': np.random.randint(100, 10000, data_size),
    'dist_agri': np.random.randint(100, 10000, data_size),
})

# Define rules to label the data (so the model learns these patterns)
conditions = [
    (data['dist_ind'] < 2000) & (data['pm25'] > 100),       # Industry
    (data['dist_road'] < 1000) & (data['no2'] > 40),        # Traffic
    (data['dist_agri'] < 2000) & (data['pm25'] > 80),       # Agriculture
    (data['ozone'] > 80)                                    # Smog
]
choices = ['Industrial Emissions', 'Vehicular Traffic', 'Agricultural Burning', 'Photochemical Smog']

data['source'] = np.select(conditions, choices, default='Natural/Background')

# 2. Train the Real Model
X = data[['pm25', 'no2', 'co', 'ozone', 'dist_road', 'dist_ind', 'dist_agri']]
y = data['source']

print("Training Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 3. Save the Model
joblib.dump(model, 'pollution_model.pkl')
print("Success! 'pollution_model.pkl' has been saved.")