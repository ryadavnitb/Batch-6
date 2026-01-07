import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------
# LOAD DATA
# ----------------------------------
df = pd.read_csv("data_for_training.csv", encoding="latin1")

print("Columns in dataset:")
print(df.columns)

# ----------------------------------
# SET TARGET COLUMN NAME HERE
# 🔴 CHANGE THIS if needed after checking columns
# ----------------------------------
TARGET_COL = "pollution_source"   # <-- update if different

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")

# ----------------------------------
# SEPARATE TARGET FIRST
# ----------------------------------
y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])

# ----------------------------------
# DROP NON-NUMERIC FEATURES
# ----------------------------------
X = X.select_dtypes(include=["number"])

# ----------------------------------
# HANDLE MISSING VALUES
# ----------------------------------
X = X.fillna(X.median())

# ----------------------------------
# ENCODE TARGET
# ----------------------------------
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# ----------------------------------
# SCALE FEATURES
# ----------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------
# TRAIN MODEL
# ----------------------------------
model = RandomForestClassifier(
    n_estimators=120,
    max_depth=12,
    random_state=42
)
model.fit(X_scaled, y_encoded)

# ----------------------------------
# SAVE FILES
# ----------------------------------
joblib.dump(model, "pollution_rf_realistic.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

print("✅ Model, scaler, and encoder saved successfully")
