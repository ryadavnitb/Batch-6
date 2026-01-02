import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Live Prediction",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Live Pollution Source Prediction")
st.caption("Predict pollution source using trained ML model")

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
@st.cache_resource
def load_models():
    model = joblib.load("best_random_forest_model.pkl")
    scaler = joblib.load("scaler.joblib")
    encoder = joblib.load("label_encoder.pkl")
    return model, scaler, encoder

try:
    model, scaler, encoder = load_models()
except:
    st.error("❌ Models not found. Run pipeline from Home page.")
    st.stop()

# -------------------------------------------------
# GET FEATURE NAMES USED DURING TRAINING
# -------------------------------------------------
feature_names = scaler.feature_names_in_

st.subheader("🧾 Input Environmental Parameters")

# -------------------------------------------------
# USER INPUTS (only for key pollutants)
# -------------------------------------------------
user_inputs = {}

c1, c2, c3 = st.columns(3)

with c1:
    user_inputs["PM25 AQ"] = st.number_input("PM2.5", 0.0, 500.0, 60.0)
    user_inputs["PM10 AQ"] = st.number_input("PM10", 0.0, 500.0, 90.0)
    user_inputs["CO AQ"] = st.number_input("CO", 0.0, 10.0, 1.2)

with c2:
    user_inputs["NO2 AQ"] = st.number_input("NO₂", 0.0, 200.0, 40.0)
    user_inputs["SO2 AQ"] = st.number_input("SO₂", 0.0, 200.0, 20.0)
    user_inputs["O3 AQ"] = st.number_input("O₃", 0.0, 200.0, 30.0)

with c3:
    st.info("Other features use trained defaults", icon="ℹ️")

# -------------------------------------------------
# BUILD FULL FEATURE VECTOR (15 FEATURES)
# -------------------------------------------------
input_vector = []

for feature in feature_names:
    if feature in user_inputs:
        input_vector.append(user_inputs[feature])
    else:
        # Default values for non-user inputs
        input_vector.append(0.0)

X = np.array(input_vector).reshape(1, -1)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if st.button("🔮 Predict Pollution Source", type="primary"):

    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    prob = model.predict_proba(X_scaled).max()

    st.session_state["predicted_source"] = encoder.inverse_transform(pred)[0]
    st.session_state["confidence"] = round(prob * 100, 2)

    st.success("✅ Prediction Successful")
    st.metric("🧪 Predicted Source", st.session_state["predicted_source"])
    st.metric("📊 Confidence", f"{st.session_state['confidence']}%")

# -------------------------------------------------
# DOWNLOAD PREDICTION RESULT
# -------------------------------------------------

# -------------------------------------------------
# DOWNLOAD PREDICTION RESULT
# -------------------------------------------------
if "predicted_source" in st.session_state:

    st.markdown("---")
    st.subheader("⬇️ Download Prediction Result")

    prediction_df = pd.DataFrame({
        "PM25 AQ": [user_inputs.get("PM25 AQ", 0)],
        "PM10 AQ": [user_inputs.get("PM10 AQ", 0)],
        "CO AQ": [user_inputs.get("CO AQ", 0)],
        "NO2 AQ": [user_inputs.get("NO2 AQ", 0)],
        "SO2 AQ": [user_inputs.get("SO2 AQ", 0)],
        "O3 AQ": [user_inputs.get("O3 AQ", 0)],
        "Predicted_Source": [st.session_state["predicted_source"]],
        "Confidence (%)": [st.session_state["confidence"]]
    })

    csv_pred = prediction_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction",
        data=csv_pred,
        file_name="live_pollution_prediction.csv",
        mime="text/csv"
    )

# -------------------------------------------------
# DEBUG (optional)
# -------------------------------------------------
with st.expander("🔍 Model Feature Info"):
    st.write("Model expects these features:")
    st.code(feature_names.tolist())
