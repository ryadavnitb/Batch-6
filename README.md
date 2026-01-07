# 🌍 EnviroScan  
## AI-Powered Pollution Source Identifier using Geospatial Analytics

---

## Author  
**Naga Jyothi**  
B.Tech – Computer Science  
S V University,Tirupati

---

## 1. Introduction

Air pollution monitoring systems usually report pollutant concentrations but often fail to identify the **source of pollution**. Understanding whether pollution originates from vehicular traffic, industrial activity, agricultural zones, or natural causes is essential for effective mitigation and policy decisions.

EnviroScan is an **AI-based pollution source identification system** that combines air quality data, weather parameters, spatial context, and machine learning to predict the most likely pollution source for a given scenario. The project emphasizes **practical deployment and interpretability** rather than purely theoretical performance.

---

## 2. Problem Statement

Given environmental data such as air quality indicators, meteorological conditions, and spatial proximity features, the goal is to **predict the dominant pollution source** affecting a location and present the result through an interactive dashboard.

---

## 3. Objectives

- Analyze air quality and weather data  
- Perform data preprocessing and feature alignment  
- Train a machine learning model for pollution source classification  
- Deploy the trained model using a Streamlit dashboard  
- Visualize pollution trends and high-risk zones  

---

## 4. Data Sources

### 4.1 Air Quality Data
- CO AQI  
- NO₂ AQI  
- Ozone AQI  
- PM2.5 AQI  
- Overall AQI  

### 4.2 Weather Data
- Temperature  
- Humidity  
- Wind speed  
- Wind direction  

### 4.3 Spatial and Contextual Features
- Road, industry, farmland, and dump site indicators  
- Distance-based proximity features  
- Location-related attributes derived from OpenStreetMap  

All features were consolidated into a single structured dataset used for both training and inference.

---

## 5. Data Preprocessing

- Standardized and cleaned dataset columns  
- Removed non-numeric attributes from model input  
- Handled missing values using median-based strategies  
- Ensured consistent feature schema between training and deployment  
- Preserved original column encodings to avoid inference mismatch  

---

## 6. Feature Engineering

Key features used by the model include:

- Pollutant concentration values (CO, NO₂, O₃, PM2.5, AQI)  
- Weather parameters (temperature, humidity, wind speed, wind direction)  
- Spatial indicators (road count, industry count, proximity distances)  

Some spatial values are set to reasonable defaults during deployment when real-time location data is unavailable. This ensures feature consistency with the trained model.

---

## 7. Machine Learning Model

The problem is formulated as a **multi-class classification task**.

### Model Used
- **Random Forest Classifier**

### Justification
- Handles non-linear feature interactions effectively  
- Performs well with mixed environmental features  
- Robust to noise and imperfect real-world data  

The trained model, scaler, and label encoder are serialized using Joblib for reuse during deployment.

---

## 8. Model Training

Model training is performed using the `train_model.py` script:

- Target labels are encoded using `LabelEncoder`  
- Features are scaled using `StandardScaler`  
- The Random Forest model is trained on aligned numeric features  
- Trained artifacts are saved as `.pkl` files  

---

## 9. Dashboard and Visualization

A **Streamlit-based interactive dashboard** was developed to demonstrate the model.

### Dashboard Features
- User-controlled pollutant and weather inputs  
- Real-time pollution source prediction  
- Prediction confidence score  
- Alert system based on confidence thresholds  
- Pie chart showing pollution source distribution  
- Embedded interactive maps:
  - Overall pollution overview  
  - Pollution heatmap  
  - High-risk zone visualization  

---

## 10. System Workflow

1. Load and preprocess dataset  
2. Train and serialize machine learning model  
3. Accept user inputs through Streamlit UI  
4. Align inputs with training feature schema  
5. Generate pollution source prediction  
6. Display results, alerts, and visualizations  

---

## 11. Deployment

- Application deployed locally using Streamlit  
- Required files:
  - `app.py`
  - `train_model.py`
  - `pollution_rf_realistic.pkl`
  - `scaler.pkl`
  - `target_encoder.pkl`
- Modular structure allows easy future extension  

---

## 12. Conclusion

EnviroScan demonstrates how machine learning and environmental data can be combined to identify pollution sources in a structured and interpretable way. The project focuses on real-world constraints such as feature consistency, data quality, and deployment practicality.

---

## 13. Future Enhancements

- Real-time API-based data ingestion  
- Automated alert notifications  
- Advanced GIS-level spatial analysis  
- Cloud-based deployment  

---

## 14. Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  
- Matplotlib  
- Folium  
- OpenStreetMap  

---

## 15. How to Run the Project

```bash
python train_model.py
python -m streamlit run app.py
