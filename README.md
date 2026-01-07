# 🌍 EnviroScan  
## AI-Powered Pollution Source Identifier using Geospatial Analytics

---

## Author  
**Naga Jyothi**  
B.Tech – Computer Science  
S V University,Tirupati

---

## 1. Introduction

Air pollution monitoring systems generally report pollutant concentrations but often fail to explain **where the pollution originates from**. Identifying the **likely source of pollution**—such as vehicular traffic, industrial activity, or natural causes—is essential for targeted mitigation and effective environmental planning.

This project implements an **AI-based pollution source identification system** that integrates air quality indicators, weather parameters, engineered environmental features, and geospatial visualization. The focus is on building a **practical, interpretable, and deployable system** rather than optimizing only for high accuracy.

---

## 2. Problem Statement

Given air quality and meteorological data for a specific location, the objective is to **predict the dominant pollution source (`pollution_source`)** and present the results using an interactive dashboard supported by visual analytics and maps.

---

## 3. Objectives

- Integrate air quality, weather, and contextual data  
- Perform structured data preprocessing and feature engineering  
- Train a machine learning model for pollution source classification  
- Deploy the trained model using a Streamlit dashboard  
- Visualize pollution patterns using charts and interactive maps  

---

## 4. Data Sources

### 4.1 Air Quality Data
- Pollutants considered:
  - CO AQI  
  - NO₂ AQI  
  - O₃ AQI  
  - PM2.5 AQI  
  - Overall AQI  

### 4.2 Weather Data
- Temperature  
- Humidity  
- Wind speed  
- Weather condition (encoded)

### 4.3 Spatial and Contextual Features
- Traffic influence indicators  
- Derived proximity-related features  
- Map layers generated using OpenStreetMap data  

All datasets were consolidated into a unified structured dataset.

---

## 5. Data Preprocessing

- Standardized column names for consistency  
- Removed inconsistent or incomplete records  
- Numerical values handled using robust statistical techniques  
- Categorical features encoded numerically  
- Feature scaling applied using `StandardScaler`  

---

## 6. Feature Engineering

To better reflect real-world environmental behavior, additional features were derived:

- **Traffic Pollution Index** based on gaseous pollutants  
- **Particle Load Factor** representing PM2.5 intensity relative to AQI  
- **Thermal Stress Index** combining temperature and humidity  
- **Wind-adjusted dispersion indicators** to account for pollutant spread  

These features improve model learning beyond raw pollutant measurements.

---

## 7. Machine Learning Model

The problem was formulated as a **multi-class classification task**.

### Selected Model
- **Random Forest Classifier**

### Reason for Selection
- Handles non-linear feature interactions  
- Performs well with mixed numerical features  
- Provides stable predictions for environmental data  

The trained model and preprocessing artifacts were saved using Joblib and reused during deployment.

---

## 8. Model Evaluation

The model was evaluated using:
- Accuracy  
- Precision  
- Recall  
- Confusion matrix  

The observed performance was consistent with expectations for a realistic environmental dataset.

---

## 9. Dashboard and Visualization

A **Streamlit-based interactive dashboard** was developed.

### Dashboard Features
- User-controlled environmental inputs  
- Real-time pollution source prediction  
- Confidence-based alert system  
- Pie chart showing pollution source distribution  
- Embedded interactive maps:
  - Overall pollution overview  
  - Pollution intensity heatmap  
  - High-risk zone visualization  

These components help convert model outputs into understandable insights.

---

## 10. System Workflow

1. Data loading and preprocessing  
2. Feature engineering  
3. Model training and evaluation  
4. Model serialization  
5. User input via dashboard  
6. Prediction, alert generation, and visualization  

---

## 11. Deployment

- Random Forest model deployed locally using Streamlit  
- Saved artifacts include:
  - `pollution_rf_realistic.pkl`  
  - `scaler.pkl`  
  - `target_encoder.pkl`  
- Modular design allows future scalability  

---

## 12. Conclusion

This project demonstrates an end-to-end approach to **pollution source identification** by combining machine learning, environmental feature engineering, and geospatial visualization. The system emphasizes clarity, interpretability, and practical deployment over artificial performance optimization.

---

## 13. Future Scope

- Real-time API-based data ingestion  
- Automated alert mechanisms  
- Advanced GIS-based spatial analysis  
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

## How to Run the Project

```bash
streamlit run app.py
