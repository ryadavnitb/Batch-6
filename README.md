# 🌍 EnviroScan:AI-Powered Pollution Source Identifier using Geospatial Analytics

## Author
**Pawan Hingane**  
B.Tech (Engineering)  
VIT Bhopal University  

---

## 1. Introduction

Air pollution has emerged as a critical environmental and public health challenge worldwide. Identifying the **source of pollution**—such as vehicular emissions, industrial activity, or natural contributors—is essential for informed decision-making, mitigation strategies, and policy development.

This project presents a **machine learning–based approach** to classify pollution sources by integrating **air quality data, meteorological conditions, spatial proximity features, and geospatial visualization**. The primary focus is on developing a **realistic, robust, and deployable system**, prioritizing generalization and interpretability over artificially inflated performance metrics.

---

## 2. Problem Statement

Given environmental data collected from multiple heterogeneous sources, the objective is to **predict the primary pollution source (`pollution_source`)** affecting a given geographical location and visually represent pollution patterns for decision support.

### Objectives
- Integrate air quality, weather, and spatial data from multiple sources  
- Perform systematic data preprocessing and feature engineering  
- Train and evaluate multiple machine learning classification models  
- Select a stable model for deployment  
- Visualize pollution severity and sources using interactive maps  
- Ensure realistic performance while avoiding overfitting  

---

## 3. Data Sources

### 3.1 Weather Data
- **Source:** OpenWeather API  
- **Features:** Temperature, humidity, wind speed, and weather description  

### 3.2 Air Pollution Data
- **Source:** Dataset provided by *Ankit Sir*  
- **Features:**  
  - Carbon Monoxide (CO) AQI  
  - Nitrogen Dioxide (NO₂) AQI  
  - Ozone (O₃) AQI  
  - Particulate Matter (PM2.5) AQI  
  - Overall Air Quality Index (AQI)  

### 3.3 Physical and Proximity Features
- **Source:** OpenStreetMap (OSM) library  
- **Features:**  
  - Distance to nearest road  
  - Distance to industrial areas  
  - Distance to traffic hotspots  

All datasets were merged using spatial coordinates and timestamps to create a unified dataset.

---

## 4. Data Preprocessing and Cleaning

### 4.1 Data Cleaning
- Standardized column names for consistency  
- Removed invalid records and missing target labels  
- Handled missing numerical values using median imputation  
- Handled missing categorical values using mode imputation  

### 4.2 Encoding
- Label encoding for categorical features  
- Label encoding for the target variable  

### 4.3 Feature Scaling
- Standardization using `StandardScaler` for uniform model input  

---

## 5. Feature Engineering

The following engineered features were derived to improve predictive performance:

- **Traffic Pollution Index:** Average of CO and NO₂ AQI  
- **Particulate Ratio:** PM2.5 AQI relative to overall AQI  
- **Heat–Humidity Index:** Combined effect of temperature and humidity  
- **Spatial Proximity Features:**  
  - Distance to road (km)  
  - Distance to industrial area (km)  
  - Distance to traffic hotspot (km)  

To simulate real-world conditions, controlled **sensor noise** and approximately **18% label noise** were introduced.

---

## 6. Machine Learning Models

| Model | Purpose |
|------|--------|
| Random Forest | Baseline and deployed production model |
| Decision Tree | Interpretable comparison model |
| XGBoost | Advanced benchmarking model |

---

## 7. Model Architecture and Training

### 7.1 Random Forest (Baseline Model)
- Depth-limited decision trees  
- Minimum sample constraints  
- Balanced class weights  

This model was selected for deployment due to its **robustness, stability, and strong generalization capability**.

### 7.2 Decision Tree
- Used primarily for interpretability  
- Hyperparameters optimized to reduce overfitting  

### 7.3 XGBoost
- Gradient boosting ensemble model  
- Used for performance comparison only  

---

## 8. Hyperparameter Tuning

Hyperparameter optimization was performed using **RandomizedSearchCV**, primarily on the Decision Tree model.  
The Random Forest configuration was intentionally preserved to maintain baseline consistency.

---

## 9. Model Evaluation

### Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- Weighted F1-score  
- Confusion Matrix  
- Stratified K-Fold Cross-Validation  

### Performance Summary
- **Random Forest Accuracy:** ~80–85%  
- Decision Tree: Lower accuracy, expected due to simplicity  
- XGBoost: Comparable or slightly higher accuracy  

---

## 10. Overfitting Analysis

Overfitting was controlled through:
- Model depth constraints  
- Minimum sample requirements  
- Noise injection  
- Cross-validation consistency  

Comparable performance across training and test datasets confirms **no overfitting**.

---

## 11. Geospatial Visualization and Mapping

To enhance interpretability and decision support, model predictions were visualized using **interactive geospatial maps**:

- Predictions and location data were loaded into an interactive mapping interface  
- **Folium** was used to generate dynamic pollution heatmaps  
- Source-specific markers were overlaid to represent pollution origins (e.g., industrial, vehicular)  
- High-risk zones were visualized using color gradients based on pollutant severity  
- Filters were implemented to explore data by:
  - Date  
  - Location  
  - Predicted pollution source  

The maps were embedded into the web dashboard for user interaction and analysis.

---

## 12. System Architecture

### System Workflow
1. Data collection from APIs and curated datasets  
2. Data preprocessing and cleaning  
3. Feature engineering and transformation  
4. Model training and evaluation  
5. Prediction generation  
6. Geospatial visualization and dashboard integration  
7. Model serialization using Joblib  

---

## 13. Deployment

- **Random Forest** selected as the production model  
- Saved artifacts include:
  - `pollution_rf_realistic.pkl`
  - `scaler.pkl`
  - `target_encoder.pkl`  

Other models are retained for experimentation and comparison.

---

## 14. Conclusion

This project delivers an end-to-end **pollution source classification system** that integrates machine learning with geospatial visualization. By focusing on realistic data handling, robust modeling, and interactive mapping, the system provides meaningful insights suitable for real-world environmental monitoring and decision-making.

---

## 15. Future Scope
- Real-time data streaming and alerts  
- Advanced GIS-based spatial analysis  
- Satellite data integration  
- Cloud-based scalable deployment  

---

## 16. References
- OpenWeather API Documentation  
- OpenStreetMap (OSM)  
- Scikit-learn Documentation  
- XGBoost Documentation

