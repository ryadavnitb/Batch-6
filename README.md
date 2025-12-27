# 🌍 Pollution Source Classification Using Machine Learning

## Author
**Pawan Hingane**  
B.Tech (Engineering)  
VIT Bhopal University  

---

## 1. Introduction

Air pollution is a major environmental and public health concern. Identifying the **source of pollution**—such as traffic, industrial activity, or mixed sources—is essential for effective mitigation and policy planning.

This project aims to develop a **machine learning–based system** to classify pollution sources using **air quality indicators, weather conditions, and physical proximity features**. The emphasis is on building a **realistic, robust, and deployable model**, rather than achieving artificially high accuracy.

---

## 2. Problem Statement

Given environmental and atmospheric data collected from multiple sources, the objective is to **predict the primary pollution source (`pollution_source`)** affecting a region.

### Objectives
- Integrate multi-source environmental data
- Perform preprocessing and feature engineering
- Train and evaluate multiple classification models
- Select a stable model for deployment
- Ensure realistic performance and avoid overfitting

---

## 3. Data Sources

### 3.1 Weather Data
- **Source:** OpenWeather API  
- **Features:** Temperature, humidity, wind speed, weather description  

### 3.2 Air Pollution Data
- **Source:** Provided by *Ankit Sir*  
- **Features:**  
  - CO AQI  
  - NO₂ AQI  
  - Ozone AQI  
  - PM2.5 AQI  
  - Overall AQI  

### 3.3 Physical / Proximity Features
- **Source:** OpenStreetMap (OSM) library  
- **Features:**  
  - Distance to nearest road  
  - Distance to industrial area  
  - Distance to traffic hotspot  

All datasets were merged using temporal and spatial alignment.

---

## 4. Data Preprocessing and Cleaning

### 4.1 Data Cleaning
- Standardized column names
- Removed invalid or missing target labels
- Filled missing numerical values using median
- Filled missing categorical values using mode

### 4.2 Encoding
- Label encoding for categorical variables
- Label encoding for the target variable

### 4.3 Feature Scaling
- Standardization using `StandardScaler`

---

## 5. Feature Engineering

The following engineered features were added:
- **Traffic Pollution Index** (average of CO and NO₂ AQI)
- **Particulate Ratio** (PM2.5 AQI / overall AQI)
- **Heat–Humidity Index** (temperature × humidity)
- **Proximity Features**
  - Distance to road (km)
  - Distance to industry (km)
  - Distance to traffic hotspot (km)

To simulate real-world uncertainty, **sensor noise** and **label noise (~18%)** were injected.

---

## 6. Machine Learning Models

| Model | Purpose |
|------|--------|
| Random Forest | Baseline & deployed model |
| Decision Tree | Baseline comparison |
| XGBoost | Advanced benchmarking |

---

## 7. Model Architecture and Training

### 7.1 Random Forest (Baseline Model)
- Depth-limited trees
- Minimum sample constraints
- Balanced class weights

Selected for deployment due to **stable and generalized performance**.

### 7.2 Decision Tree
- Used for interpretability
- Hyperparameters tuned using `RandomizedSearchCV`

### 7.3 XGBoost
- Gradient boosting model
- Used only for performance comparison

---

## 8. Hyperparameter Tuning

Hyperparameter tuning was performed using **RandomizedSearchCV** on the Decision Tree model.  
The Random Forest model was intentionally kept unchanged to preserve baseline performance.

---

## 9. Model Evaluation

### Metrics Used
- Accuracy
- Precision
- Recall
- F1-score (weighted)
- Confusion Matrix
- Stratified K-Fold Cross-Validation

### Results Summary
- **Random Forest Accuracy:** ~80–85%
- Decision Tree: Lower accuracy (expected)
- XGBoost: Comparable or slightly higher accuracy

---

## 10. Overfitting Analysis

Overfitting was controlled using:
- Depth limitation
- Minimum sample constraints
- Noise injection
- Cross-validation consistency

Similar performance across training, validation, and test sets confirms **no overfitting**.

---

## 11. System Architecture

### System Flow
1. Data collection from APIs and datasets  
2. Data preprocessing and cleaning  
3. Feature engineering  
4. Model training and evaluation  
5. Model export using Joblib  
6. Integration with dashboard  

---

## 12. Deployment

- **Random Forest** used as production model
- Saved artifacts:
  - `pollution_rf_realistic.pkl`
  - `scaler.pkl`
  - `target_encoder.pkl`
- Other models retained for comparison only

---

## 13. Conclusion

This project demonstrates a **robust and realistic machine learning pipeline** for pollution source classification. By prioritizing generalization and explainability, the model avoids overfitting and is suitable for real-world deployment.

---

## 14. Future Scope
- Real-time data streaming
- GIS-based proximity refinement
- Satellite data integration
- Cloud-based deployment

---

## 15. References
- OpenWeather API Documentation  
- OpenStreetMap (OSM)  
- Scikit-learn Documentation  
- XGBoost Documentation  
