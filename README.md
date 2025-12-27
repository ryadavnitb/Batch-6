# 🌍 EnviroScan:AI-Powered Pollution Source Identifier using Geospatial Analytics

## Author
**Pawan Hingane**  
B.Tech (Engineering)  
VIT Bhopal University  

---

## 1. Introduction

Air pollution has emerged as a critical environmental and public health challenge worldwide. Identifying the dominant **source of pollution**—such as vehicular emissions, industrial activity, or natural contributors—is essential for informed decision-making, mitigation strategies, and policy development.

This project presents a **machine learning–based approach** to classify pollution sources by integrating **air quality data, meteorological conditions, and spatial proximity features**. The primary focus is on developing a **realistic, robust, and deployable system**, prioritizing generalization and reliability over artificially inflated performance metrics.

---

## 2. Problem Statement

Given environmental data collected from multiple heterogeneous sources, the objective is to **predict the primary pollution source (`pollution_source`)** affecting a given geographical location.

### Objectives
- Integrate air quality, weather, and spatial data from multiple sources  
- Perform systematic data preprocessing and feature engineering  
- Train and evaluate multiple machine learning classification models  
- Select a stable and interpretable model for deployment  
- Ensure realistic performance while avoiding overfitting  

---

## 3. Data Sources

### 3.1 Weather Data
- **Source:** OpenWeather API  
- **Features:** Temperature, humidity, wind speed, and general weather conditions  

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
  - Distance to industrial regions  
  - Distance to high-traffic zones  

All datasets were merged using spatial coordinates and timestamps to form a unified dataset suitable for modeling.

---

## 4. Data Preprocessing and Cleaning

### 4.1 Data Cleaning
- Standardized column names for consistency  
- Removed invalid records and missing target labels  
- Handled missing numerical values using median imputation  
- Handled missing categorical values using mode imputation  

### 4.2 Encoding
- Label encoding applied to categorical variables  
- Target variable (`pollution_source`) encoded numerically  

### 4.3 Feature Scaling
- Numerical features standardized using `StandardScaler` to ensure uniform model input  

---

## 5. Feature Engineering

To enhance model performance and interpretability, several derived features were created:

- **Traffic Pollution Index:** Mean of CO and NO₂ AQI values  
- **Particulate Ratio:** Ratio of PM2.5 AQI to overall AQI  
- **Heat–Humidity Index:** Product of temperature and humidity  
- **Spatial Proximity Features:**  
  - Distance to road (km)  
  - Distance to industrial area (km)  
  - Distance to traffic hotspot (km)  

To better reflect real-world sensor behavior and uncertainty, controlled **sensor noise** and approximately **18% label noise** were introduced into the dataset.

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
- Minimum sample constraints for splits  
- Balanced class weights  

This model was selected for deployment due to its **stable performance, resistance to overfitting, and strong generalization capability**.

### 7.2 Decision Tree
- Used primarily for interpretability  
- Hyperparameters optimized to reduce overfitting  

### 7.3 XGBoost
- Gradient boosting–based ensemble model  
- Used strictly for comparative performance analysis  

---

## 8. Hyperparameter Tuning

Hyperparameter optimization was performed using **RandomizedSearchCV**, primarily on the Decision Tree model.  
The Random Forest configuration was intentionally kept unchanged to preserve baseline consistency and avoid unnecessary complexity.

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
- XGBoost: Comparable or marginally higher accuracy  

The results indicate a strong balance between performance and generalization.

---

## 10. Overfitting Analysis

Overfitting was mitigated through:
- Controlled model depth and minimum sample constraints  
- Introduction of realistic noise into features and labels  
- Consistent performance across training, validation, and test sets  

The similarity in evaluation metrics across datasets confirms that the selected model does **not exhibit overfitting**.

---

## 11. System Architecture

### System Workflow
1. Data collection from APIs and curated datasets  
2. Data preprocessing and cleaning  
3. Feature engineering and transformation  
4. Model training and evaluation  
5. Model serialization using Joblib  
6. Integration with a prediction dashboard  

---

## 12. Deployment

- **Random Forest** model selected for production use  
- Saved artifacts include:
  - `pollution_rf_realistic.pkl`
  - `scaler.pkl`
  - `target_encoder.pkl`  

Other trained models are retained for experimentation and comparison purposes.

---

## 13. Conclusion

This project demonstrates an end-to-end **machine learning pipeline for pollution source classification**, integrating environmental, meteorological, and spatial data. By emphasizing realism, interpretability, and generalization, the system provides reliable predictions suitable for practical deployment and future extension.

---

## 14. Future Scope
- Real-time data ingestion and streaming  
- Advanced geospatial visualization and heatmaps  
- Integration of satellite-based pollution data  
- Cloud-based scalable deployment  

---

## 15. References
- OpenWeather API Documentation  
- OpenStreetMap (OSM)  
- Scikit-learn Documentation  
- XGBoost Documentation  
