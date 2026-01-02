# 🌍 EnviroScan-pro: AI-Powered Pollution Source Identifier using Geospatial Analytics

An AI-driven system to identify and visualize pollution sources using air quality data, weather parameters, and geospatial analytics.

---

## 👤 Author
**Jeeva J**  
B.Tech (Artifical Intelligence and Data Scenice)  
Gnanamani College of Technology  

**Project Type:** Infosys Springboard Project  

---

## 1. Introduction
Air pollution has emerged as a critical environmental and public health challenge worldwide. Identifying the source of pollution—such as vehicular emissions, industrial activity, or natural contributors—is essential for informed decision-making, mitigation strategies, and policy development.

This project presents a **machine learning–based approach** to classify pollution sources by integrating:
- Air quality data  
- Meteorological conditions  
- Spatial proximity features  
- Interactive geospatial visualization  

The system is designed to be **realistic, robust, and deployable**, prioritizing **generalization and interpretability** over artificially inflated performance metrics.

---

## 2. Problem Statement
Given environmental data collected from multiple heterogeneous sources, the objective is to **predict the primary pollution source** affecting a given geographical location and **visually represent pollution patterns** for decision support.

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
**Source:** OpenWeather API  
**Features:**
- Temperature  
- Humidity  
- Wind speed  
- Weather description  

### 3.2 Air Pollution Data
**Source:** Dataset provided by *Ankit Sir*  
**Features:**
- Carbon Monoxide (CO) AQI  
- Nitrogen Dioxide (NO₂) AQI  
- Ozone (O₃) AQI  
- Particulate Matter (PM2.5) AQI  
- Overall Air Quality Index (AQI)  

### 3.3 Physical and Proximity Features
**Source:** OpenStreetMap (OSM)  
**Features:**
- Distance to nearest road  
- Distance to industrial areas  
- Distance to traffic hotspots  

All datasets were merged using **spatial coordinates and timestamps** to create a unified dataset.

---

## 4. Data Preprocessing and Cleaning

### 4.1 Data Cleaning
- Standardized column names  
- Removed invalid records and missing target labels  
- Median imputation for missing numerical values  
- Mode imputation for missing categorical values  

### 4.2 Encoding
- Label encoding for categorical features  
- Label encoding for the target variable  

### 4.3 Feature Scaling
- Standardization using `StandardScaler` for uniform model input  

---

## 5. Feature Engineering
The following engineered features were created to enhance predictive performance:

- **Traffic Pollution Index:** Average of CO and NO₂ AQI  
- **Particulate Ratio:** PM2.5 AQI relative to overall AQI  
- **Heat–Humidity Index:** Combined effect of temperature and humidity  
- **Spatial Proximity Features:**
  - Distance to road (km)
  - Distance to industrial area (km)
  - Distance to traffic hotspot (km)

To simulate real-world conditions:
- Controlled sensor noise was added  
- ~18% label noise was introduced  

---

## 6. Machine Learning Models

| Model          | Purpose |
|----------------|--------|
| Random Forest  | Baseline & deployed production model |
| Decision Tree  | Interpretable comparison model |
| XGBoost        | Advanced benchmarking model |

---

## 7. Model Architecture and Training

### 7.1 Random Forest (Baseline Model)
- Depth-limited decision trees  
- Minimum sample constraints  
- Balanced class weights  

**Selected for deployment** due to its robustness, stability, and strong generalization capability.

### 7.2 Decision Tree
- Used primarily for interpretability  
- Hyperparameters tuned to reduce overfitting  

### 7.3 XGBoost
- Gradient boosting ensemble model  
- Used strictly for performance comparison  

---

## 8. Hyperparameter Tuning
- RandomizedSearchCV applied mainly to the Decision Tree  
- Random Forest parameters intentionally preserved to maintain baseline consistency  

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
- **Decision Tree:** Lower accuracy (expected due to simplicity)  
- **XGBoost:** Comparable or slightly higher accuracy  

---

## 10. Overfitting Analysis
Overfitting was controlled using:
- Depth constraints  
- Minimum sample thresholds  
- Noise injection  
- Cross-validation consistency  

Comparable training and test performance confirms **no overfitting**.

---

## 11. Geospatial Visualization and Mapping
To enhance interpretability and decision support:

- Predictions plotted on interactive maps  
- **Folium** used for pollution heatmaps  
- Source-specific markers (industrial, vehicular, etc.)  
- High-risk zones visualized using color gradients  

Filters enabled:
- Date  
- Location  
- Predicted pollution source  

Maps are embedded directly into the web dashboard.

---

## 12. System Architecture
**Workflow:**
1. Data collection from APIs and curated datasets  
2. Data preprocessing and cleaning  
3. Feature engineering  
4. Model training and evaluation  
5. Prediction generation  
6. Geospatial visualization  
7. Dashboard integration  
8. Model serialization using Joblib  

---

## 13. Deployment
- **Random Forest** selected as production model  
- Saved artifacts:
  - `pollution_rf_realistic.pkl`
  - `scaler.pkl`
  - `target_encoder.pkl`  

Other models retained for experimentation and benchmarking.

---

## 14. Conclusion
EnviroScan delivers an **end-to-end pollution source classification system** combining machine learning with geospatial analytics. The project emphasizes realistic data handling, robust modeling, and intuitive visualization, making it suitable for real-world environmental monitoring and decision support.

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

---

⭐ *This project was developed as part of the Infosys Springboard learning initiative.*
