# 🌍 EnviroScan – AI-Powered Pollution Source Identification

EnviroScan is an end-to-end data analytics and machine learning project that identifies **pollution sources** using air quality, weather, and geospatial data. The system classifies pollution into multiple categories and provides interactive visualizations through a Streamlit dashboard.

---

## 📌 Problem Statement
Air quality data such as AQI is widely available, but identifying the **source of pollution** (vehicular, industrial, agricultural, etc.) is challenging. Without source identification, effective mitigation and policy decisions become difficult.

---

## 🎯 Project Objectives
- Identify pollution sources using AI techniques  
- Analyze air quality, weather, and geospatial data  
- Classify pollution into multiple source categories  
- Visualize pollution patterns using maps and dashboards  
- Provide alerts and downloadable reports  

---

## 📊 Datasets Used
- Global Air Pollution Dataset  
- Pollution and Weather Dataset (sourced from OpenWeatherMap API as static data)

### Dataset Statistics
- **Total Records:** 21,882  
- **Countries Covered:** 175  
- **Granularity:** City-level and location-based  

### Features
- AQI Value  
- PM2.5, NO₂, CO, O₃  
- Temperature, Humidity, Wind Speed  
- Latitude, Longitude, Timestamp  

---

## 🧹 Data Preprocessing
- Handled missing values and inconsistencies  
- Merged pollution and weather datasets  
- Converted timestamps to datetime format  
- Performed feature engineering (time-based and geospatial features)

---

## 🧠 Pollution Source Labeling
Since ground-truth pollution source labels were not available, **heuristic-based rules** were applied using domain knowledge.

### Pollution Categories
- Vehicular  
- Industrial  
- Agricultural  
- Residential  
- Natural  

Labeling was based on pollutant concentrations, weather context, and geospatial proximity.

---

## 🤖 Model Training & Evaluation
- **Model Used:** Random Forest Classifier  
- **Train–Test Split:** 80% training, 20% testing  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score  
- **Result:** Achieved approximately **90% realistic accuracy**

---

## 🖥️ Dashboard & Visualization
- Built using **Streamlit**
- Interactive filters (country, city, source, date range)
- Charts for pollution trends and source distribution
- Geospatial maps and heatmaps using Folium
- Real-time alert simulation
- Downloadable CSV reports

---

## 🌐 Live API Integration
- OpenWeatherMap API used to fetch **live air pollution data**
- Live data is processed and used for real-time pollution source prediction
- Static datasets were used for training to ensure reproducibility

---

## 🚀 How to Run the Project
```bash
pip install -r requirements.txt
streamlit run app.py
