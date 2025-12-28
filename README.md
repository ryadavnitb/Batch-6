# 🌍 EnviroScan AI-Powered Pollution Source Identifier using Geospatial Analytics

EnviroScan is an **AI-based environmental monitoring system** that not only measures pollution levels but also **identifies the most likely sources of pollution** using **Machine Learning, Weather Data, and Geospatial Analytics**.  
The system visualizes pollution hotspots, predicts risk zones, and provides alerts to support **data-driven environmental decision-making**.

---

## 📌 Project Statement

Traditional pollution monitoring systems focus only on pollutant concentration values and **do not identify pollution sources**, limiting effective intervention by authorities.

This project leverages:

- Machine Learning models  
- Weather parameters  
- Spatial proximity and geolocation features  

to **predict pollution sources** such as **industrial activity, vehicular traffic, agricultural burning, or natural causes**, generate **geospatial heatmaps**, and trigger **alerts for high-risk zones**.

---

## 🎯 Project Outcomes

- Predict likely **sources of pollution** (industrial, vehicular, agricultural, natural)
- Display **real-time pollution hotspots and risk zones**
- Trigger **pollution alerts** based on threshold exceedance
- Support **urban planning and environmental policy-making**
- Generate **reports and visual analytics** for agencies

---

## 📁 Main Project File

```

AI-Powered Pollution Source Identifier using Geospatial Analytics.ipynb

```

### Description

This notebook is the **main implementation file** of the project and contains:

- Complete data preprocessing pipeline  
- Feature engineering and spatial analysis  
- Pollution source labeling logic  
- Machine learning model training  
- Model evaluation and comparison  
- Visual outputs and analysis  

---

## 🧠 System Architecture

**Input → Processing → Prediction → Visualization**

- Pollution Data (OpenAQ API)  
- Weather Data (OpenWeatherMap API)  
- Location Features (OpenStreetMap / OSMnx)  
- Feature Engineering and Source Labeling  
- Machine Learning Models (Random Forest, Decision Tree, XGBoost)  
- Dashboard and Heatmap Visualization  

---

## 🔁 Data Flow and Machine Learning Workflow

1. Collect pollution, weather, and location data  
2. Clean and normalize datasets  
3. Engineer spatial and temporal features  
4. Label pollution sources using heuristic rules  
5. Train machine learning models  
6. Predict pollution sources  
7. Visualize results on maps and dashboards  

---

## 🧩 Modules Implemented

### Module 1: Data Collection
- Air Quality: PM2.5, PM10, NO₂, CO, SO₂, O₃  
- Weather: Temperature, Humidity, Wind Speed  
- Location features using OpenStreetMap  
- Data stored in CSV/JSON format  

### Module 2: Data Cleaning and Feature Engineering
- Duplicate and missing value handling  
- Data normalization  
- Spatial distance calculations  
- Temporal feature extraction  

### Module 3: Source Labeling and Simulation
Rule-based labeling:
- High NO₂ + proximity to roads → **Vehicular**
- High SO₂ + proximity to industries → **Industrial**
- High PM + farmland during dry season → **Agricultural**
- Background conditions → **Natural**

### Module 4: Model Training and Prediction
Models used:
- Random Forest  
- Decision Tree  
- XGBoost (**Best Performing Model**)  

Evaluation Metrics:
- Accuracy  
- Precision  
- Recall  
- F1-score  

### Module 5: Geospatial Mapping and Heatmap Visualization
- Interactive Folium maps  
- Pollution intensity heatmaps  
- Source-based markers  
- Location and date filtering  

### Module 6: Real-Time Dashboard and Alerts
- Streamlit-based interactive dashboard  
- Pollution predictions with confidence scores  
- Trend charts and pie charts  
- Heatmap overlays  
- Alert notifications  
- Downloadable reports  

---

## 🗂️ Project Structure

```

EnviroScan/
│
├── AI-Powered Pollution Source Identifier using Geospatial Analytics.ipynb
├── Dashboard/
│   └── app.py
├── Data/
│   ├── final_labeled_dataset.csv
│   ├── india_air_quality.csv
│   ├── india_weather.csv
│   ├── india_locations.csv
│   ├── india_features.csv
│   └── india_merged_all_rows_columns.csv
├── Scripts/
│   ├── pollution.py
│   ├── weather.py
│   ├── locations.py
│   ├── features.py
│   └── merged_core.py
├── Model_Dataset/
├── pollution_dashboard_map.html
├── requirements.txt
├── LICENSE
└── README.md

````

---

## 🖥️ Dashboard Features

- Pollution prediction results  
- Heatmaps of high-risk zones  
- Source distribution charts  
- Trend analysis over time  
- Alert notifications  

---

## ⚙️ Installation and Execution

### Clone Repository
```bash
git clone https://github.com/your-username/EnviroScan.git
cd EnviroScan
````

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Main Notebook

```bash
jupyter notebook
```

Open:

```
AI-Powered Pollution Source Identifier using Geospatial Analytics.ipynb
```

### Run Dashboard

```bash
streamlit run Dashboard/app.py
```

---

## 📈 Results and Insights

* XGBoost achieved the **highest accuracy**
* Weather and spatial features strongly influence pollution sources
* Heatmaps clearly identify **pollution hotspots**
* The system enables **actionable environmental insights**

---

## 🚀 Future Enhancements

* Real-time API integration
* Satellite data analysis
* Deep learning models
* Pollution forecasting
* Mobile dashboard support

---

## 👩‍💻 Author

**Likhitha**
B.Tech (Engineering)

---
