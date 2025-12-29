# 🌍 EnviroScan  
**AI-Powered Pollution Source Identification using Geospatial Analytics**

This project provides a **Streamlit web application** combined with a full machine learning pipeline to identify likely sources of air pollution across global locations. Using real-time air quality and weather data from OpenWeatherMap, it computes a custom Pollution Risk Index (PRI), predicts dominant pollution sources, and visualizes high-risk zones on interactive maps.

---
## 🚀 Live App
Try the app online:  
[Streamlit Community Cloud](https://infosys-assignments-at2uqfsa2ibrlg82xqtwwq.streamlit.app/)  
*(Link to be updated after deployment)*

---
## 📝 Features
- Global dataset of ~22,000 locations enriched with real-time pollutant and weather data
- Custom **Pollution Risk Index (PRI)** calculation (health-weighted + weather-adjusted)
- Pollution **severity classification** (Low / Moderate / High / Severe) with confidence scores
- **Source prediction** using trained ML models (vehicular, industrial, agricultural, natural/mixed)
- Interactive **geospatial heatmaps** and filters (by country, severity, source)
- Model comparison (Decision Tree, Random Forest, XGBoost)
- Exportable insights and visualizations

---

## 📁 Project Structure
```
enviroscan/
│
├─ prototype.py # Streamlit main app
├─ requirements.txt # Python dependencies
├─ pollution.csv
├─ README.md
└─ models 
    ├─ xgb_model.joblib
    ├─rf_model.joblib
    └─ dt_model.onlib
```
---

## 🛠 How to Run Locally
1. Clone the repository:
```bash
git clone https://github.com/your-username/EnviroScan.git
cd EnviroScan

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt

streamlit run app/app.py
```

## ⚙️ Requirements

Python 3.12+ (recommended for Streamlit Cloud compatibility)
Python 3.10+
streamlit
pandas, numpy
scikit-learn, xgboost
matplotlib, seaborn, plotly, folium
requests, aiohttp

## 📂 Model Options
Pollution source and severity prediction using tree-based models
Trained on engineered features from OpenWeatherMap data
Best model saved and loaded in the Streamlit app

# ⚠️ Disclaimer
This tool is intended for educational and research purposes only.
Pollution source predictions are based on heuristic rules and machine learning models trained on simulated patterns.
Results are indicative and should not be used for regulatory, legal, or policy decisions.
Always refer to official environmental monitoring agencies for accurate data.

## 👨‍💻 Author  
Name: Anish Ravi Nadar  
GitHub: https://github.com/AnishRN  
Academic Project – AI & Geospatial Environmental Analytics  

## ⭐ Acknowledgements  

WHO, CPCB India, US EPA  
OpenWeatherMap  
OpenStreetMap Contributors  
scikit-learn, pandas, Streamlit, Folium communities  

---

