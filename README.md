🌍 AI-EnviroScan: Pollution Source Identification and Visualization System

📌 Author

Riya Verma

B-Tech Computer Science & Engineering (3rd Year)
NIST University, Berhampur, Odisha

🎓 Internship

Infosys SpringBoard Virtual Internship 6.0
Mentor: Rahul

1️⃣ Project Overview

AI-EnviroScan is an end-to-end Machine Learning and Geospatial Analytics system designed to identify pollution sources and visualize air quality patterns across Indian cities.

The project integrates air quality data, weather parameters, spatial proximity features, and machine learning models to classify pollution sources such as Vehicular, Industrial, Agricultural, Burning, and Natural.
It further provides interactive visualizations and a dashboard to help users understand pollution trends and risks.

2️⃣ Key Features

🔍 Pollution Source Identification using Machine Learning

🗺️ Spatial Analysis using OpenStreetMap (roads, industries, dumps, agriculture)

⏱️ Temporal Analysis (hourly, daily, seasonal trends)

🤖 Multiple ML Models (Random Forest, Decision Tree, XGBoost)

📊 Interactive Dashboard built with Streamlit

🔥 Geospatial Heatmaps using Folium

📄 Automated PDF Pollution Reports with health precautions

3️⃣ Tech Stack

Programming Language: Python

Data Handling: Pandas, NumPy

Machine Learning: Scikit-learn, XGBoost

Imbalanced Data Handling: SMOTE (imbalanced-learn)

Geospatial Processing: GeoPandas, Shapely, OSMnx

APIs & Data: OpenAQ API, OpenStreetMap (OSM)

Visualization: Matplotlib, Folium

Dashboard: Streamlit

4️⃣ Data Sources

1.OpenAQ API-

Air quality and meteorological data (PM2.5, PM10, NO₂, SO₂, CO, O₃, temperature, humidity, wind)

2.OpenStreetMap (OSM)-

Roads, industrial zones, dump sites, agricultural land features

5️⃣ Data Processing Pipeline

1.Data Collection

OpenAQ API queried for multiple Indian states and districts

2.Data Cleaning & Preprocessing

Invalid coordinates removed

Outliers filtered

Missing values handled using median imputation

3.Feature Engineering

Temporal features: hour, weekday, season

Spatial features: distance to roads, industries, dumps, agriculture

Normalization and standard scaling

4.Source Labeling

Rule-based heuristic labeling using pollutant thresholds and proximity features

5.Model Training & Evaluation

Multiple models trained and compared

SMOTE applied to handle class imbalance

6️⃣ Machine Learning Models

🌳 Random Forest (Final Selected Model)

🌲 Decision Tree

⚡ XGBoost

Final Model Selection:
Random Forest was selected based on Macro F1-score, stability, and interpretability.

7️⃣ Evaluation Metrics

Accuracy

Precision & Recall

Macro F1-score

Confusion Matrix

Cross-validation scores

⚠️ Note: High accuracy is expected since labels were generated using domain-driven heuristics. The objective was rule learning and automation, not pattern discovery.

8️⃣ Dashboard

The Streamlit dashboard provides:

City-wise filtering

Date range selection

Pollution trend analysis

Source distribution charts

Interactive heatmap visualization

Dummy live prediction demo

Downloadable PDF pollution report with precautions

9️⃣ Project Structure
AI-EnviroScan/
│

├── data/
│    ├── raw/
│    ├── processed/
│    └── osm_files/
│

├── notebooks/
│    ├── data_collection.ipynb
│    ├── preprocessing.ipynb
│    ├── feature_engineering.ipynb
│    ├── modeling.ipynb
│

├── models/
│    ├── pollution_source_random_forest_model.joblib
│

├── dashboard/
│  ├── module6_dashboard.py
│

├── maps/
│    ├── final_pollution_source_heatmap.html
│

├── reports/
│    ├── EnviroScan_Report.pdf
│

├── README.md
 └── requirements.txt

🔟 How to Run the Project

Step 1: Clone the Repository

git clone https://github.com/your-username/AI-EnviroScan.git

cd AI-EnviroScan

Step 2: Install Dependencies

pip install -r requirements.txt


Step 3: Run the Dashboard

streamlit run dashboard/module6_dashboard.py

1️⃣1️⃣ Results & Insights

Vehicular and industrial sources dominate urban pollution

PM2.5 and NO₂ strongly correlate with proximity to roads

Spatial proximity significantly improves source classification

Random Forest achieves the best balance of accuracy and robustness

1️⃣2️⃣ Future Enhancements

🔄 Real-time sensor data integration

📡 SMS / Email alert system

🧠 Deep learning models (LSTM for time-series forecasting)

☁️ Cloud deployment (Streamlit Cloud / Hugging Face Spaces)

1️⃣3️⃣ Acknowledgements

Infosys SpringBoard – Internship platform

Mentor: Rahul – Guidance and review

OpenAQ – Open air quality data

OpenStreetMap Contributors – Geospatial data
