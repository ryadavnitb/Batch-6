🌍 EnviroScan: AI-Based Pollution Source Attribution Using Geospatial Analytics

EnviroScan is an AI-powered environmental intelligence system that goes beyond traditional air quality monitoring.
Instead of only reporting how polluted an area is, EnviroScan identifies where the pollution is coming from — such as vehicular traffic, industrial activity, agricultural burning, or natural background sources — using machine learning and geospatial analytics.

This project was developed as part of the Infosys Springboard Virtual Internship 6.0 to address real-world environmental monitoring challenges.

🚀 Key Objectives

Move from pollution measurement to pollution source attribution

Combine air quality indicators with geospatial proximity features

Use machine learning to classify dominant pollution sources

Provide an interactive, visual dashboard for analysis and decision-making

🧠 System Overview

The system follows a modular and scalable pipeline:

Data Processing

Loads air quality and weather data

Simulates realistic geospatial features (distance to roads, industries, agriculture)

Applies rule-based logic to label pollution sources

Machine Learning Engine

Trains a Random Forest classifier

Evaluates model accuracy and class-wise performance

Saves trained models for reuse (.pkl files)

Interactive Web Application

Built using Streamlit

Provides live geospatial maps, analytics dashboards, and AI predictions

Supports manual sensor input for forensic source analysis

🗂️ Project Structure
EnviroScan/
│
├── dataset/
│   └── Pollution_Weather_datset.csv
│
├── src/
│   ├── data_processor.py        # Data cleaning, feature engineering, source labeling
│   └── model_engine.py          # ML model training, evaluation, prediction
│
├── train_model.py               # Trains and saves the final ML model
├── app.py                       # Streamlit application
├── model.pkl                    # Internal Random Forest model
├── pollution_model.pkl          # Trained production-grade model
├── requirements.txt             # Project dependencies
└── README.md

🧪 Machine Learning Approach

Algorithm Used: Random Forest Classifier

Why Random Forest?

High accuracy compared to baseline models

Robust to noisy environmental data

Interpretable feature importance for policy insights

Features Used:

PM2.5, NO₂, CO, Ozone AQI values

Temperature, humidity, wind speed

Distance to roads, industries, agricultural zones

Target Classes:

Vehicular

Industrial

Agricultural

Natural / Background

📊 Application Features
🌍 Live Geospatial Map

Heatmaps showing pollution intensity

Clustered markers colored by pollution source

Region-based filtering

📈 Analytics Dashboard

Pollution source distribution

Weather vs pollution correlation

High-risk zone indicators

Summary KPIs

🤖 AI Source Predictor

Manual input of sensor values

Predicts pollution source with confidence score

Visual indicators for risk interpretation

🛠️ Tech Stack

Programming Language: Python

Libraries & Frameworks:

pandas, numpy

scikit-learn

Streamlit

Folium & streamlit-folium

Plotly

joblib

⚙️ Installation & Execution
1️⃣ Clone the Repository
git clone https://github.com/your-username/enviroscan.git
cd enviroscan

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Train the Model (Optional)
python train_model.py

4️⃣ Run the Application
streamlit run app.py

🎯 Use Cases

Smart city pollution monitoring

Environmental impact assessment

Policy-driven emission control

Urban planning and traffic regulation

Academic research and demonstrations

📌 Future Enhancements

Integration with real-time sensor APIs

Deep learning–based source attribution

Temporal forecasting of pollution spread

Government-grade alert and reporting system

👤 Author

Srujan D
Final-Year Engineering Student
Infosys Springboard Virtual Internship 6.0
