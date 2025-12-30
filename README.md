# EnviroScan: AI-Powered Pollution Source Identifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

EnviroScan is an AI-based system designed to identify sources of air pollution using machine learning, geospatial data, and real-time environmental inputs. It classifies pollution into five categories (Vehicular, Industrial, Agricultural, Burning, Natural) and provides tools for visualization, alerts, and decision-making to support urban planning and policy.

This project was developed as part of an 8-week initiative under the mentorship of Rahul sir, presented by S. Kalinga, a 3rd-year CSE student at VIT-AP University.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [System Architecture](#system-architecture)
- [Data Collection & Pre-processing](#data-collection--pre-processing)
- [Feature Engineering & Source Labeling](#feature-engineering--source-labeling)
- [Model Training & Tuning](#model-training--tuning)
- [Dashboard & Visualization](#dashboard--visualization)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Conclusion](#conclusion)
- [References](#references)
- [Contact](#contact)

## Problem Statement

Existing pollution monitoring systems typically measure pollutant levels but fail to identify specific sources. This limits the ability of authorities and urban planners to take targeted and effective actions. Authorities know how bad pollution is, but not where it comes from.

### Why It Matters
- Approximately 7 to 9 million deaths occur every year globally due to air pollution.
- Approximately 46% of the population in India lives in areas exceeding WHO air quality guidelines for particulate matter (PM2.5) levels.

## Objectives

- Classify pollution into 5 sources: Vehicular, Industrial, Agricultural, Burning, Natural.
- Generate real-time pollution heatmaps.
- Trigger alerts for high-risk zones.
- Support urban planning & policy decisions.
- Provide downloadable reports.

## System Architecture

The system integrates data from multiple sources into a unified pipeline:
<img width="2154" height="601" alt="image" src="https://github.com/user-attachments/assets/0460396f-6b50-4ba2-ae13-06bc9f35771b" />

- **Inputs**:
  - Pollution Sensors/OpenAQ (PM2.5, PM10, NO₂, SO₂, O₃, CO).
  - Weather API: Wind, Temperature.
  - OSM Location DB: Roads, Industries.

- **Processing**:
  - Feature Engineering.
  - Cleaned Data.
  - ML Classifier (RandomForest/XGBoost).

- **Outputs**:
  - Prediction Score + Confidence.
  - Streamlit Dashboard: Map, Charts, Alerts.

### Technologies Used
- **Backend**: Python, Scikit-learn, XGBoost, Pandas, NumPy.
- **APIs**: OpenWeatherMap, OpenAQ, OpenStreetMap.
- **Frontend**: Streamlit, Folium, Plotly.

## Data Collection & Pre-processing

### Data Sources
1. **Air Quality (OpenAQ)**: PM2.5, PM10, NO₂, SO₂, O₃, CO.
2. **Weather (OpenWeatherMap)**: Temperature, Humidity, Wind Speed & Direction.
3. **Geospatial (OpenStreetMap via OSMnx)**: Roads, Industries, Farmland, Dump Sites.

Total datapoints: 14,324.

### Pre-processing Steps
1. Handled missing values & duplicates.
2. Detection and removal of outliers.
3. Standardized timestamps & coordinates.

## Feature Engineering & Source Labeling

### Features Created
- Distance to road, industry, farm, dump.
- Hour, day, month, season.

### Source Labeling (Rule-Based)
- **Vehicular**: High NO₂ + near roads (1.5 KM).
- **Industrial**: High SO₂ + near factories (3 KM).
- **Agricultural**: Seasonal PM + farmland proximity (5 KM).
- **Burning**: Extreme PM2 levels + CO (5 KM).
- **Natural**: Low pollution + remote locations.
<img width="736" height="587" alt="image" src="https://github.com/user-attachments/assets/febcdc20-df73-453a-94a8-4bb947cf66f3" />

## Model Training & Tuning

### Data Split
- Training Set: 80% (~11,500 samples).
- Test Set: 20% (~2,865 samples).

### Models Tested & Results
- XGBoost Classifier.
- Random Forest Classifier.
- Decision Tree Classifier.

**Best Model**: XGBoost  
**Accuracy**: 94%  
<img width="1152" height="922" alt="image" src="https://github.com/user-attachments/assets/cace627e-14b4-4290-b449-135c4608e162" />


Hyper-parameter Tuning: GridSearchCV.

## Dashboard & Visualization

- Location-based pollution prediction.
- Interactive heatmaps (Folium).
- Real-time pollution alerts.
- Trend charts & source distribution.
- Downloadable reports (JSON/CSV).
- 

## Results

The XGBoost model outperformed others with 94% accuracy. Key metrics from model comparison (Accuracy, Precision, Recall, F1-Score) show XGBoost as superior across the board.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/kalingasajja/EnviroScan.git
   cd EnviroScan
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   (Assumed dependencies: pandas, numpy, scikit-learn, xgboost, streamlit, folium, plotly, osmnx, requests)

3. Set up API keys:
   - Obtain keys for OpenAQ, OpenWeatherMap, and configure them in a `.env` file.

## Usage

1. Run the dashboard:
   ```
   streamlit run app.py
   ```

2. Input location or use real-time data to predict pollution sources.
3. View heatmaps, alerts, and download reports.

## Conclusion

I successfully built an AI-based system to identify pollution sources using geospatial and environmental data by integrating real-time pollution, weather, and location features into a unified data pipeline. I trained and evaluated multiple machine learning models, with XGBoost performing as the best model, delivering accurate and reliable predictions. I also developed interactive maps and a real-time dashboard to visualize pollution hotspots and risk zones effectively. Overall, I achieved the project goals and demonstrated how AI can support data-driven environmental decision-making.

## References

- OpenAQ API Documentation
- OpenWeatherMap API Documentation
- OpenStreetMap Contributors
- Scikit-learn Documentation
- Streamlit Documentation
- World Health Organization (Air Quality Guidelines)

## Contact

- **Author**: S. Kalinga
- **GitHub**: [kalingasajja](https://github.com/kalingasajja)
- **University**: VIT-AP University
- **Mentor**: Rahul sir

For questions or contributions, open an issue or pull request!
