 EnviroScan: AI-Powered Pollution Source Identifier


Real-time AI system that identifies pollution sources with 94% accuracy using machine learning and geospatial analytics

Traditional pollution monitoring systems measure how bad the air quality is, but fail to identify where the pollution comes from. EnviroScan solves this critical gap by using XGBoost machine learning to classify pollution into 5 distinct sources: Vehicular, Industrial, Agricultural, Burning, and Natural.

📋 Table of Contents

Problem Statement
Key Features
System Architecture
Technologies Used
Installation
Usage
Model Performance
Project Structure
API Configuration
Screenshots
Contributing
License
Acknowledgments
Contact


🎯 Problem Statement
The Challenge

7-9 million deaths occur annually due to air pollution globally
46% of India's population lives in areas exceeding WHO air quality guidelines
Current monitoring systems measure pollutant levels but fail to identify specific sources
Authorities and urban planners lack actionable data for targeted interventions

Our Solution
EnviroScan leverages machine learning, weather data, and geospatial analytics to:

✅ Predict the source of pollution (not just the level)
✅ Generate real-time pollution heatmaps
✅ Trigger alerts for high-risk zones
✅ Support data-driven policy decisions


✨ Key Features
FeatureDescription🤖 AI-Powered ClassificationIdentifies pollution sources with 89.88% accuracy using XGBoost🗺️ Interactive MapsReal-time pollution heatmaps with source markers using Folium📊 Live DashboardStreamlit-based interface with charts, alerts, and analytics🌐 API IntegrationFetches live weather and air quality data from OpenWeatherMap🚨 Smart AlertsThreshold-based risk warnings (Safe/Moderate/Danger)📥 Export ReportsDownloadable CSV/JSON reports for authorities🎯 5-Class PredictionVehicular, Industrial, Agricultural, Burning, Natural⚡ Real-Time AnalysisPrediction time < 0.1 seconds

🏗️ System Architecture
┌─────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION                          │
├─────────────────┬───────────────────┬──────────────────────────┤
│  OpenWeatherMap │     OpenAQ API    │   OpenStreetMap (OSM)   │
│  - Temperature  │   - PM2.5, PM10   │   - Roads proximity     │
│  - Humidity     │   - NO₂, SO₂      │   - Industry distance   │
│  - Wind Speed   │   - O₃, CO        │   - Farm locations      │
└─────────────────┴───────────────────┴──────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                           │
│  17 Features: Pollutants + Weather + Proximity + Temporal      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MACHINE LEARNING MODEL                       │
│         XGBoost Classifier (89.88% Accuracy)                    │
│         413 trees | Optimized hyperparameters                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   STREAMLIT DASHBOARD                           │
│  Interactive Maps | Charts | Alerts | Reports                  │
└─────────────────────────────────────────────────────────────────┘

🛠️ Technologies Used
Backend & ML

Python 3.8+ - Core programming language
XGBoost 2.0+ - Gradient boosting for classification
Scikit-learn - Data preprocessing and model evaluation
Pandas & NumPy - Data manipulation and analysis
Joblib - Model serialization

APIs & Data Sources

OpenWeatherMap API - Live weather and air quality data
OpenAQ API - Historical pollution data
OpenStreetMap (OSMnx) - Geospatial features

Frontend & Visualization

Streamlit 1.32+ - Interactive web dashboard
Folium - Interactive maps and heatmaps
Plotly - Dynamic charts and graphs
Streamlit-Folium - Map integration


📦 Installation
Prerequisites

Python 3.8 or higher
pip package manager
OpenWeatherMap API key (free tier available)

Step 1: Clone the Repository
bashgit clone https://github.com/YOUR_USERNAME/enviroscan.git
cd enviroscan
Step 2: Create Virtual Environment (Recommended)
bashpython -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
Step 3: Install Dependencies
bashpip install -r requirements.txt
Step 4: Download/Place Model File
Ensure enviroscan_final_model.joblib is in the project root directory.
Step 5: Configure API Key

Sign up for free at OpenWeatherMap
Get your API key
Either:

Enter it in the sidebar when running the app, OR
Edit dashboard.py and replace YOUR_API_KEY_HERE




🚀 Usage
Running the Dashboard
bashstreamlit run dashboard.py
The dashboard will open in your browser at http://localhost:8501
Using the Application

Select Location

Click anywhere on the map to select a target location
Or manually enter coordinates in the sidebar


Fetch Data (if API configured)

Live weather and air quality data will be fetched automatically
Falls back to demo mode if API unavailable


Analyze

Click the "🚀 Analyze Location" button
Wait for AI processing (~2-3 seconds)


View Results

Pollution source prediction with confidence score
Interactive heatmap showing pollution spread
Risk level assessment (Safe/Moderate/Danger)
Pollutant concentrations and probability distribution


Export Report

Click "📥 Download Report" to save CSV file
Includes 7-day historical trend data




📊 Model Performance
Overall Metrics
MetricScoreAccuracy89.88%Precision (weighted)90.18%Recall (weighted)89.88%F1-Score (weighted)89.89%Prediction Time< 0.1s
Per-Class Performance
SourcePrecisionRecallF1-ScoreSupportNatural99%98%98%1,048Agricultural97%90%93%234Burning90%86%88%365Vehicular80%91%85%706Industrial84%76%80%512
Hyperparameter Optimization

Method: RandomizedSearchCV with 3-fold CV
Configurations Tested: 50 random combinations
Best Parameters:

python  n_estimators: 413
  learning_rate: 0.0464
  max_depth: 7
  subsample: 0.81
  colsample_bytree: 0.93

Improvement: +6-8% accuracy gain from tuning

Dataset Statistics

Total Samples: ~14,325
Training Set: 11,460 samples (80%)
Test Set: 2,865 samples (20%)
Features: 17 engineered features
Classes: 5 pollution sources


📁 Project Structure
enviroscan/
├── dashboard.py                    # Main Streamlit application
├── enviroscan_final_model.joblib   # Trained XGBoost model
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── LICENSE                         # MIT License
│
├── notebooks/                      # Jupyter notebooks (optional)
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_hyperparameter_tuning.ipynb
│
├── data/                           # Data directory (not tracked)
│   ├── raw/
│   └── processed/
│
└── docs/                           # Documentation
    ├── system_architecture.png
    ├── confusion_matrix.png
    └── presentation.pdf

🔑 API Configuration
OpenWeatherMap API
Get Your Free API Key

Visit https://openweathermap.org/api
Sign up for a free account
Navigate to "API Keys" section
Copy your API key

Configure in Application
Option 1: Runtime Configuration (Recommended)

Enter API key in the sidebar when running the app

Option 2: Hardcode in Script
Edit dashboard.py:
pythonOPENWEATHER_API_KEY = "your_actual_api_key_here"
API Endpoints Used

Current Weather: /data/2.5/weather
Air Pollution: /data/2.5/air_pollution

Rate Limits (Free Tier)

1,000 API calls per day
60 calls per minute


📸 Screenshots
Dashboard Overview
Show Image
Pollution Heatmap
Show Image
Source Classification
Show Image
Analytics Charts
Show Image

🤝 Contributing
Contributions are welcome! Here's how you can help:
Reporting Bugs

Use GitHub Issues to report bugs
Include screenshots and error messages
Describe steps to reproduce

Suggesting Enhancements

Open an issue with the "enhancement" label
Describe the feature and its benefits
Provide examples if possible

Pull Requests

Fork the repository
Create a feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

Development Setup
bash# Clone your fork
git clone https://github.com/YOUR_USERNAME/enviroscan.git

# Create branch
git checkout -b feature/your-feature

# Make changes and test
streamlit run dashboard.py

# Commit and push
git add .
git commit -m "Your descriptive commit message"
git push origin feature/your-feature

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
MIT License

Copyright (c) 2024 S. Kalinga

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

🙏 Acknowledgments
Data Sources

OpenWeatherMap - Weather and air quality data
OpenAQ - Open air quality data platform
OpenStreetMap - Geospatial features

Libraries & Frameworks

XGBoost - Gradient boosting framework
Scikit-learn - Machine learning library
Streamlit - Dashboard framework
Folium - Interactive maps

Guidance

Project Mentor: Rahul Sir
Institution: VIT-AP University
Duration: 8 weeks internship project

References

World Health Organization - Air Quality Guidelines
Research papers on air pollution source apportionment
Various online resources and documentation


📬 Contact
S. Kalinga
3rd Year Computer Science Engineering
VIT-AP University

GitHub: @YOUR_GITHUB_USERNAME
Email: your.email@example.com
LinkedIn: Your LinkedIn Profile

Project Link: https://github.com/YOUR_USERNAME/enviroscan

🌟 Star This Repository
If you find this project useful, please consider giving it a ⭐! It helps others discover the project.

🚀 Future Enhancements

 Mobile application (Android/iOS)
 Historical data analysis and forecasting
 Multi-city comparison tool
 Email/SMS alert notifications
 Integration with government pollution APIs
 Deep learning models (LSTM for time-series)
 Satellite imagery integration
 User authentication and personalized dashboards
 RESTful API for third-party integration
 Deployment on cloud platforms (AWS/Azure/GCP)


📈 Project Impact
Potential Applications

Smart Cities: Real-time pollution source tracking
Urban Planning: Data-driven infrastructure decisions
Environmental Compliance: Industrial monitoring
Public Health: Location-specific air quality advisories
Policy Making: Evidence-based emission regulations

Success Metrics

89.88% source classification accuracy
<0.1s prediction latency
Supports 5 distinct pollution categories
Processes 17 environmental features
Trained on 14,325+ real-world data points


<div align="center">
e
From data to decisions - making air quality actionable
</div>
