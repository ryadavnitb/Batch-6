EnviroScan – Air Pollution Analysis

Milestone 1 & 2 – README

Project Overview
This project is part of the Infosys Springboard Internship – EnviroScan. The objective of the first two weeks is to load the dataset, clean it, explore it, generate visualizations, compare pollution data across different entities, and prepare the cleaned dataset for future modules.

Folder Structure
All files are stored in a single folder:

global_air_pollution_dataset.csv
milestone1_2.ipynb
README.md

No subfolders were created in this stage.

Dataset Information
The dataset contains air quality information for several cities across the world.
Important columns include:

Country

City

AQI Value

AQI Category

CO AQI Value

Ozone AQI Value

NO2 AQI Value

PM2.5 AQI Value

Each row represents AQI measurements for a specific city.

Module 1 (Week 1): Data Collection & Exploration

Loading the Dataset
The dataset was loaded using pandas, and the shape and first few rows were examined.

Dataset Information
Using the info() and describe() functions, we viewed the column data types, non-null counts, and statistical summaries.

Initial Visualizations

AQI Value Distribution Histogram

AQI Category Count Bar Chart

These visualizations helped understand the overall pollution pattern and category distribution.

Module 2 (Week 2): Data Cleaning & Preparation

Handling Missing Values
Rows containing any missing values were removed using dropna(). This ensures clean and consistent data for analysis.

Ensuring Numeric Data
Pollutant AQI columns were verified and converted to numeric types to avoid calculation issues.

Visual Comparisons
The following comparison visualizations were created:

Average AQI for each pollutant (CO, Ozone, NO2, PM2.5)

Top 15 most polluted cities

Top 15 most polluted countries

Correlation heatmap showing relationships between pollutant values

These visuals provide insights into which pollutants, cities, and regions have higher pollution levels.

Saving Cleaned Data
The cleaned dataset was saved in the same folder as:

global_air_pollution_cleaned_dropna.csv

This file will be used in later modules.

Summary of Work Completed (Up to Week 2)

Dataset loaded successfully

Missing rows removed

Clean dataset prepared

Data types validated

Initial EDA completed

Pollutant, city, and country comparisons visualized

Correlation analysis performed

Cleaned dataset saved

Next Steps
The upcoming modules will include:

Module 3: Pollution source labeling

Module 4: Building a prediction model

Additional insights and advanced analysis