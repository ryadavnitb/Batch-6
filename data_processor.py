import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnviroDataProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.feature_columns = [
            'PM2.5 AQI Value', 'NO2 AQI Value', 'CO AQI Value', 'Ozone AQI Value',
            'Temperature (C)', 'Humidity (%)', 'Wind Speed (m/s)',
            'Distance_to_Road', 'Distance_to_Industry', 'Distance_to_Agriculture'
        ]

    def load_data(self):
        """Loads and performs initial cleaning."""
        try:
            self.df = pd.read_csv(self.filepath)
            # Drop rows with missing critical target values if any
            self.df.dropna(subset=['AQI Value'], inplace=True)
            logger.info(f"Data Loaded Successfully: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

    def simulate_geospatial_features(self):
        """
        ADVANCED FEATURE: This function creates synthetic geospatial data 
        based on pollution signatures to enable the ML model.
        
        Logic:
        - High NO2 -> Likely close to roads.
        - High PM2.5 + SO2 -> Likely close to industries.
        - High Ozone + Dust -> Likely agricultural/natural.
        """
        if self.df is None: return

        np.random.seed(42)
        n = len(self.df)

        # 1. Simulate Distance to Road (Inversely proportional to NO2)
        # We add noise to make it realistic
        self.df['Distance_to_Road'] = 1000 - (self.df['NO2 AQI Value'] * 5) + np.random.normal(0, 50, n)
        self.df['Distance_to_Road'] = self.df['Distance_to_Road'].clip(lower=10, upper=5000)

        # 2. Simulate Distance to Industry (Inversely proportional to PM2.5 and CO)
        self.df['Distance_to_Industry'] = 2000 - (self.df['PM2.5 AQI Value'] * 3 + self.df['CO AQI Value'] * 5) + np.random.normal(0, 100, n)
        self.df['Distance_to_Industry'] = self.df['Distance_to_Industry'].clip(lower=50, upper=10000)

        # 3. Simulate Distance to Agriculture (Correlation with Ozone/Temp)
        self.df['Distance_to_Agriculture'] = 1500 - (self.df['Ozone AQI Value'] * 4) + np.random.normal(0, 80, n)
        self.df['Distance_to_Agriculture'] = self.df['Distance_to_Agriculture'].clip(lower=20, upper=8000)
        
        logger.info("Geospatial Simulation Complete.")

    def label_pollution_source(self):
        """
        Rule-Based Labeling Logic [cite: 60-64].
        Classifies the source based on the features we just engineered.
        """
        def categorize(row):
            # Vehicular: High NO2 & Close to Road
            if row['NO2 AQI Value'] > 30 and row['Distance_to_Road'] < 300:
                return 'Vehicular'
            
            # Industrial: High PM2.5/CO & Close to Industry
            elif (row['PM2.5 AQI Value'] > 50 or row['CO AQI Value'] > 5) and row['Distance_to_Industry'] < 500:
                return 'Industrial'
            
            # Agricultural: High Ozone
            elif row['Ozone AQI Value'] > 40 and row['Distance_to_Agriculture'] < 400:
                return 'Agricultural'
            
            # Natural/Background
            else:
                return 'Natural/Background'

        self.df['Pollution_Source'] = self.df.apply(categorize, axis=1)
        logger.info("Source Labeling Complete.")
        return self.df

    def get_processed_data(self):
        self.load_data()
        self.simulate_geospatial_features()
        self.label_pollution_source()
        return self.df