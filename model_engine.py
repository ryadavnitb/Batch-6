import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PollutionPredictor:
    def __init__(self, df):
        self.df = df
        self.model = None
        self.features = [
            'PM2.5 AQI Value', 'NO2 AQI Value', 'CO AQI Value', 'Ozone AQI Value',
            'Temperature (C)', 'Humidity (%)', 'Wind Speed (m/s)',
            'Distance_to_Road', 'Distance_to_Industry', 'Distance_to_Agriculture'
        ]
        self.target = 'Pollution_Source'

    def train_model(self):
        """Trains a Random Forest Classifier."""
        X = self.df[self.features]
        y = self.df[self.target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize Advanced Model
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        
        logger.info("Training Model...")
        self.model.fit(X_train, y_train)

        # Evaluate
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logger.info(f"Model Training Complete. Accuracy: {acc:.2f}")
        logger.info("\n" + classification_report(y_test, preds))

        return acc

    def save_model(self, path='model.pkl'):
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path='model.pkl'):
        self.model = joblib.load(path)
        return self.model

    def predict(self, input_data):
        """Predicts source and returns probability/confidence."""
        if not self.model:
            raise Exception("Model not loaded!")
        
        # input_data should be a DataFrame with correct feature columns
        prediction = self.model.predict(input_data)[0]
        proba = self.model.predict_proba(input_data)[0]
        confidence = max(proba) * 100
        
        return prediction, confidence