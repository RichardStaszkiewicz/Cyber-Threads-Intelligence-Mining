import os
import json
import numpy as np
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from joblib import dump, load

class ThreatClassifier:
    """
    Classifies cyber threats using Machine Learning.

    Attributes:
        ioc_file (str): Path to validated IOCs JSON file.
        model_file (str): Path to save trained model.
    """

    def __init__(self, ioc_file="logs/validated_iocs.json", model_file="threat_model.joblib", log_file="logs/threat_classification.log"):
        """
        Initializes the ThreatClassifier.

        Args:
            ioc_file (str, optional): Path to validated IOCs JSON file. Defaults to "logs/validated_iocs.json".
            model_file (str, optional): Path to save trained model. Defaults to "threat_model.joblib".
            log_file (str, optional): Path to save logs. Defaults to "logs/threat_classification.log".
        """
        self.ioc_file = ioc_file
        self.model_file = model_file
        os.makedirs("logs", exist_ok=True)

        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def load_iocs(self):
        """Loads validated IOCs."""
        with open(self.ioc_file, "r", encoding="utf-8") as file:
            return json.load(file)

    def extract_features(self):
        """Extracts textual features for ML classification."""
        ioc_data = self.load_iocs()
        texts = []
        labels = []

        for entry in ioc_data:
            texts.append(entry["ioc"])
            labels.append(1 if entry["virustotal"] or entry["alienvault_otx"] else 0)  # 1: malicious, 0: benign

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts).toarray()
        y = np.array(labels)

        return X, y, vectorizer

    def train_model(self):
        """Trains an ML model for threat classification."""
        X, y, vectorizer = self.extract_features()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        logging.info("\n" + classification_report(y_test, y_pred))

        dump((model, vectorizer), self.model_file)
        print(f"Model trained and saved to {self.model_file}")

    def predict(self, ioc):
        """Predicts whether an IOC is malicious."""
        model, vectorizer = load(self.model_file)
        X = vectorizer.transform([ioc]).toarray()
        prediction = model.predict(X)
        return "Malicious" if prediction[0] == 1 else "Benign"

    def run(self):
        """Runs training and sample prediction."""
        self.train_model()
        sample_ioc = "malicious-ip-123.45.67.89"
        print(f"Prediction for {sample_ioc}: {self.predict(sample_ioc)}")

# Example usage:
if __name__ == "__main__":
    classifier = ThreatClassifier()
    classifier.run()
