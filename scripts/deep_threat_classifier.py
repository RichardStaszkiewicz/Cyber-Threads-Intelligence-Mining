import os
import json
import numpy as np
import pandas as pd
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump, load

class DeepThreatClassifier:
    """
    Implements a Deep Learning-based threat classifier using LSTM and BERT.

    Attributes:
        ioc_file (str): Path to validated IOCs JSON file.
        lstm_model_file (str): Path to save LSTM model.
        bert_model_file (str): Path to save BERT model.
        log_file (str): Path to save logs.
    """

    def __init__(self, ioc_file="logs/validated_iocs.json", log_dir="logs"):
        """
        Initializes the DeepThreatClassifier.

        Args:
            ioc_file (str, optional): Path to validated IOCs JSON file. Defaults to "logs/validated_iocs.json".
            log_dir (str, optional): Directory for logs. Defaults to "logs".
        """
        self.ioc_file = ioc_file
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.lstm_model_file = os.path.join(self.log_dir, "lstm_model.h5")
        self.bert_model_file = os.path.join(self.log_dir, "bert_model.joblib")

        logging.basicConfig(
            filename=os.path.join(self.log_dir, "deep_threat_classification.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def load_iocs(self):
        """Loads validated IOCs."""
        with open(self.ioc_file, "r", encoding="utf-8") as file:
            return json.load(file)

    def preprocess_data(self):
        """Tokenizes and prepares data for training."""
        ioc_data = self.load_iocs()
        texts = []
        labels = []

        for entry in ioc_data:
            texts.append(entry["ioc"])
            labels.append(1 if entry["virustotal"] or entry["alienvault_otx"] else 0)  # 1: Malicious, 0: Benign

        return texts, np.array(labels)

    def train_lstm(self):
        """Trains an LSTM model for IOC classification."""
        texts, labels = self.preprocess_data()
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(texts)

        sequences = tokenizer.texts_to_sequences(texts)
        max_length = max(len(seq) for seq in sequences)
        padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post")

        X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

        model = Sequential([
            Embedding(input_dim=5000, output_dim=128, input_length=max_length),
            Bidirectional(LSTM(64, return_sequences=True)),
            LSTM(32),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid")
        ])

        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

        model.save(self.lstm_model_file)
        logging.info("LSTM model trained and saved.")

    def predict_lstm(self, ioc):
        """Predicts whether an IOC is malicious using LSTM."""
        model = load_model(self.lstm_model_file)
        tokenizer = Tokenizer(num_words=5000)

        sequence = tokenizer.texts_to_sequences([ioc])
        max_length = model.input_shape[1]
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding="post")

        prediction = model.predict(padded_sequence)
        return "Malicious" if prediction[0][0] > 0.5 else "Benign"

    def train_bert(self):
        """Trains a BERT model for IOC classification."""
        texts, labels = self.preprocess_data()
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        encodings = tokenizer(texts, truncation=True, padding=True, max_length=64, return_tensors="tf")
        X_train, X_test, y_train, y_test = train_test_split(encodings["input_ids"], labels, test_size=0.2, random_state=42)

        model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        model.fit(X_train, y_train, epochs=3, batch_size=16, validation_data=(X_test, y_test))

        dump(model, self.bert_model_file)
        logging.info("BERT model trained and saved.")

    def predict_bert(self, ioc):
        """Predicts whether an IOC is malicious using BERT."""
        model = load(self.bert_model_file)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        encoding = tokenizer(ioc, truncation=True, padding=True, max_length=64, return_tensors="tf")
        prediction = model.predict(encoding["input_ids"])

        return "Malicious" if np.argmax(prediction.logits) == 1 else "Benign"

    def run(self):
        """Runs training and sample predictions."""
        logging.info("Training LSTM model...")
        self.train_lstm()
        logging.info("Training BERT model...")
        self.train_bert()

        sample_ioc = "malicious-ip-123.45.67.89"
        print(f"LSTM Prediction for {sample_ioc}: {self.predict_lstm(sample_ioc)}")
        print(f"BERT Prediction for {sample_ioc}: {self.predict_bert(sample_ioc)}")

# Example usage:
if __name__ == "__main__":
    classifier = DeepThreatClassifier()
    classifier.run()
