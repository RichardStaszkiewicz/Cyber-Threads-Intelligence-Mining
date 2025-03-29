from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
from transformers import BertTokenizer, TFBertForSequenceClassification
import numpy as np

app = FastAPI()

# Load models
lstm_model = load("../models/lstm_model.h5")
bert_model = TFBertForSequenceClassification.from_pretrained("../models/bert_model")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class ThreatRequest(BaseModel):
    text: str

@app.post("/predict/lstm")
def predict_lstm(request: ThreatRequest):
    """Predict if text is malicious using LSTM."""
    sequence = tokenizer.texts_to_sequences([request.text])
    padded_sequence = np.pad(sequence, [(0, 0), (0, 128 - len(sequence[0]))], mode="constant")

    prediction = lstm_model.predict(padded_sequence)
    return {"threat_level": "Malicious" if prediction[0][0] > 0.5 else "Benign"}

@app.post("/predict/bert")
def predict_bert(request: ThreatRequest):
    """Predict if text is malicious using BERT."""
    inputs = tokenizer(request.text, return_tensors="tf", truncation=True, padding=True, max_length=128)
    outputs = bert_model(inputs["input_ids"])
    prediction = np.argmax(outputs.logits.numpy())

    return {"threat_level": "Malicious" if prediction == 1 else "Benign"}

@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "CTI Model API Running"}
