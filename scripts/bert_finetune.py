import os
import json
import numpy as np
import pandas as pd
import logging
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from joblib import dump, load
import tensorflow as tf

class CyberThreatBERT:
    """
    Fine-tunes BERT on cybersecurity threat intelligence data.

    Attributes:
        dataset_file (str): Path to labeled cybersecurity dataset.
        model_dir (str): Directory to save fine-tuned model.
    """

    def __init__(self, dataset_file="datasets/cti_dataset.csv", model_dir="models", log_dir="logs"):
        """
        Initializes the CyberThreatBERT model.

        Args:
            dataset_file (str, optional): Path to labeled cybersecurity dataset. Defaults to "datasets/cti_dataset.csv".
            model_dir (str, optional): Directory to save fine-tuned model. Defaults to "models".
            log_dir (str, optional): Directory for logs. Defaults to "logs".
        """
        self.dataset_file = dataset_file
        self.model_dir = model_dir
        self.log_dir = log_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

        logging.basicConfig(
            filename=os.path.join(self.log_dir, "bert_finetune.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def load_dataset(self):
        """
        Loads and preprocesses the cybersecurity dataset.

        Returns:
            Dataset: Hugging Face dataset formatted for BERT fine-tuning.
        """
        if not os.path.exists(self.dataset_file):
            logging.error(f"Dataset not found: {self.dataset_file}")
            return None

        df = pd.read_csv(self.dataset_file)  # Assumes 'text' and 'label' columns
        df = df.dropna()

        # Tokenize text
        encodings = self.tokenizer(list(df["text"]), truncation=True, padding=True, max_length=128)
        labels = list(df["label"])

        dataset = Dataset.from_dict({**encodings, "label": labels})
        train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
        
        logging.info(f"Dataset loaded: {len(df)} samples")
        return train_test_split

    def fine_tune(self):
        """
        Fine-tunes BERT on the cybersecurity dataset.
        """
        dataset = self.load_dataset()
        if dataset is None:
            return

        training_args = TrainingArguments(
            output_dir=self.model_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            logging_dir=self.log_dir,
            logging_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        trainer.train()
        trainer.save_model(self.model_dir)
        logging.info("BERT fine-tuning completed and model saved.")

    def predict(self, text):
        """
        Predicts whether a given cybersecurity-related text is malicious or benign.

        Args:
            text (str): Input text.

        Returns:
            str: "Malicious" or "Benign"
        """
        inputs = self.tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
        outputs = self.model(inputs["input_ids"])
        prediction = np.argmax(outputs.logits.numpy())

        return "Malicious" if prediction == 1 else "Benign"

    def run(self):
        """Runs fine-tuning and sample prediction."""
        logging.info("Starting BERT fine-tuning...")
        self.fine_tune()

        sample_text = "New malware detected targeting financial institutions."
        print(f"Prediction for sample text: {self.predict(sample_text)}")

# Example usage:
if __name__ == "__main__":
    classifier = CyberThreatBERT()
    classifier.run()
