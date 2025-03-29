import os
import re
import nltk
import spacy
import logging
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

nltk.download("punkt")
nltk.download("stopwords")

class TextPreprocessor:
    """
    Preprocesses cybersecurity threat intelligence text data while preserving IoCs.
    """

    def __init__(self, input_dir="scraped_data", output_dir="processed_data"):
        """
        Initializes the TextPreprocessor.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.stop_words = set(stopwords.words("english"))
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logging.info("Loaded SpaCy language model successfully.")
        except OSError:
            logging.warning("SpaCy language model not found. Downloading...")
            spacy.cli.download('en_core_web_sm')
            self.nlp = spacy.load("en_core_web_sm")
            logging.info("Downloaded and loaded SpaCy language model.")
    
    def clean_text(self, text):
        """
        Cleans text but preserves IoCs (IPv4, IPv6, hashes, URLs, domains).
        """
        logging.debug("Cleaning text while preserving IoCs...")
        
        # Preserve IoCs before text cleaning
        ioc_patterns = {
            "IPv4": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            "IPv6": r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',
            "MD5": r'\b[a-fA-F0-9]{32}\b',
            "SHA1": r'\b[a-fA-F0-9]{40}\b',
            "SHA256": r'\b[a-fA-F0-9]{64}\b',
            "URL": r'\bhttps?://\S+\b',
            "Domain": r'\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
        }
        
        preserved_iocs = []
        for key, pattern in ioc_patterns.items():
            matches = re.findall(pattern, text)
            preserved_iocs.extend(matches)
            logging.debug(f"Found {len(matches)} {key} IoCs.")
        
        # Perform cleaning but exclude IoCs
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only words
        text = re.sub(r'\s+', ' ', text).strip().lower()
        
        return text, preserved_iocs
    
    def tokenize_and_lemmatize(self, text):
        """
        Tokenizes text into words and applies lemmatization.
        """
        logging.debug("Tokenizing and lemmatizing text...")
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if token.text not in self.stop_words and token.is_alpha]
        logging.debug(f"Extracted {len(tokens)} tokens after lemmatization.")
        return tokens
    
    def preprocess_file(self, filepath):
        """
        Processes a single text file while preserving IoCs.
        """
        logging.info(f"Processing file: {filepath}")
        
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                raw_text = file.read()
            
            cleaned_text, preserved_iocs = self.clean_text(raw_text)
            tokens = self.tokenize_and_lemmatize(cleaned_text)
            processed_text = " ".join(tokens) + "\n" + " ".join(preserved_iocs)  # Append IoCs to ensure retention
            
            logging.info(f"Successfully processed file: {filepath}")
            return processed_text
        except Exception as e:
            logging.error(f"Error processing file {filepath}: {e}")
            return ""

    def run(self):
        """
        Executes text preprocessing for all files in the input directory.
        """
        logging.info("Starting text preprocessing...")
        for filename in os.listdir(self.input_dir):
            input_path = os.path.join(self.input_dir, filename)
            output_path = os.path.join(self.output_dir, filename)

            if os.path.isfile(input_path):
                logging.info(f"Processing: {filename}")
                processed_text = self.preprocess_file(input_path)
                
                with open(output_path, "w", encoding="utf-8") as file:
                    file.write(processed_text)
                
                logging.info(f"Saved processed file: {output_path}")
        logging.info("Text preprocessing completed.")

# Example usage:
if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    preprocessor.run()