import os
import re
import json
import spacy
import pandas as pd
import tldextract

class IOCExtractor:
    """
    Extracts Indicators of Compromise (IOCs) such as IP addresses, domains, hashes, and URLs
    from preprocessed cybersecurity text.

    Attributes:
        input_dir (str): Directory containing preprocessed text files.
        output_file (str): Path to save extracted IOCs.
        nlp (spacy.lang.en.English): SpaCy NLP model for NER.
    """

    def __init__(self, input_dir="scraped_data", output_file="extracted_iocs.json"):
        """
        Initializes the IOCExtractor.

        Args:
            input_dir (str, optional): Directory of preprocessed text files. Defaults to "processed_data".
            output_file (str, optional): Path to save extracted IOCs. Defaults to "extracted_iocs.json".
        """
        self.input_dir = input_dir
        self.output_file = output_file
        self.nlp = spacy.load("en_core_web_sm")

        # IOC Patterns
        self.patterns = {
            "IPv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
            "IPv6": re.compile(r"\b([a-fA-F0-9:]+:+)+[a-fA-F0-9]+\b"),
            "MD5": re.compile(r"\b[a-fA-F0-9]{32}\b"),
            "SHA1": re.compile(r"\b[a-fA-F0-9]{40}\b"),
            "SHA256": re.compile(r"\b[a-fA-F0-9]{64}\b"),
            "URL": re.compile(r"https?://[^\s]+"),
            "Domain": re.compile(r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,6}\b"),
        }

    def extract_iocs(self, text):
        """
        Extracts IOCs using regex patterns.

        Args:
            text (str): Input text.

        Returns:
            dict: Extracted IOCs categorized into different types.
        """
        extracted = {key: set(pattern.findall(text)) for key, pattern in self.patterns.items()}

        # Filter out false positives in domain extraction
        extracted["Domain"] = {dom for dom in extracted["Domain"] if self.validate_domain(dom)}

        return {key: list(values) for key, values in extracted.items()}

    def validate_domain(self, domain):
        """
        Validates a domain to ensure it's not an IP or a false positive.

        Args:
            domain (str): The extracted domain.

        Returns:
            bool: True if valid, False otherwise.
        """
        if re.match(self.patterns["IPv4"], domain) or re.match(self.patterns["IPv6"], domain):
            return False  # It's an IP, not a domain

        extracted = tldextract.extract(domain)
        return bool(extracted.suffix)  # Ensure it has a valid TLD

    def extract_named_entities(self, text):
        """
        Extracts cybersecurity-related named entities using SpaCy NER.

        Args:
            text (str): Input text.

        Returns:
            dict: Extracted named entities categorized by label.
        """
        doc = self.nlp(text)
        entities = {}

        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT", "MISC"]:
                entities.setdefault(ent.label_, set()).add(ent.text)

        return {key: list(values) for key, values in entities.items()}

    def process_file(self, filepath):
        """
        Extracts IOCs and named entities from a single file.

        Args:
            filepath (str): Path to the text file.

        Returns:
            dict: Extracted IOCs and named entities.
        """
        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()

        iocs = self.extract_iocs(text)
        named_entities = self.extract_named_entities(text)

        return {"file": os.path.basename(filepath), "IOCs": iocs, "Named_Entities": named_entities}

    def run(self):
        """
        Processes all files in the input directory and saves extracted IOCs.
        """
        results = []

        for filename in os.listdir(self.input_dir):
            filepath = os.path.join(self.input_dir, filename)

            if os.path.isfile(filepath):
                print(f"Extracting IOCs from: {filename}")
                extracted_data = self.process_file(filepath)
                results.append(extracted_data)

        with open(self.output_file, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)

        print(f"Extraction complete. Results saved to {self.output_file}")


# Example usage:
if __name__ == "__main__":
    extractor = IOCExtractor()
    extractor.run()
