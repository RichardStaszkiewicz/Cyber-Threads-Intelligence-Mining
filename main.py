import os
import logging
import requests
import time
import subprocess
from pathlib import Path

# Ensure all directories exist
directories = ["datasets", "logs", "models", "processed_data", "scraped_data"]
for dir_name in directories:
    Path(dir_name).mkdir(parents=True, exist_ok=True)

# Configure logging
log_file = "logs/main.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# URLs for datasets
DATASET_URL = "https://raw.githubusercontent.com/example/cti-dataset/main/cti_dataset.csv"
SCRAPED_REPORTS_URL = "https://raw.githubusercontent.com/example/cti-reports/main/reports.zip"

def download_file(url, output_path):
    """Downloads a file from a URL if it does not already exist."""
    if not os.path.exists(output_path):
        logging.info(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            logging.info(f"Downloaded {output_path}")
        else:
            logging.error(f"Failed to download {url}")

def setup_datasets():
    """Downloads necessary datasets if they do not exist."""
    dataset_path = "datasets/cti_dataset.csv"
    scraped_reports_path = "datasets/reports.zip"

    # Download CTI dataset
    download_file(DATASET_URL, dataset_path)

    # Download and extract scraped reports
    download_file(SCRAPED_REPORTS_URL, scraped_reports_path)
    if os.path.exists(scraped_reports_path):
        subprocess.run(["unzip", "-o", scraped_reports_path, "-d", "scraped_data"])
        logging.info("Extracted scraped reports.")

def run_module(script_name):
    """Runs a script and logs its output."""
    logging.info(f"Running {script_name}...")
    start_time = time.time()
    result = subprocess.run(["python", f"scripts/{script_name}"], capture_output=True, text=True)

    if result.returncode == 0:
        logging.info(f"{script_name} completed successfully.")
        logging.info(result.stdout)
    else:
        logging.error(f"{script_name} failed!")
        logging.error(result.stderr)

    logging.info(f"{script_name} execution time: {time.time() - start_time:.2f} seconds")

def main():
    """Runs the full CTI pipeline end-to-end."""
    logging.info("==== Starting Cyber Threat Intelligence Pipeline ====")

    # Step 1: Download required datasets
    setup_datasets()

    # Step 2: Scrape Cybersecurity Reports
    run_module("scraper.py")

    # Step 3: Preprocess the extracted reports
    run_module("preprocessor.py")

    # Step 4: Extract IOCs
    run_module("ioc_extractor.py")

    # Step 5: Validate IOCs using external threat intelligence APIs
    run_module("threat_validator.py")

    # Step 6: Build and analyze the knowledge graph
    run_module("graph_analysis.py")

    # Step 7: Train and test traditional ML-based classifiers
    run_module("threat_classifier.py")

    # Step 8: Fine-tune BERT for cybersecurity text classification
    run_module("bert_finetune.py")

    # Step 9: Train deep learning models (LSTM & BERT)
    run_module("deep_threat_classifier.py")

    # Step 10: Run API service for real-time predictions
    # run_module("api.py")

    logging.info("==== Cyber Threat Intelligence Pipeline Completed Successfully ====")

if __name__ == "__main__":
    main()
