#TODO: Add more threat intelligence services, get the proper Alienvault OTX API to work according to a specific endpoints
import os
import json
import yaml
import logging
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

class ThreatValidator:
    """
    Validates extracted IOCs against external threat intelligence feeds.

    Attributes:
        ioc_file (str): Path to extracted IOCs JSON file.
        log_file (str): Path to store validation logs.
        api_keys (dict): API keys for external threat intelligence services.
    """

    def __init__(self, ioc_file="extracted_iocs.json", log_dir="logs", config_file="config.yaml", check_virustotal_limits=4, check_alienvault_otx_limits=-1):
        """
        Initializes the ThreatValidator.

        Args:
            ioc_file (str, optional): Path to extracted IOCs JSON file. Defaults to "extracted_iocs.json".
            log_dir (str, optional): Directory to store logs. Defaults to "logs".
            config_file (str, optional): Path to the configuration file. Defaults to "config.yaml".
        """
        self.check_alienvault_otx_limits = check_alienvault_otx_limits
        self.check_virustotal_limits = check_virustotal_limits
        self.ioc_file = ioc_file
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"threat_validation_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

        # Load API keys from config file
        self.api_keys = self.load_config(config_file)

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )

    def load_config(self, config_file):
        """
        Loads API keys from a YAML configuration file.

        Args:
            config_file (str): Path to the configuration file.

        Returns:
            dict: Dictionary containing API keys.
        """
        if not os.path.exists(config_file):
            logging.error(f"Configuration file not found: {config_file}")
            return {}

        with open(config_file, "r", encoding="utf-8") as file:
            try:
                config = yaml.safe_load(file)
                return config.get("api_keys", {})
            except yaml.YAMLError as e:
                logging.error(f"Error reading config file: {e}")
                return {}

    def load_iocs(self):
        """
        Loads extracted IOCs from the JSON file.

        Returns:
            list: List of IOCs from the file.
        """
        if not os.path.exists(self.ioc_file):
            logging.error(f"IOC file not found: {self.ioc_file}")
            return []

        with open(self.ioc_file, "r", encoding="utf-8") as file:
            return json.load(file)

    def check_virustotal(self, ioc):
        """
        Checks an IOC against VirusTotal.

        Args:
            ioc (str): IOC (IP, domain, or hash) to check.

        Returns:
            dict: Response from VirusTotal.
        """
        url = f"https://www.virustotal.com/api/v3/search?query={ioc}"
        headers = {"x-apikey": self.api_keys.get("VirusTotal", "")}

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            logging.info(f"VirusTotal response for {ioc}: {data}")
            return data
        except requests.RequestException as e:
            logging.error(f"Error checking VirusTotal for {ioc}: {e}")
            return None

    def check_alienvault_otx(self, ioc):
        """
        Checks an IOC against AlienVault OTX.

        Args:
            ioc (str): IOC (IP, domain, or hash) to check.

        Returns:
            dict: Response from AlienVault OTX.
        """
        url = f"https://otx.alienvault.com/api/v1/indicators/domain/{ioc}/general"
        headers = {"X-OTX-API-KEY": self.api_keys.get("AlienVaultOTX", "")}

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            logging.info(f"AlienVault OTX response for {ioc}: {data}")
            return data
        except requests.RequestException as e:
            logging.error(f"Error checking AlienVault OTX for {ioc}: {e}")
            return None

    def validate_ioc(self, ioc, check_virustotal=True, check_alienvault_otx=True):
        """
        Validates a single IOC using VirusTotal and AlienVault.

        Args:
            ioc (str): The IOC to validate.

        Returns:
            dict: Combined validation results.
        """
        results = {
            "ioc": ioc,
            "virustotal": self.check_virustotal(ioc) if check_virustotal else None,
            "alienvault_otx": self.check_alienvault_otx(ioc) if check_alienvault_otx else None
        }
        return results

    def run_validation(self):
        """
        Executes IOC validation with multithreading.
        """
        ioc_data = self.load_iocs()
        all_iocs = []

        for entry in ioc_data:
            for category, iocs in entry["IOCs"].items():
                all_iocs.extend(iocs)

        logging.info(f"Validating {len(all_iocs)} IOCs...")

        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_ioc = {
                executor.submit(self.validate_ioc, ioc, check_virustotal=(i < self.check_virustotal_limits or self.check_virustotal_limits == -1), check_alienvault_otx=(i < self.check_alienvault_otx_limits or self.check_alienvault_otx_limits == -1)): ioc
                for i, ioc in enumerate(all_iocs)
            }
            for future in future_to_ioc:
                result = future.result()
                results.append(result)

        # Save results
        output_file = os.path.join(self.log_dir, "validated_iocs.json")
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)

        logging.info(f"Validation complete. Results saved to {output_file}")

# Example usage
if __name__ == "__main__":
    validator = ThreatValidator(check_virustotal_limits=0, check_alienvault_otx_limits=2)
    validator.run_validation()
