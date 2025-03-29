import requests
from bs4 import BeautifulSoup
import os
import re
import time
import logging
import concurrent.futures
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CyberThreatScraper:
    """
    A web scraper for extracting cybersecurity threat intelligence articles from security blogs.

    Attributes:
        urls (list): List of URLs to scrape.
        output_dir (str): Directory to save scraped articles.
    """

    def __init__(self, urls, output_dir="scraped_data", max_retries=3, proxy=None):
        """
        Initializes the CyberThreatScraper with a list of URLs and an output directory.

        Args:
            urls (list): List of security blog URLs to scrape.
            output_dir (str, optional): Directory to store scraped text.
            max_retries (int, optional): Maximum number of retries for failed requests. Defaults to 3.
            proxy (dict, optional): Proxy settings. Example: {'http': 'http://proxy.com:8080'}
        """
        self.urls = [url for url in urls if self.is_valid_url(url)]
        self.output_dir = output_dir
        self.max_retries = max_retries
        self.proxy = proxy
        os.makedirs(self.output_dir, exist_ok=True)

    def is_valid_url(self, url):
        """Checks if a URL is valid."""
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)

    def fetch_html(self, url, retries=0):
        """
        Fetches the HTML content of a webpage with retries.

        Args:
            url (str): The URL of the page to scrape.
            retries (int): Number of retries in case of failure.

        Returns:
            str: HTML content of the page, or None if request fails.
        """
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, proxies=self.proxy, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            if retries < self.max_retries:
                wait = 2 ** retries  # Exponential backoff
                logging.warning(f"Retrying {url} in {wait} seconds... ({retries + 1}/{self.max_retries})")
                time.sleep(wait)
                return self.fetch_html(url, retries + 1)
            logging.error(f"Failed to fetch {url} after {self.max_retries} retries: {e}")
            return None

    def extract_text(self, html):
        """
        Extracts readable text content from HTML using BeautifulSoup.

        Args:
            html (str): The HTML content of the webpage.

        Returns:
            str: Extracted article text.
        """
        soup = BeautifulSoup(html, 'html.parser')

        # Common article containers
        possible_selectors = [
            'article',
            'div.entry-content',
            'div.post-content',
            'div.blog-content',
            'div.article-body',
            'section.article'
        ]

        for selector in possible_selectors:
            article = soup.select_one(selector)
            if article:
                return article.get_text(separator="\n", strip=True)

        logging.warning("No structured content found; extracting main text body.")
        return soup.get_text(separator="\n", strip=True)  # Fallback to full page text

    def save_text(self, url, text):
        """
        Saves the extracted text to a file.

        Args:
            url (str): The source URL.
            text (str): Extracted article content.
        """
        filename = re.sub(r'\W+', '_', url)[:50] + ".txt"
        filepath = os.path.join(self.output_dir, filename)

        try:
            with open(filepath, "w", encoding="utf-8") as file:
                file.write(text)
            logging.info(f"Saved: {filepath}")
        except IOError as e:
            logging.error(f"Error saving file {filepath}: {e}")

    def scrape_url(self, url):
        """
        Scrapes a single URL.

        Args:
            url (str): The URL to scrape.
        """
        logging.info(f"Scraping: {url}")
        html = self.fetch_html(url)
        if html:
            text = self.extract_text(html)
            if text:
                self.save_text(url, text)
            else:
                logging.warning(f"No readable content found on {url}")

    def run(self, use_threads=True):
        """
        Executes the scraping process for all URLs in the list.

        Args:
            use_threads (bool): Whether to use multithreading for faster scraping.
        """
        if use_threads:
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                executor.map(self.scrape_url, self.urls)
        else:
            for url in self.urls:
                self.scrape_url(url)


if __name__ == "__main__":
    security_blogs = [
        "https://krebsonsecurity.com/",
        "https://thehackernews.com/",
        "https://thehackernews.com/2025/03/this-malicious-pypi-package-stole.html",
        "https://www.darkreading.com/",
        "https://www.schneier.com/",
        "https://www.bleepingcomputer.com/",
        "https://www.csoonline.com/",
        "https://threatpost.com/",
        "https://nakedsecurity.sophos.com/",
        "https://www.securityweek.com/",
        "https://www.welivesecurity.com/"
    ]

    scraper = CyberThreatScraper(security_blogs)
    scraper.run()