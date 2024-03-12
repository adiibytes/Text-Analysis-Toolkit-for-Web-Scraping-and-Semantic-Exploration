import requests
from bs4 import BeautifulSoup

class WebScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def get_html(self, url):
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL: {e}")
            return None

    def parse_html(self, html):
        try:
            soup = BeautifulSoup(html, 'html.parser')
            return soup
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return None

    def extract_text(self, soup):
        try:
            text = ' '.join(p.get_text() for p in soup.find_all('p'))
            return text
        except Exception as e:
            print(f"Error extracting text: {e}")
            return None

    def extract_paragraphs(self, soup):
        try:
            paragraphs = [p.get_text() for p in soup.find_all('p')]
            return paragraphs
        except Exception as e:
            print(f"Error extracting paragraphs: {e}")
            return None
