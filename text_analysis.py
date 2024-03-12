from web_scraper import WebScraper
from text_preprocessor import TextPreprocessor
from semantic_analyzer import SemanticAnalyzer

class TextAnalysisApp:
    def __init__(self):
        self.web_scraper = WebScraper()
        self.text_preprocessor = TextPreprocessor()
        self.semantic_analyzer = SemanticAnalyzer()

    def run(self):
        try:
            # Prompt the user for a URL
            url = input("Enter the URL to scrape: ")

            # Scrape text content from the URL
            html_content = self.web_scraper.get_html(url)
            if html_content:
                parsed_html = self.web_scraper.parse_html(html_content)
                if parsed_html:
                    text_data = self.web_scraper.extract_text(parsed_html)
                    if text_data:
                        print("\n-- Text Data Extracted --")
                        print(text_data[:200] + "...")  # Display a snippet of the extracted text
                        print("\n-- Preprocessing Text --")
                        # Preprocess the text
                        clean_text = self.text_preprocessor.preprocess_text(text_data)
                        if clean_text:
                            print(clean_text[:200] + "...")  # Display a snippet of the preprocessed text
                            print("\n-- Analyzing Sentiment --")
                            # Analyze sentiment
                            sentiment = self.semantic_analyzer.analyze_sentiment(clean_text)
                            if sentiment:
                                print("Sentiment:", sentiment)
                            else:
                                print("Failed to analyze sentiment.")
                        else:
                            print("Failed to preprocess text.")
                    else:
                        print("Failed to extract text data.")
                else:
                    print("Failed to parse HTML.")
            else:
                print("Failed to fetch HTML content.")
        except Exception as e:
            print(f"An error occurred: {e}")

def main():
    app = TextAnalysisApp()
    app.run()

if __name__ == "__main__":
    main()
