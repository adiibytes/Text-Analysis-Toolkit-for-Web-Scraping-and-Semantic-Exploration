from textblob import TextBlob

class SemanticAnalyzer:
    def __init__(self):
        pass

    def analyze_sentiment(self, text):
        if text is None:
            return None
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment
            return sentiment
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return None
