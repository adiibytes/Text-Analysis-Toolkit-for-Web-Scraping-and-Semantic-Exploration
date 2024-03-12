import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        if text is None:
            return None
        try:
            tokens = word_tokenize(text)
            filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words]
            clean_tokens = [re.sub(r'[^a-zA-Z]', '', word) for word in filtered_tokens if len(word) > 1]
            clean_text = ' '.join(clean_tokens).lower()
            return clean_text
        except Exception as e:
            print(f"Error preprocessing text: {e}")
            return None
