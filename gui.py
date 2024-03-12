import tkinter as tk
from tkinter import messagebox
from web_scraper import WebScraper
from text_preprocessor import TextPreprocessor
from semantic_analyzer import SemanticAnalyzer
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
import topic_modelling  # Update import statement
class TextAnalysisGUI:
    def __init__(self, master):
        self.master = master
        master.title("Text Analysis Toolkit")

        # Define colors
        self.bg_color = "#f0f0f0"
        self.text_color = "#333333"
        self.heading_color = "#008080"
        self.button_bg_color = "#008080"
        self.button_fg_color = "white"

        # Apply styles
        master.configure(bg=self.bg_color)
        self.label_style = {"bg": self.bg_color, "fg": self.text_color}
        self.entry_style = {"bg": "white"}
        self.button_style = {"bg": self.button_bg_color, "fg": self.button_fg_color}

        self.label = tk.Label(master, text="Enter URL:", **self.label_style)
        self.label.pack()

        self.url_entry = tk.Entry(master, **self.entry_style)
        self.url_entry.pack()

        self.analyze_button = tk.Button(master, text="Analyze Text", command=self.analyze_text, **self.button_style)
        self.analyze_button.pack()

        self.summarize_button = tk.Button(master, text="Summarize Text", command=self.summarize_text, **self.button_style)
        self.summarize_button.pack()

        self.text_output = tk.Text(master, height=15, width=50, bg="white")
        self.text_output.pack()

        self.web_scraper = WebScraper()
        self.text_preprocessor = TextPreprocessor()
        self.semantic_analyzer = SemanticAnalyzer()

        self.stop_words = set(stopwords.words("english"))
        # New button for discovering topics
        self.topic_modeling_button = tk.Button(master, text="Discover Topics", command=self.discover_topics)
        self.topic_modeling_button.pack()

        # New instance variable for topic modeling
        self.topic_modeler = topic_modelling  # Update this with appropriate import alias
      
    def discover_topics(self):
        url = self.url_entry.get()
        if url:
            # Fetch HTML content, parse it, and extract text data
            html_content = self.web_scraper.get_html(url)
            parsed_html = self.web_scraper.parse_html(html_content)
        
            if parsed_html:
                text_data = self.web_scraper.extract_text(parsed_html)
                if text_data:
                    # Preprocess the text data
                    preprocessed_text = ' '.join(self.text_preprocessor.preprocess_text(text_data))
                
                    # Vectorize the preprocessed text data
                    vectorizer, tfidf_matrix = self.topic_modeler.vectorize_text([preprocessed_text])

                
                    # Train the LDA model
                    num_topics = 5  # Example number of topics
                    lda_model = self.topic_modeler.train_lda_model(tfidf_matrix, num_topics, vectorizer)
                
                    # Visualize the discovered topics
                    self.topic_modeler.visualize_topics(lda_model, self.master) 
                else:
                    messagebox.showerror("Error", "Failed to extract text data from the URL.")
            else:
                messagebox.showerror("Error", "Failed to parse HTML content.")
        else:
            messagebox.showwarning("Warning", "Please enter a URL.")


    def analyze_text(self):
        url = self.url_entry.get()
        if url:
            html_content = self.web_scraper.get_html(url)
            if html_content:
                parsed_html = self.web_scraper.parse_html(html_content)
                if parsed_html:
                    paragraphs = self.web_scraper.extract_paragraphs(parsed_html)
                    if paragraphs:
                        self.text_output.delete(1.0, tk.END)
                        self.text_output.insert(tk.END, "-- Sentiment Analysis Results --\n", "heading")  # Apply a tag to the heading
                        for i, paragraph in enumerate(paragraphs, start=1):
                            clean_text = self.text_preprocessor.preprocess_text(paragraph)
                            if clean_text:
                                sentiment = self.semantic_analyzer.analyze_sentiment(clean_text)
                                if sentiment:
                                    self.text_output.insert(tk.END, f"Paragraph {i} Sentiment: {sentiment}\n")
                                else:
                                    self.text_output.insert(tk.END, f"Error: Failed to analyze sentiment for paragraph {i}\n")
                            else:
                                self.text_output.insert(tk.END, f"Error: Failed to preprocess text for paragraph {i}\n")
                    else:
                        self.text_output.delete(1.0, tk.END)
                        self.text_output.insert(tk.END, "Error: Failed to extract paragraphs from the web page.")
                else:
                    self.text_output.delete(1.0, tk.END)
                    self.text_output.insert(tk.END, "Error: Failed to parse HTML content.")
            else:
                self.text_output.delete(1.0, tk.END)
                self.text_output.insert(tk.END, "Error: Failed to fetch HTML content.")
        else:
            messagebox.showwarning("Warning", "Please enter a URL.")

    def summarize_text(self):
        url = self.url_entry.get()
        if url:
            html_content = self.web_scraper.get_html(url)
            if html_content:
                parsed_html = self.web_scraper.parse_html(html_content)
                if parsed_html:
                    text_data = self.web_scraper.extract_text(parsed_html)
                    if text_data:
                        self.text_output.delete(1.0, tk.END)
                        self.text_output.insert(tk.END, "-- Text Summary --\n", "heading")  # Apply a tag to the heading
                        summary = self.summarize_text_data(text_data)
                        if summary:
                            self.text_output.insert(tk.END, summary)
                        else:
                            self.text_output.insert(tk.END, "Error: Failed to summarize text.")
                    else:
                        self.text_output.insert(tk.END, "Error: Failed to extract text data.")
                else:
                    self.text_output.insert(tk.END, "Error: Failed to parse HTML content.")
            else:
                self.text_output.insert(tk.END, "Error: Failed to fetch HTML content.")
        else:
            messagebox.showwarning("Warning", "Please enter a URL.")

    def summarize_text_data(self, text_data):
        sentences = sent_tokenize(text_data)
        word_frequencies = self._calculate_word_frequencies(text_data)
        sentence_scores = defaultdict(int)
        for sentence in sentences:
            for word in word_tokenize(sentence.lower()):
                if word in word_frequencies:
                    sentence_scores[sentence] += word_frequencies[word]
        summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
        summary = ' '.join(summary_sentences[:5])  # Concatenate top 5 sentences to form summary
        return summary

    def _calculate_word_frequencies(self, text_data):
        words = word_tokenize(text_data)
        filtered_words = [word for word in words if word.lower() not in self.stop_words and word.isalnum()]
        word_frequencies = FreqDist(filtered_words)
        return word_frequencies

def main():
    root = tk.Tk()
    app = TextAnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
