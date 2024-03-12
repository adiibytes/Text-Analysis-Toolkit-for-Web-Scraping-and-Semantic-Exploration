import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim.models import LdaModel
from gensim import corpora
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

# Vectorization function
def vectorize_text(text_data):
    vectorizer = TfidfVectorizer(tokenizer=preprocess_text)
    tfidf_matrix = vectorizer.fit_transform(text_data)
    return vectorizer, tfidf_matrix

# LDA model training function
def train_lda_model(tfidf_matrix, num_topics, vectorizer):
    corpus = gensim.matutils.Sparse2Corpus(tfidf_matrix.T)
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dict(enumerate(vectorizer.get_feature_names_out())))
    return lda_model



# Visualization function
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

def visualize_topics(lda_model, master):
    topics = lda_model.show_topics(formatted=False)
    for idx, topic in enumerate(topics):
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)

        # Extract words and weights for the topic
        words = [word for word, weight in topic[1]]
        weights = [weight for word, weight in topic[1]]

        print("Words:", words)  # Debugging statement
        print("Weights:", weights)  # Debugging statement

        # Plot words and their corresponding weights
        ax.bar(words, weights, color='skyblue')
        ax.set_xlabel('Words')
        ax.set_ylabel('Word Weights')
        ax.set_title(f'Topic {idx+1} Word Weights')

        # Rotate x-axis labels for better visibility
        plt.xticks(rotation=45)

        # Embed the matplotlib plot in the tkinter GUI window
        canvas = FigureCanvasTkAgg(fig, master=master)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        plt.close(fig)  # Close the matplotlib figure to avoid displaying multiple windows
