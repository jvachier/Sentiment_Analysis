from collections import Counter

import nltk
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize NLTK tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = None

    def preprocess_text(self, text):
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens]
        tokens = [token for token in tokens if token.isalnum()]
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return " ".join(tokens)

    def encode_map_fn(self, text, label):
        return tf.py_function(self.encode, inp=[text, label], Tout=(tf.int64, tf.int64))

    def encode(self, text_tensor, label):
        text = text_tensor.numpy().decode("utf-8")
        text = self.preprocess_text(text)
        encoded_text = self.tokenizer.texts_to_sequences([text])[0]
        return encoded_text, label

    def fit_tokenizer(self, ds_raw):
        reviews = [review.numpy().decode("utf-8") for review, _ in ds_raw]
        reviews = [self.preprocess_text(review) for review in reviews]
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer()
        self.tokenizer.fit_on_texts(reviews)
        token_counts = Counter(
            token
            for text in reviews
            for token in self.tokenizer.texts_to_sequences([text])[0]
        )
        return token_counts
