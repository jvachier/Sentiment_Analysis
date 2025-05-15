from collections import Counter

# import nltk
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK data
# nltk.download("punkt_tab")
# nltk.download("stopwords")
# nltk.download("wordnet")

# Initialize NLTK tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


class TextPreprocessor:
    """
    A class to preprocess text data for sentiment analysis.

    Attributes:
        stop_words (set): A set of stopwords to remove from the text.
        lemmatizer (WordNetLemmatizer): An instance of WordNetLemmatizer for lemmatizing words.
        tokenizer (Tokenizer): A Keras tokenizer for encoding text into sequences.
    """

    def __init__(self):
        """
        Initialize the TextPreprocessor class.
        """
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = None

    def preprocess_text(self, text):
        """
        Preprocess the input text by tokenizing, lowercasing, removing non-alphanumeric tokens,
        removing stopwords, and lemmatizing.

        Args:
            text (str): The input text to preprocess.

        Returns:
            str: The preprocessed text.
        """
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens]
        tokens = [token for token in tokens if token.isalnum()]
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return " ".join(tokens)

    def encode_map_fn(self, text, label):
        """
        Map function to encode text and label into sequences.

        Args:
            text (tf.Tensor): The input text tensor.
            label (tf.Tensor): The label tensor.

        Returns:
            tuple: Encoded text and label as tensors.
        """
        return tf.py_function(self.encode, inp=[text, label], Tout=(tf.int64, tf.int64))

    def encode(self, text_tensor, label):
        """
        Encode the input text into a sequence of integers using the tokenizer.

        Args:
            text_tensor (tf.Tensor): The input text tensor.
            label (int): The label associated with the text.

        Returns:
            tuple: Encoded text as a list of integers and the label.
        """
        text = text_tensor.numpy().decode("utf-8")
        text = self.preprocess_text(text)
        encoded_text = self.tokenizer.texts_to_sequences([text])[0]
        return encoded_text, label

    def fit_tokenizer(self, ds_raw):
        """
        Fit the tokenizer on the raw dataset.

        Args:
            ds_raw (tf.data.Dataset): The raw dataset containing text and labels.

        Returns:
            Counter: A counter object with token frequencies.
        """
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


class TextPreprocessorNoLabel:
    """
    A class to preprocess text data without labels for inference.

    Attributes:
        stop_words (set): A set of stopwords to remove from the text.
        lemmatizer (WordNetLemmatizer): An instance of WordNetLemmatizer for lemmatizing words.
        tokenizer (Tokenizer): A Keras tokenizer for encoding text into sequences.
    """

    def __init__(self):
        """
        Initialize the TextPreprocessorNoLabel class.
        """
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = None

    def preprocess_text(self, text):
        """
        Preprocess the input text by tokenizing, lowercasing, removing non-alphanumeric tokens,
        removing stopwords, and lemmatizing.

        Args:
            text (str): The input text to preprocess.

        Returns:
            str: The preprocessed text.
        """
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens]
        tokens = [token for token in tokens if token is not None and token.isalnum()]
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return " ".join(tokens)

    def encode_map_fn(self, text):
        """
        Map function to encode text into sequences.

        Args:
            text (tf.Tensor): The input text tensor.

        Returns:
            tf.Tensor: Encoded text as a tensor.
        """
        return tf.py_function(self.encode, inp=[text], Tout=(tf.int64))

    def encode(self, text_tensor):
        """
        Encode the input text into a sequence of integers using the tokenizer.

        Args:
            text_tensor (tf.Tensor): The input text tensor.

        Returns:
            list: Encoded text as a list of integers.
        """
        text = text_tensor.numpy().decode("utf-8")
        text = self.preprocess_text(text)
        encoded_text = self.tokenizer.texts_to_sequences([text])[0]
        return encoded_text

    def fit_tokenizer(self, ds_raw):
        """
        Fit the tokenizer on the raw dataset.

        Args:
            ds_raw (pd.DataFrame): The raw dataset containing text.

        Returns:
            Counter: A counter object with token frequencies.
        """
        reviews = [review for review in ds_raw["Review"]]
        reviews = [self.preprocess_text(review) for review in reviews]
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer()
        self.tokenizer.fit_on_texts(reviews)
        token_counts = Counter(
            token
            for text in reviews
            for token in self.tokenizer.texts_to_sequences([text])[0]
        )
        return token_counts
