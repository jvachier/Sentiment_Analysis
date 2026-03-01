import pytest
import tensorflow as tf
from pydantic import ValidationError
from src.modules.text_vectorizer_sentiment_analysis import (
    TextVectorizer,
    TextVectorizerConfig,
)


def test_text_vectorizer_config_valid():
    """Test TextVectorizerConfig with valid parameters."""
    config = TextVectorizerConfig(max_tokens=10000, output_sequence_length=200)
    assert config.max_tokens == 10000
    assert config.output_sequence_length == 200


def test_text_vectorizer_config_invalid_max_tokens():
    """Test TextVectorizerConfig with invalid max_tokens."""
    with pytest.raises(ValidationError):
        TextVectorizerConfig(max_tokens=0, output_sequence_length=200)


def test_text_vectorizer_config_invalid_sequence_length():
    """Test TextVectorizerConfig with invalid output_sequence_length."""
    with pytest.raises(ValidationError):
        TextVectorizerConfig(max_tokens=10000, output_sequence_length=-1)


def test_text_vectorizer_config_defaults():
    """Test TextVectorizerConfig with default values."""
    config = TextVectorizerConfig()
    assert config.max_tokens == 20000
    assert config.output_sequence_length == 500


def test_text_vectorizer_initialization():
    """Test TextVectorizer initialization."""
    vectorizer = TextVectorizer(max_tokens=5000, output_sequence_length=100)
    assert vectorizer.max_tokens == 5000
    assert vectorizer.output_sequence_length == 100
    assert isinstance(vectorizer.text_vec, tf.keras.layers.TextVectorization)


def test_text_vectorizer_adapt():
    """Test TextVectorizer adapt method."""
    # Create a simple dataset
    texts = ["hello world", "test data", "machine learning"]
    labels = [0, 1, 0]
    ds_train = tf.data.Dataset.from_tensor_slices((texts, labels))

    # Initialize and adapt vectorizer
    vectorizer = TextVectorizer(max_tokens=100, output_sequence_length=10)
    vectorizer.adapt(ds_train)

    # Check that vocabulary was built
    vocab = vectorizer.text_vec.get_vocabulary()
    assert len(vocab) > 0
    assert "" in vocab  # padding token
    assert "[UNK]" in vocab  # unknown token


def test_text_vectorizer_vectorize_datasets():
    """Test TextVectorizer vectorize_datasets method."""
    # Create datasets
    texts_train = ["hello world", "test data"]
    labels_train = [1, 0]
    ds_train = tf.data.Dataset.from_tensor_slices((texts_train, labels_train))

    texts_valid = ["validation text"]
    labels_valid = [1]
    ds_valid = tf.data.Dataset.from_tensor_slices((texts_valid, labels_valid))

    texts_test = ["test text"]
    labels_test = [0]
    ds_test = tf.data.Dataset.from_tensor_slices((texts_test, labels_test))

    # Initialize, adapt, and vectorize
    vectorizer = TextVectorizer(max_tokens=100, output_sequence_length=10)
    vectorizer.adapt(ds_train)

    result = vectorizer.vectorize_datasets(ds_train, ds_valid, ds_test)
    ds_train_vec = result["train_data"]
    _ds_valid_vec = result["valid_data"]
    _ds_test_vec = result["test_data"]

    # Check that datasets are properly vectorized
    for batch in ds_train_vec.take(1):
        texts, labels = batch
        assert texts.shape[1] == 10  # output_sequence_length
        assert len(labels.shape) == 1
        break


def test_text_vectorizer_with_empty_texts():
    """Test TextVectorizer with empty texts."""
    texts = ["", "non-empty text"]
    labels = [0, 1]
    ds_train = tf.data.Dataset.from_tensor_slices((texts, labels))

    vectorizer = TextVectorizer(max_tokens=100, output_sequence_length=10)
    vectorizer.adapt(ds_train)

    # Vectorize and check it handles empty strings
    result = vectorizer.vectorize_datasets(ds_train, ds_train, ds_train)
    ds_train_vec = result["train_data"]

    for batch in ds_train_vec.take(1):
        texts, labels = batch
        assert texts.shape[0] > 0  # should have batches
        break


def test_text_vectorizer_vocab_size():
    """Test that vocabulary size respects max_tokens."""
    texts = [f"word{i}" for i in range(200)]
    labels = [i % 2 for i in range(200)]
    ds_train = tf.data.Dataset.from_tensor_slices((texts, labels))

    max_tokens = 50
    vectorizer = TextVectorizer(max_tokens=max_tokens, output_sequence_length=10)
    vectorizer.adapt(ds_train)

    vocab = vectorizer.text_vec.get_vocabulary()
    # Vocab includes padding and [UNK] tokens
    assert len(vocab) <= max_tokens


def test_text_vectorizer_sequence_length():
    """Test that sequences are properly truncated/padded."""
    texts = [
        "short",
        "this is a much longer text that should be truncated to the max length",
    ]
    labels = [0, 1]
    ds_train = tf.data.Dataset.from_tensor_slices((texts, labels))

    sequence_length = 5
    vectorizer = TextVectorizer(max_tokens=100, output_sequence_length=sequence_length)
    vectorizer.adapt(ds_train)

    result = vectorizer.vectorize_datasets(ds_train, ds_train, ds_train)
    ds_train_vec = result["train_data"]

    for batch in ds_train_vec.take(1):
        texts_vec, _ = batch
        assert texts_vec.shape[1] == sequence_length
        break
