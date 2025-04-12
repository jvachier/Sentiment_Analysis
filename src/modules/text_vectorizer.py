import tensorflow as tf
from pydantic import BaseModel, Field, ValidationError


class TextVectorizerConfig(BaseModel):
    """
    Configuration for the TextVectorizer class.
    """

    max_tokens: int = Field(
        default=20000,
        ge=1,  # greater than or equal to
        description="Maximum number of tokens for the TextVectorization layer. Must be a positive integer.",
    )
    output_sequence_length: int = Field(
        default=500,
        ge=1,  # greater than or equal to
        description="Maximum sequence length for the TextVectorization layer. Must be a positive integer.",
    )


class TextVectorizer:
    """
    A class to handle text vectorization and dataset preparation for sentiment analysis.

    Attributes:
        max_tokens (int): Maximum number of tokens for the TextVectorization layer.
        output_sequence_length (int): Maximum sequence length for the TextVectorization layer.
        text_vec (tf.keras.layers.TextVectorization): The TextVectorization layer.
    """

    def __init__(self, max_tokens=20000, output_sequence_length=500):
        """
        Initialize the TextVectorizer class.

        Args:
            max_tokens (int): Maximum number of tokens for the TextVectorization layer.
            output_sequence_length (int): Maximum sequence length for the TextVectorization layer.
        """
        self.max_tokens = max_tokens
        self.output_sequence_length = output_sequence_length
        self.text_vec = tf.keras.layers.TextVectorization(
            max_tokens=self.max_tokens,
            output_mode="int",
            output_sequence_length=self.output_sequence_length,
        )

    def __post_init__(self):
        """
        Post-initialization method for the TextVectorizer class.

        This method is automatically called after the object is initialized. It attempts
        to create a `TextVectorizerConfig` instance using the provided `max_tokens` and
        `output_sequence_length` attributes. If the configuration is invalid, it raises
        a `ValueError` with details about the validation error.

        Raises:
            ValueError: If the configuration is invalid due to a `ValidationError`.
        """
        try:
            self.config = TextVectorizerConfig(
                max_tokens=self.max_tokens,
                output_sequence_length=self.output_sequence_length,
            )
        except ValidationError as e:
            raise ValueError(f"Invalid configuration: {e}")

    def adapt(self, ds_raw_train: tf.data.Dataset) -> None:
        """
        Adapt the TextVectorization layer to the training dataset.

        Args:
            ds_raw_train (tf.data.Dataset): The raw training dataset.
        """
        train_texts = ds_raw_train.map(lambda text, label: text)
        self.text_vec.adapt(train_texts)

    def vectorize_datasets(
        self,
        ds_raw_train: tf.data.Dataset,
        ds_raw_valid: tf.data.Dataset,
        ds_raw_test: tf.data.Dataset,
    ) -> dict:
        """
        Vectorize the datasets using the TextVectorization layer.

        Args:
            ds_raw_train (tf.data.Dataset): The raw training dataset.
            ds_raw_valid (tf.data.Dataset): The raw validation dataset.
            ds_raw_test (tf.data.Dataset): The raw test dataset.

        Returns:
            dict: A dictionary containing the vectorized and batched datasets.
        """
        # Map the datasets using the TextVectorization layer
        ds_train = ds_raw_train.map(lambda text, label: (self.text_vec(text), label))
        ds_valid = ds_raw_valid.map(lambda text, label: (self.text_vec(text), label))
        ds_test = ds_raw_test.map(lambda text, label: (self.text_vec(text), label))

        # Batch and pad the datasets
        train_data = ds_train.padded_batch(32, padded_shapes=([-1], []))
        valid_data = ds_valid.padded_batch(32, padded_shapes=([-1], []))
        test_data = ds_test.padded_batch(32, padded_shapes=([-1], []))

        vectorized_dataset = {
            "train_data": train_data,
            "valid_data": valid_data,
            "test_data": test_data,
        }

        return vectorized_dataset

    def get_text_vectorization_layer(self) -> tf.keras.layers.TextVectorization:
        """
        Retrieve the TextVectorization layer.

        Returns:
            tf.keras.layers.TextVectorization: The TextVectorization layer.
        """
        return self.text_vec
