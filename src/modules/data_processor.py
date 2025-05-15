import polars as pl
import tensorflow as tf
import string
import re
from typing import Tuple, Dict


class DatasetProcessor:
    """
    A class to handle dataset loading, processing, and splitting.
    """

    def __init__(self, file_path: str, delimiters: str = r"|"):
        """
        Initialize the DatasetProcessor class.

        Args:
            file_path (str): Path to the Parquet file.
            delimiters (str): Delimiters for splitting text columns.
        """
        self.file_path = file_path
        self.delimiters = delimiters
        self.split_df = None

    def load_data(self) -> None:
        """Load the Parquet file using Polars."""
        self.df = pl.read_parquet(self.file_path)

    def process_data(self) -> None:
        """Process the dataset by splitting, cleaning, and tokenizing."""
        # Split the 'en' column into rows based on delimiters
        if "en" in self.df.columns:
            en_split = self.df.select(pl.col("en").str.split(self.delimiters)).explode(
                "en"
            )

        # Split the 'fr' column into rows based on delimiters
        if "fr" in self.df.columns:
            fr_split = self.df.select(pl.col("fr").str.split(self.delimiters)).explode(
                "fr"
            )

        # Combine the split results into a new DataFrame
        self.split_df = pl.concat([en_split, fr_split], how="horizontal")

        # Remove rows with null values
        self.split_df = self.split_df.drop_nulls()

        # Add start and end tokens to French sentences
        self.split_df = self.split_df.with_columns(
            pl.col("fr")
            .map_elements(lambda fr: f"[start] {fr} [end]", return_dtype=pl.Utf8)
            .alias("fr")
        )

        # Ensure columns are strings
        self.split_df = self.split_df.with_columns(
            pl.col("en").cast(pl.Utf8),
            pl.col("fr").cast(pl.Utf8),
        )

    def shuffle_and_split(
        self,
        val_split: float = 0.15,
    ) -> Dict[str, pl.DataFrame]:
        """
        Shuffle and split the dataset into training, validation, and test sets.

        Args:
            val_split (float): Fraction of the dataset to use for validation.
            test_fraction (float): Fraction of the dataset to use for testing purposes.
            sample_fraction (float): Fraction of the dataset to sample for training.

        Returns:
            dict: A dictionary containing the training, validation, and test splits.
        """

        # Calculate the number of samples for validation and test sets
        num_val_samples = int(val_split * len(self.split_df))
        num_train_samples = len(self.split_df) - 2 * num_val_samples

        # Ensure no overlap and correct slicing
        self.train_df = self.split_df[:num_train_samples]
        self.val_df = self.split_df[
            num_train_samples : num_train_samples + num_val_samples
        ]
        self.test_df = self.split_df[num_train_samples + num_val_samples :]

        return {
            "train": self.train_df,
            "validation": self.val_df,
            "test": self.test_df,
        }


class TextPreprocessor:
    """
    A class to handle text vectorization and dataset creation.
    """

    def __init__(self, vocab_size: int = 20000, sequence_length: int = 20):
        """
        Initialize the TextPreprocessor class.

        Args:
            vocab_size (int): Maximum vocabulary size.
            sequence_length (int): Maximum sequence length.
        """
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

        # Define custom standardization function
        self.strip_chars = string.punctuation.replace("[", "").replace("]", "")

        def custom_standardization(input_string):
            lowercase = tf.strings.lower(input_string)
            return tf.strings.regex_replace(
                lowercase, f"[{re.escape(self.strip_chars)}]", ""
            )

        # Define TextVectorization layers
        self.source_vectorization = tf.keras.layers.TextVectorization(
            max_tokens=self.vocab_size,
            output_mode="int",
            output_sequence_length=self.sequence_length,
        )
        self.target_vectorization = tf.keras.layers.TextVectorization(
            max_tokens=self.vocab_size,
            output_mode="int",
            output_sequence_length=self.sequence_length + 1,
            standardize=custom_standardization,
        )

    def adapt(self, train_df: pl.DataFrame) -> None:
        """
        Adapt the vectorization layers to the training data.

        Args:
            train_df (pl.DataFrame): Training DataFrame with 'en' and 'fr' columns.
        """
        self.source_vectorization.adapt(train_df["en"].to_list())
        self.target_vectorization.adapt(train_df["fr"].to_list())

    def format_dataset(
        self, eng: tf.Tensor, fr: tf.Tensor
    ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
        """
        Format the dataset for training.

        Args:
            eng (tf.Tensor): English input tensor.
            fr (tf.Tensor): French input tensor.

        Returns:
            tuple: A tuple containing formatted inputs and targets.
        """
        eng = tf.cast(self.source_vectorization(eng), tf.int32)  # Ensure int32
        fr = tf.cast(self.target_vectorization(fr), tf.int32)  # Ensure int32
        return (
            {
                "english": eng,
                "french": fr[:, :-1],  # Input sequence for the decoder
            },
            fr[:, 1:],  # Target sequence for the decoder
        )

    def make_dataset(self, df, batch_size: int = 256) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset.

        Args:
            df (pl.DataFrame): DataFrame with 'en' and 'fr' columns.
            batch_size (int): Batch size for the dataset.

        Returns:
            tf.data.Dataset: A TensorFlow dataset.
        """
        dataset = tf.data.Dataset.from_tensor_slices(
            (df["en"].to_list(), df["fr"].to_list())
        )
        dataset = dataset.map(
            lambda en, fr: (tf.cast(en, tf.string), tf.cast(fr, tf.string)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(self.format_dataset, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset.shuffle(2048).prefetch(tf.data.AUTOTUNE)
