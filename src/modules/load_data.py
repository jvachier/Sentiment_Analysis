import pandas as pd
import tensorflow as tf
from pydantic import BaseModel, FilePath, Field, ValidationError


class DataLoaderConfig(BaseModel):
    """
    Configuration for the DataLoader class.
    """

    data_path: FilePath = Field(
        default="./data/tripadvisor_hotel_reviews.csv",
        description="Path to the CSV file containing the dataset.",
    )


class DataLoader:
    """
    A class to handle loading and preprocessing of sentiment analysis data.

    Attributes:
        data_path (str): Path to the CSV file containing the dataset.
    """

    def __init__(self, data_path: str):
        """
        Initialize the DataLoader class.

        Args:
            data_path (str): Path to the CSV file containing the dataset.
        """
        self.data_path = data_path

    def __post_init__(self):
        """
        Validate the data_path using Pydantic's DataLoaderConfig.
        """
        try:
            self.config = DataLoaderConfig(data_path=self.data_path)
        except ValidationError as e:
            raise ValueError(f"Invalid configuration: {e}")

    def load_data(self) -> dict:
        """
        Load and preprocess the dataset for sentiment analysis.

        The dataset is expected to have the following columns:
        - "Review": The text of the review.
        - "Rating": The numerical rating (used to derive sentiment labels).

        Steps:
        1. Load the dataset from the CSV file.
        2. Convert ratings into binary sentiment labels:
           - Ratings < 3 are labeled as 0 (negative sentiment).
           - Ratings >= 3 are labeled as 1 (positive sentiment).
        3. Shuffle the dataset and split it into training, validation, and test sets.

        Returns:
            tuple: A tuple containing:
                - ds_raw (tf.data.Dataset): The full dataset.
                - ds_raw_train (tf.data.Dataset): The training dataset.
                - ds_raw_valid (tf.data.Dataset): The validation dataset.
                - ds_raw_test (tf.data.Dataset): The test dataset.
                - target (pd.Series): The sentiment labels.
        """
        # Load the dataset
        df = pd.read_csv(self.data_path, encoding="utf-8")

        # Convert ratings into binary sentiment labels
        df.loc[df.Rating < 3, "Label"] = 0
        df.loc[df.Rating >= 3, "Label"] = 1
        df["Label"] = df["Label"].astype(int)
        df.drop(columns=["Rating"], inplace=True)

        # Extract the target labels
        target = df.pop("Label")

        # Create a TensorFlow dataset
        ds_raw = tf.data.Dataset.from_tensor_slices(
            (df["Review"].values, target.values)
        )
        ds_raw = ds_raw.shuffle(len(df), reshuffle_each_iteration=False, seed=42)

        # Split the dataset into training, validation, and test sets
        ds_raw_test = ds_raw.take(int(len(df) * 0.3))
        ds_raw_train_valid = ds_raw.skip(int(len(df) * 0.3))
        ds_raw_train = ds_raw_train_valid.take(int(len(ds_raw_train_valid) * 0.7))
        ds_raw_valid = ds_raw_train_valid.skip(int(len(ds_raw_train_valid) * 0.7))

        data_loaded = {
            "raw": ds_raw,
            "train": ds_raw_train,
            "valid": ds_raw_valid,
            "test": ds_raw_test,
            "target": target,
        }

        return data_loaded
