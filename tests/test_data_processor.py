import pytest
import polars as pl
import tensorflow as tf
from typing import Dict, Any
from modules.data_processor import DatasetProcessor, TextPreprocessor


@pytest.fixture
def sample_data() -> pl.DataFrame:
    """
    Fixture to create a sample dataset.

    Returns:
        pl.DataFrame: A sample dataset with English and French text.
    """
    data: Dict[str, list[str]] = {
        "en": ["Hello|Hi", "Goodbye|Bye"],
        "fr": ["Bonjour|Salut", "Au revoir|Adieu"],
    }
    return pl.DataFrame(data)


def test_dataset_processor(sample_data: pl.DataFrame) -> None:
    """
    Test the DatasetProcessor class.

    Args:
        sample_data (pl.DataFrame): A sample dataset fixture.
    """
    # Save sample data to a temporary Parquet file
    sample_data.write_parquet("test_data.parquet")

    # Initialize and process the data
    processor: DatasetProcessor = DatasetProcessor("test_data.parquet")
    processor.load_data()
    processor.process_data()
    data_splits: Dict[str, pl.DataFrame] = processor.shuffle_and_split()

    # Check if the data is processed correctly
    assert len(processor.split_df) > 0, "Processed dataset is empty!"
    assert "fr" in processor.split_df.columns, "'fr' column is missing!"
    assert processor.split_df["fr"][0].startswith("[start]"), "Start token missing!"

    # Check if the splits are correct
    total_rows: int = len(processor.split_df)
    assert (
        len(data_splits["train"])
        + len(data_splits["validation"])
        + len(data_splits["test"])
        == total_rows
    ), "Splits do not add up to the total rows!"

    # Debugging: Print split sizes
    print(f"Train size: {len(data_splits['train'])}")
    print(f"Validation size: {len(data_splits['validation'])}")
    print(f"Test size: {len(data_splits['test'])}")

    # Cleanup
    import os

    os.remove("test_data.parquet")


def test_text_preprocessor(sample_data: pl.DataFrame) -> None:
    """
    Test the TextPreprocessor class.

    Args:
        sample_data (pl.DataFrame): A sample dataset fixture.
    """
    # Save sample data to a temporary Parquet file
    sample_data.write_parquet("test_data.parquet")

    # Initialize and process the data
    processor: DatasetProcessor = DatasetProcessor("test_data.parquet")
    processor.load_data()
    processor.process_data()
    data_splits: Dict[str, pl.DataFrame] = processor.shuffle_and_split()
    train_df: pl.DataFrame = data_splits["train"]

    # Initialize the TextPreprocessor
    preprocessor: TextPreprocessor = TextPreprocessor()
    preprocessor.adapt(train_df)

    # Check if vectorization works
    train_ds: tf.data.Dataset = preprocessor.make_dataset(train_df)
    for inputs, targets in train_ds.take(1):
        assert inputs["english"].shape[0] > 0, "English input is empty!"
        assert inputs["french"].shape[0] > 0, "French input is empty!"
        assert targets.shape[0] > 0, "Targets are empty!"

        # Debugging: Print shapes of inputs and targets
        print(f"English shape: {inputs['english'].shape}")
        print(f"French shape: {inputs['french'].shape}")
        print(f"Targets shape: {targets.shape}")

    # Cleanup
    import os

    os.remove("test_data.parquet")
