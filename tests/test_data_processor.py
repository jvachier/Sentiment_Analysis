import pytest
import polars as pl
from scripts.data_processor import DatasetProcessor, TextPreprocessor


@pytest.fixture
def sample_data():
    """Fixture to create a sample dataset."""
    data = {
        "en": ["Hello|Hi", "Goodbye|Bye"],
        "fr": ["Bonjour|Salut", "Au revoir|Adieu"],
    }
    return pl.DataFrame(data)


def test_dataset_processor(sample_data):
    """Test the DatasetProcessor class."""
    # Save sample data to a temporary Parquet file
    sample_data.write_parquet("test_data.parquet")

    # Initialize and process the data
    processor = DatasetProcessor("test_data.parquet")
    processor.load_data()
    processor.process_data()
    data_splits = processor.shuffle_and_split()

    # Check if the data is processed correctly
    assert len(processor.split_df) > 0
    assert "fr" in processor.split_df.columns
    assert processor.split_df["fr"][0].startswith("[start]")

    # Check if the splits are correct
    total_rows = len(processor.split_df)
    assert (
        len(data_splits["train"])
        + len(data_splits["validation"])
        + len(data_splits["test"])
        == total_rows
    )

    # Cleanup
    import os

    os.remove("test_data.parquet")


def test_text_preprocessor(sample_data):
    """Test the TextPreprocessor class."""
    # Save sample data to a temporary Parquet file
    sample_data.write_parquet("test_data.parquet")

    # Initialize and process the data
    processor = DatasetProcessor("test_data.parquet")
    processor.load_data()
    processor.process_data()
    data_splits = processor.shuffle_and_split()
    train_df = data_splits["train"]

    # Initialize the TextPreprocessor
    preprocessor = TextPreprocessor()
    preprocessor.adapt(train_df)

    # Check if vectorization works
    train_ds = preprocessor.make_dataset(train_df)
    for inputs, targets in train_ds.take(1):
        assert inputs["english"].shape[0] > 0
        assert inputs["french"].shape[0] > 0
        assert targets.shape[0] > 0

    # Cleanup
    import os

    os.remove("test_data.parquet")
