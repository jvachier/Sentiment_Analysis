from loading_kaggle_dataset_utils import (
    KaggleDatasetLoader,
    KaggleDatasetConfig,
)
import logging
from pydantic import ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Define the configuration
config = KaggleDatasetConfig(
    dataset_name="dhruvildave/en-fr-translation-dataset",
    file_path="en-fr.csv",
    output_path="src/data/en-fr.parquet",
)

# Initialize the KaggleDatasetLoader
loader = KaggleDatasetLoader(config=config)

# Process the dataset
try:
    loader.process()
except ValidationError as e:
    logging.error(f"Validation error: {e}")
except ValueError as e:
    logging.error(f"Processing error: {e}")
