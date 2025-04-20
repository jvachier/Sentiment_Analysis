import logging
import kaggle
import kagglehub
from kagglehub import KaggleDatasetAdapter
from modules.mem_reduction import MemoryReducer
import pandas as pd
from pydantic import BaseModel, Field, validator


class KaggleDatasetConfig(BaseModel):
    """
    Pydantic model to validate Kaggle dataset configuration.
    """

    dataset_name: str = Field(
        default="devicharith/language-translation-englishfrench",
        description="The Kaggle dataset identifier (e.g., 'devicharith/language-translation-englishfrench/data').",
    )
    file_path: str = Field(
        default="eng_-french.csv",
        description="The file path within the Kaggle dataset to load (e.g., 'data.csv').",
    )
    output_path: str = Field(
        default="src/data/en-fr.parquet",
        description="The path to save the processed dataset (e.g., 'en-fr.parquet').",
    )
    required_columns: list[str] = Field(
        default=["English words/sentences", "French words/sentences"],
        description="The required columns in the dataset.",
    )

    @validator("dataset_name")
    def validate_dataset_name(cls, value):
        if not value or "/" not in value:
            raise ValueError(
                "Invalid dataset_name. It must be in the format 'owner/dataset-name'."
            )
        return value

    @validator("file_path")
    def validate_file_path(cls, value):
        if not value.endswith(".csv"):
            raise ValueError("Invalid file_path. It must point to a CSV file.")
        return value

    @validator("output_path")
    def validate_output_path(cls, value):
        if not value.endswith(".parquet"):
            raise ValueError("Invalid output_path. It must point to a Parquet file.")
        return value


class KaggleDatasetLoader:
    """
    A class to handle loading datasets from Kaggle and optimizing memory usage.
    """

    def __init__(self, config: KaggleDatasetConfig):
        """
        Initialize the KaggleDatasetLoader class.

        Args:
            config (KaggleDatasetConfig): The validated configuration for the Kaggle dataset.
        """
        self.config = config
        self.memory_reducer = MemoryReducer(verbose=True)

    def authenticate(self):
        """
        Authenticate with Kaggle using the kaggle.json file.
        """
        logging.info("Authenticating with Kaggle.")
        kaggle.api.authenticate()
        logging.info("Authentication successful.")

    def load_dataset(self) -> pd.DataFrame:
        """
        Load the dataset from Kaggle using KaggleHub.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        logging.info(
            f"Loading dataset: {self.config.dataset_name}, file: {self.config.file_path}"
        )
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            self.config.dataset_name,
            self.config.file_path,
        )
        logging.info("Dataset loaded successfully.")
        return df

    def validate_columns(self, df: pd.DataFrame) -> None:
        """
        Validate that the DataFrame contains only the required columns.

        Args:
            df (pd.DataFrame): The DataFrame to validate.

        Raises:
            ValueError: If the DataFrame does not contain the required columns.
        """
        logging.info("Validating dataset columns.")
        missing_columns = [
            col for col in self.config.required_columns if col not in df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        logging.info("Column validation successful.")

    def optimize_and_save(self, df: pd.DataFrame) -> None:
        """
        Optimize the dataset's memory usage and save it as a Parquet file.

        Args:
            df (pd.DataFrame): The DataFrame to optimize and save.
        """
        logging.info("Optimizing memory usage.")
        df = self.memory_reducer.reduce(df)

        # Rename columns to standardize them
        df = df.rename(
            columns={
                "English words/sentences": "en",
                "French words/sentences": "fr",
            }
        )

        logging.info(
            f"Memory optimization complete. Saving dataset to {self.config.output_path}"
        )
        df.to_parquet(self.config.output_path, engine="pyarrow")
        logging.info("Dataset saved successfully.")

    def process(self) -> None:
        """
        Authenticate, load, validate, optimize, and save the dataset.
        """
        logging.info("Starting dataset processing.")
        self.authenticate()
        df = self.load_dataset()
        self.validate_columns(df)
        self.optimize_and_save(df)
        logging.info("Dataset processing complete.")
