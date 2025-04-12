from enum import Enum
from pathlib import Path


class DatasetPaths(str, Enum):
    """
    Enum for dataset-related paths.
    """

    RAW_DATA = Path("./data/tripadvisor_hotel_reviews.csv")
    PROCESSED_DATA = Path("./data/processed_data.csv")


class ModelPaths(str, Enum):
    """
    Enum for model-related paths.
    """

    TRAINED_MODEL = Path("./models/sentiment_keras_binary.keras")
    INFERENCE_MODEL = Path("./models/inference_model.keras")
    OPTUNA_MODEL = Path("./models/optuna_model_binary.json")
