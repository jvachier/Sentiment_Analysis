from enum import Enum
from pathlib import Path


class DatasetPaths(str, Enum):
    """
    Enum for dataset-related paths.
    """

    RAW_DATA = str(Path("src/data/tripadvisor_hotel_reviews.csv"))


class ModelPaths(str, Enum):
    """
    Enum for model-related paths.
    """

    MODEL_BUILDER_CONFIG = str(Path("src/configurations/model_builder_config.json"))
    MODEL_TRAINER_CONFIG = str(Path("src/configurations/model_trainer_config.json"))
    TRAINED_MODEL = str(Path("src/models/sentiment_keras_binary.keras"))
    INFERENCE_MODEL = str(Path("src/models/inference_model.keras"))
    TRANSFORMER_MODEL = str(Path("src/models/transformer_best_model.keras"))


class OptunaPaths(str, Enum):
    """
    Enum for Optuna-related paths.
    """

    OPTUNA_CONFIG = str(Path("src/configurations/optuna_config.json"))
    OPTUNA_MODEL = str(Path("src/models/optuna_model_binary.json"))


class TextVectorizerConfig(int, Enum):
    """
    Enum for TextVectorizer.
    """

    max_tokens = 20000
    output_sequence_length = 500
