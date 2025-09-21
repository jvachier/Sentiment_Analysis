import json
from src.modules.model_sentiment_analysis import (
    ModelTrainer,
    OptunaOptimizer,
    ModelBuilder,
)
from src.modules.utils import ModelPaths, OptunaPaths


def test_model_trainer_initialization():
    """
    Test if ModelTrainer initializes with correct values from the configuration file.
    """
    with open(ModelPaths.MODEL_TRAINER_CONFIG.value, "r") as config_file:
        config = json.load(config_file)

    trainer = ModelTrainer(config_path=ModelPaths.MODEL_TRAINER_CONFIG.value)
    assert trainer.learning_rate == config["learning_rate"]
    assert trainer.epochs == config["epochs"]


def test_optuna_optimizer_initialization():
    """
    Test if OptunaOptimizer initializes with correct values from the configuration file.
    """
    with open(OptunaPaths.OPTUNA_CONFIG.value, "r") as config_file:
        config = json.load(config_file)

    optimizer = OptunaOptimizer(config_path=OptunaPaths.OPTUNA_CONFIG.value)
    assert optimizer.max_token == config["max_token"]
    assert optimizer.embedding_dim == config["embedding_dim"]
    assert optimizer.dropout_rate == config["dropout_rate"]


def test_model_builder_initialization():
    """
    Test if ModelBuilder initializes with correct values from the configuration file.
    """
    with open(ModelPaths.MODEL_BUILDER_CONFIG.value, "r") as config_file:
        config = json.load(config_file)

    builder = ModelBuilder(config_path=ModelPaths.MODEL_BUILDER_CONFIG.value)
    assert builder.embedding_dim == config["embedding_dim"]
    assert builder.lstm_units == config["lstm_units"]
    assert builder.dropout_rate == config["dropout_rate"]
    assert builder.max_token == config["max_token"]


def test_model_builder_get_model_api():
    """
    Test if ModelBuilder's get_model_api method returns a valid Keras model.
    """
    builder = ModelBuilder(config_path=ModelPaths.MODEL_BUILDER_CONFIG.value)
    model = builder.get_model_api()
    assert model is not None
    assert len(model.layers) > 0  # Check that the model has layers
    assert model.input_shape == (None, None)  # Check input shape
    assert model.output_shape == (None, 1)  # Check output shape
