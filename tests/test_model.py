import pytest
from src.modules.model import ModelTrainer, OptunaOptimizer, ModelBuilder


def test_model_trainer_initialization():
    """
    Test if ModelTrainer initializes with correct values.
    """
    trainer = ModelTrainer(learning_rate=0.001, epochs=10)
    assert trainer.learning_rate == 0.001
    assert trainer.epochs == 10


def test_optuna_optimizer_initialization():
    """
    Test if OptunaOptimizer initializes with correct values.
    """
    optimizer = OptunaOptimizer(max_token=15000, embedding_dim=100, dropout_rate=0.3)
    assert optimizer.max_token == 15000
    assert optimizer.embedding_dim == 100
    assert optimizer.dropout_rate == 0.3


def test_model_builder_initialization():
    """
    Test if ModelBuilder initializes with correct values.
    """
    builder = ModelBuilder(
        embedding_dim=64, lstm_units=256, dropout_rate=0.4, max_token=25000
    )
    assert builder.embedding_dim == 64
    assert builder.lstm_units == 256
    assert builder.dropout_rate == 0.4
    assert builder.max_token == 25000


def test_model_builder_get_model_api():
    """
    Test if ModelBuilder's get_model_api method returns a valid Keras model.
    """
    builder = ModelBuilder(
        embedding_dim=64, lstm_units=128, dropout_rate=0.5, max_token=20000
    )
    model = builder.get_model_api()
    assert model is not None
    assert len(model.layers) == 10  # Check the number of layers in the model
    assert model.input_shape == (None, None)  # Check input shape
    assert model.output_shape == (None, 1)  # Check output shape
