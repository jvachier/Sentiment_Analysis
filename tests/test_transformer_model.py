import pytest
import tensorflow as tf
from modules.data_processor import DatasetProcessor, TextPreprocessor
from modules.transformer_components import (
    PositionalEmbedding,
    TransformerEncoder,
    TransformerDecoder,
    evaluate_bleu,
)
from translation_french_english import transformer_model
from modules.utils import ModelPaths
import os


@pytest.fixture
def setup_data():
    """
    Fixture to set up a smaller dataset and preprocessor for testing.
    """
    processor = DatasetProcessor(file_path="src/data/en-fr.parquet")
    processor.load_data()
    processor.process_data()
    data_splits = processor.shuffle_and_split()
    train_df = data_splits["train"].sample(n=100)  # Use only 100 samples for training
    val_df = data_splits["validation"].sample(
        n=50
    )  # Use only 50 samples for validation
    test_df = data_splits["test"].sample(n=50)  # Use only 50 samples for testing

    preprocessor = TextPreprocessor()
    preprocessor.adapt(train_df)

    train_ds = preprocessor.make_dataset(train_df)
    val_ds = preprocessor.make_dataset(val_df)
    test_ds = preprocessor.make_dataset(test_df)

    return preprocessor, train_ds, val_ds, test_ds


def test_transformer_model_build(setup_data):
    """
    Test if the Transformer model is built correctly.
    """
    preprocessor, train_ds, val_ds, _ = setup_data
    transformer_model_path = "src/models/test_transformer_model.keras"

    # Build the model
    model = transformer_model(transformer_model_path, preprocessor, train_ds, val_ds)

    # Check if the model is compiled
    assert model.optimizer is not None, "Model is not compiled."
    assert model.loss is not None, "Loss function is not defined."
    assert model.metrics is not None, "Metrics are not defined."


def test_transformer_model_training(setup_data):
    """
    Test if the Transformer model can be trained without errors.
    """
    preprocessor, train_ds, val_ds, _ = setup_data
    transformer_model_path = "src/models/test_transformer_model.keras"

    # Build the model
    model = transformer_model(transformer_model_path, preprocessor, train_ds, val_ds)

    # Train the model for 1 epoch
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1,
        verbose=0,
    )

    # Check if training history is returned
    assert "loss" in history.history, "Training loss is not recorded."
    assert "val_loss" in history.history, "Validation loss is not recorded."


def test_transformer_model_evaluation(setup_data):
    """
    Test if the Transformer model can be evaluated without errors.
    """
    preprocessor, train_ds, val_ds, test_ds = setup_data
    transformer_model_path = "src/models/test_transformer_model.keras"

    # Build the model
    model = transformer_model(transformer_model_path, preprocessor, train_ds, val_ds)

    # Evaluate the model
    results = model.evaluate(test_ds, verbose=0)

    # Check if evaluation results are returned
    assert len(results) == 2, "Evaluation did not return loss and accuracy."
    assert results[0] >= 0, "Test loss is invalid."
    assert 0 <= results[1] <= 1, "Test accuracy is invalid."


def test_transformer_model_bleu_score(setup_data):
    """
    Test if the BLEU score can be calculated for the Transformer model.
    """
    preprocessor, train_ds, val_ds, test_ds = setup_data
    transformer_model_path = "src/models/test_transformer_model.keras"

    # Build the model
    model = transformer_model(transformer_model_path, preprocessor, train_ds, val_ds)

    # Calculate BLEU score
    bleu_score = evaluate_bleu(model, test_ds, preprocessor)

    # Check if BLEU score is valid
    assert 0 <= bleu_score <= 1, "BLEU score is invalid."


def test_transformer_model_loading(setup_data):
    """
    Test if the Transformer model can be loaded from a saved file.
    """
    preprocessor, train_ds, val_ds, _ = setup_data
    transformer_model_path = "src/models/test_transformer_model.keras"

    # Build and save the model
    model = transformer_model(transformer_model_path, preprocessor, train_ds, val_ds)
    model.save(transformer_model_path)

    # Load the model
    loaded_model = tf.keras.models.load_model(
        transformer_model_path,
        custom_objects={
            "PositionalEmbedding": PositionalEmbedding,
            "TransformerEncoder": TransformerEncoder,
            "TransformerDecoder": TransformerDecoder,
        },
    )

    # Check if the loaded model is valid
    assert loaded_model is not None, "Failed to load the Transformer model."
    assert loaded_model.optimizer is not None, "Loaded model is not compiled."
