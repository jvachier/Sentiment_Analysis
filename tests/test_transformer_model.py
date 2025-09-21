import pytest
import tensorflow as tf
import pandas as pd
from typing import Tuple
from src.modules.data_processor import TextPreprocessor
from src.modules.transformer_components import (
    PositionalEmbedding,
    TransformerEncoder,
    TransformerDecoder,
    evaluate_bleu,
)
from src.translation_french_english import transformer_model


@pytest.fixture
def setup_data() -> Tuple[
    TextPreprocessor, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset
]:
    """
    Fixture to set up a mocked dataset and preprocessor for testing.

    Returns:
        Tuple containing:
            - TextPreprocessor: The initialized text preprocessor
            - tf.data.Dataset: Training dataset
            - tf.data.Dataset: Validation dataset
            - tf.data.Dataset: Test dataset
    """

    # Create a small mock dataset
    mock_data: dict[str, list[str]] = {
        "en": ["hello", "how are you", "good morning", "thank you", "goodbye"],
        "fr": ["bonjour", "comment ça va", "bon matin", "merci", "au revoir"],
    }
    mock_df = pd.DataFrame(mock_data)

    # Split the mock dataset
    train_df: pd.DataFrame = mock_df.sample(frac=0.6, random_state=42)
    val_df: pd.DataFrame = mock_df.drop(train_df.index).sample(
        frac=0.5, random_state=42
    )
    test_df: pd.DataFrame = mock_df.drop(train_df.index).drop(val_df.index)

    # Initialize the preprocessor
    preprocessor: TextPreprocessor = TextPreprocessor()
    preprocessor.adapt(train_df)

    # Create TensorFlow datasets
    train_ds: tf.data.Dataset = preprocessor.make_dataset(train_df)
    val_ds: tf.data.Dataset = preprocessor.make_dataset(val_df)
    test_ds: tf.data.Dataset = preprocessor.make_dataset(test_df)

    return preprocessor, train_ds, val_ds, test_ds


def test_transformer_model_build(
    setup_data: Tuple[
        TextPreprocessor, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset
    ],
) -> None:
    """
    Test if the Transformer model is built correctly.

    Args:
        setup_data: Fixture providing preprocessor and datasets
    """
    preprocessor, train_ds, val_ds, _ = setup_data
    transformer_model_path = "src/models/test_transformer_model.keras"

    # Build the model
    model: tf.keras.Model = transformer_model(
        transformer_model_path, preprocessor, train_ds, val_ds
    )

    # Check if the model is compiled
    assert model.optimizer is not None, "Model is not compiled."
    assert model.loss is not None, "Loss function is not defined."
    assert model.metrics is not None, "Metrics are not defined."


def test_transformer_model_training(
    setup_data: Tuple[
        TextPreprocessor, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset
    ],
) -> None:
    """
    Test if the Transformer model can be trained without errors.

    Args:
        setup_data: Fixture providing preprocessor and datasets
    """
    preprocessor, train_ds, val_ds, _ = setup_data
    transformer_model_path = "src/models/test_transformer_model.keras"

    # Build the model
    model: tf.keras.Model = transformer_model(
        transformer_model_path, preprocessor, train_ds, val_ds
    )

    # Train the model for 1 epoch
    history: tf.keras.callbacks.History = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1,
        verbose=0,
    )

    # Check if training history is returned
    assert "loss" in history.history, "Training loss is not recorded."
    assert "val_loss" in history.history, "Validation loss is not recorded."


def test_transformer_model_evaluation(
    setup_data: Tuple[
        TextPreprocessor, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset
    ],
) -> None:
    """
    Test if the Transformer model can be evaluated without errors.

    Args:
        setup_data: Fixture providing preprocessor and datasets
    """
    preprocessor, train_ds, val_ds, test_ds = setup_data
    transformer_model_path = "src/models/test_transformer_model.keras"

    # Build the model
    model: tf.keras.Model = transformer_model(
        transformer_model_path, preprocessor, train_ds, val_ds
    )

    # Evaluate the model
    results: list[float] = model.evaluate(test_ds, verbose=0)

    # Check if evaluation results are returned
    assert len(results) == 2, "Evaluation did not return loss and accuracy."
    assert results[0] >= 0, "Test loss is invalid."
    assert 0 <= results[1] <= 1, "Test accuracy is invalid."


def test_transformer_model_bleu_score(
    setup_data: Tuple[
        TextPreprocessor, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset
    ],
) -> None:
    """
    Test if the BLEU score can be calculated for the Transformer model.

    Args:
        setup_data: Fixture providing preprocessor and datasets
    """
    preprocessor, train_ds, val_ds, test_ds = setup_data
    transformer_model_path = "src/models/test_transformer_model.keras"

    # Build the model
    model: tf.keras.Model = transformer_model(
        transformer_model_path, preprocessor, train_ds, val_ds
    )

    # Calculate BLEU score
    bleu_score: float = evaluate_bleu(model, test_ds, preprocessor)

    # Check if BLEU score is valid
    assert 0 <= bleu_score <= 1, "BLEU score is invalid."


def test_transformer_model_loading(
    setup_data: Tuple[
        TextPreprocessor, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset
    ],
) -> None:
    """
    Test if the Transformer model can be loaded from a saved file.

    Args:
        setup_data: Fixture providing preprocessor and datasets
    """
    preprocessor, train_ds, val_ds, _ = setup_data
    transformer_model_path = "src/models/test_transformer_model.keras"

    # Build and save the model
    model: tf.keras.Model = transformer_model(
        transformer_model_path, preprocessor, train_ds, val_ds
    )
    model.save(transformer_model_path)

    # Load the model
    loaded_model: tf.keras.Model = tf.keras.models.load_model(
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
