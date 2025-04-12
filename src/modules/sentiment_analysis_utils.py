from modules.model import ModelBuilder, ModelTrainer, OptunaOptimizer
from modules.utils import ModelPaths, OptunaPaths
import os
import tensorflow as tf
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def train_or_load_model(
    train_data: tf.data.Dataset,
    valid_data: tf.data.Dataset,
    test_data: tf.data.Dataset,
) -> tf.keras.Model:
    """
    Train the model if not already saved, otherwise load the pre-trained model.

    Args:
        train_data (tf.data.Dataset): Training dataset.
        valid_data (tf.data.Dataset): Validation dataset.
        test_data (tf.data.Dataset): Test dataset.
        model_builder (ModelBuilder): Instance of the ModelBuilder class.

    Returns:
        tf.keras.Model: The trained or loaded model.
    """
    model_path = ModelPaths.TRAINED_MODEL.value
    if os.path.isfile(model_path):
        logging.info("Loading the pre-trained sentiment analysis model.")
        return tf.keras.models.load_model(model_path)

    logging.info("Training the sentiment analysis model.")
    model_builder = ModelBuilder(config_path=ModelPaths.MODEL_BUILDER_CONFIG.value)
    model = model_builder.get_model_api()
    trainer = ModelTrainer(config_path=ModelPaths.MODEL_TRAINER_CONFIG.value)
    trainer.train_and_evaluate(model, train_data, valid_data, test_data)
    return model


def create_or_load_inference_model(
    model: tf.keras.Model, text_vec: tf.keras.layers.TextVectorization
) -> tf.keras.Model:
    """
    Create and save the inference model if not already saved, otherwise load it.

    Args:
        model (tf.keras.Model): The trained model.
        text_vec (tf.keras.layers.TextVectorization): Text vectorization layer.

    Returns:
        tf.keras.Model: The inference model.
    """
    inference_model_path = ModelPaths.INFERENCE_MODEL.value
    if os.path.isfile(inference_model_path):
        logging.info("Loading the inference model.")
        return tf.keras.models.load_model(inference_model_path)

    logging.info("Creating and saving the inference model.")
    trainer = ModelTrainer()
    inference_model = trainer.inference_model(model, text_vec)
    inference_model.save(inference_model_path)
    return inference_model


def perform_hyperparameter_optimization(
    train_data: tf.data.Dataset,
    valid_data: tf.data.Dataset,
    test_data: tf.data.Dataset,
) -> None:
    """
    Perform hyperparameter optimization using Optuna.

    Args:
        train_data (tf.data.Dataset): Training dataset.
        valid_data (tf.data.Dataset): Validation dataset.
        test_data (tf.data.Dataset): Test dataset.
    """
    logging.info("Performing hyperparameter optimization using Optuna.")
    optimiser = OptunaOptimizer(
        config_path=OptunaPaths.OPTUNA_CONFIG.value,
    )
    optimiser.optimize(train_data, valid_data, test_data)
