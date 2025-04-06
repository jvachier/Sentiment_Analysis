from modules.load_data import DataLoader

from modules.model import ModelBuilder, ModelTrainer, OptunaOptimizer
from modules.text_vecto import TextVectorizer
import os
import tensorflow as tf
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

OPTUNA = False


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
    model_path = "./models/sentiment_keras_binary.keras"
    if os.path.isfile(model_path):
        logging.info("Loading the pre-trained sentiment analysis model...")
        return tf.keras.models.load_model(model_path)

    logging.info("Training the sentiment analysis model...")
    model_builder = ModelBuilder()
    model = model_builder.get_model_api()
    trainer = ModelTrainer()
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
    inference_model_path = "./models/inference_model.keras"
    if os.path.isfile(inference_model_path):
        logging.info("Loading the inference model...")
        return tf.keras.models.load_model(inference_model_path)

    logging.info("Creating and saving the inference model...")
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
    logging.info("Performing hyperparameter optimization using Optuna...")
    optimiser = OptunaOptimizer()
    optimiser.Optuna(train_data, valid_data, test_data)


def main():
    """
    Main function to execute the sentiment analysis pipeline.

    Steps:
    1. Load and preprocess the dataset.
    2. Train the sentiment analysis model if not already trained.
    3. Perform hyperparameter optimization using Optuna (optional).
    4. Save and load the inference model.
    5. Test the inference model with a sample input.
    """
    # Load the dataset
    logging.info("Loading the dataset...")
    data_loader = DataLoader(data_path="./data/tripadvisor_hotel_reviews.csv")
    ds_raw, ds_raw_train, ds_raw_valid, ds_raw_test, target = data_loader.load_data()

    # Initialize the TextVectorization layer and adapt it to the training data
    logging.info("Initializing and adapting the TextVectorization layer...")
    with tf.device("/CPU:0"):
        text_vectorizer = TextVectorizer(max_tokens=20000, output_sequence_length=500)
        text_vectorizer.adapt(ds_raw_train)
        text_vec = text_vectorizer.get_text_vectorization_layer()
        vectorized_dataset = text_vectorizer.vectorize_datasets(
            ds_raw_train, ds_raw_valid, ds_raw_test
        )
        train_data = vectorized_dataset.get("train_data")
        valid_data = vectorized_dataset.get("valid_data")
        test_data = vectorized_dataset.get("test_data")

    # Initialize the sentiment analysis model
    logging.info("Initializing the sentiment analysis model...")

    # Perform hyperparameter optimization using Optuna if enabled
    if OPTUNA and os.path.isfile("./models/optuna_model_binary.json") is False:
        perform_hyperparameter_optimization(train_data, valid_data, test_data)

    # Train or load the model
    model = train_or_load_model(train_data, valid_data, test_data)

    # Create or load the inference model
    inference_model = create_or_load_inference_model(model, text_vec)

    # Test the inference model with a sample input
    logging.info("Testing the inference model with a sample input...")
    raw_text_data = tf.convert_to_tensor(["I hate this hotel, it was horrible"])
    predictions = inference_model(raw_text_data)

    prediction_percentage = float(predictions[0] * 100)
    if prediction_percentage < 50:
        logging.info(f"Prediction: {prediction_percentage:.2f}% - Negative Sentiment")
    else:
        logging.info(f"Prediction: {prediction_percentage:.2f}% - Positive Sentiment")


if __name__ == "__main__":
    """
    Entry point of the script. Executes the main function.
    """
    logging.info("Starting the sentiment analysis pipeline...")
    main()
