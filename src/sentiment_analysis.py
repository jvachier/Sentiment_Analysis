from dotenv import load_dotenv
from src.modules.load_data import DataLoader

from src.modules.text_vectorizer_sentiment_analysis import TextVectorizer
from src.modules.utils import DatasetPaths, OptunaPaths, TextVectorizerConfig
from src.modules.sentiment_analysis_utils import (
    train_or_load_model,
    create_or_load_inference_model,
    perform_hyperparameter_optimization,
)
import os
import tensorflow as tf
import logging

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main() -> None:
    """
    Main function to execute the sentiment analysis pipeline.

    Steps:
    1. Load and preprocess the dataset.
    2. Train the sentiment analysis model if not already trained.
    3. Perform hyperparameter optimization using Optuna (optional).
    4. Save and load the inference model.
    5. Test the inference model with a sample input.
    """
    # Get OPTUNA value from environment variables
    OPTUNA = os.getenv("OPTUNA", "False").lower() == "true"

    # Load the dataset
    logging.info("Loading the dataset.")
    data_path = DatasetPaths.RAW_DATA.value
    data_loader = DataLoader(data_path)
    datasets = data_loader.load_data()
    ds_raw_train = datasets["train"]
    ds_raw_valid = datasets["valid"]
    ds_raw_test = datasets["test"]

    # Initialize the TextVectorization layer and adapt it to the training data
    logging.info("Initializing and adapting the TextVectorization layer.")
    with tf.device("/CPU:0"):
        text_vectorizer = TextVectorizer(
            max_tokens=TextVectorizerConfig.max_tokens.value,
            output_sequence_length=TextVectorizerConfig.output_sequence_length.value,
        )
        text_vectorizer.adapt(ds_raw_train)
        text_vec = text_vectorizer.get_text_vectorization_layer()
        vectorized_dataset = text_vectorizer.vectorize_datasets(
            ds_raw_train, ds_raw_valid, ds_raw_test
        )
        train_data = vectorized_dataset.get("train_data")
        valid_data = vectorized_dataset.get("valid_data")
        test_data = vectorized_dataset.get("test_data")

    # Initialize the sentiment analysis model
    logging.info("Initializing the sentiment analysis model.")

    # Perform hyperparameter optimization using Optuna if enabled
    if OPTUNA and os.path.isfile(OptunaPaths.OPTUNA_MODEL.value) is False:
        perform_hyperparameter_optimization(train_data, valid_data, test_data)

    # Train or load the model
    model = train_or_load_model(train_data, valid_data, test_data)

    # Create or load the inference model
    inference_model = create_or_load_inference_model(model, text_vec)

    # Test the inference model with a sample input
    logging.info("Testing the inference model with a sample input.")
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
    logging.info("Starting the sentiment analysis pipeline.")
    main()
