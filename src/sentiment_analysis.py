from modules.load_data import DataLoader
from modules.model import SentimentModelKeras
from modules.text_vecto import TextVectorizer
import os
import tensorflow as tf
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

OPTUNA = False


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
    sentiment_keras = SentimentModelKeras()

    # Perform hyperparameter optimization using Optuna if enabled
    if OPTUNA and os.path.isfile("./models/optuna_model_binary.json") is False:
        logging.info("Performing hyperparameter optimization using Optuna...")
        sentiment_keras.Optuna(train_data, valid_data, test_data)

    # Train the model if it is not already saved
    if os.path.isfile("./models/sentiment_keras_binary.keras") is False:
        logging.info("Training the sentiment analysis model...")
        model = sentiment_keras.get_model_api()
        sentiment_keras.train_and_evaluate(model, train_data, valid_data, test_data)
    else:
        logging.info("Loading the pre-trained sentiment analysis model...")
        model = tf.keras.models.load_model("./models/sentiment_keras_binary.keras")

    # Create and save the inference model if it is not already saved
    if os.path.isfile("./models/inference_model.keras") is False:
        logging.info("Creating and saving the inference model...")
        inference_model = sentiment_keras.inference_model(model, text_vec)
        inference_model.save("./models/inference_model.keras")
    else:
        logging.info("Loading the inference model...")
        inference_model = tf.keras.models.load_model("./models/inference_model.keras")

    # Test the inference model with a sample input
    logging.info("Testing the inference model with a sample input...")
    raw_text_data = tf.convert_to_tensor(["I hate this hotel, it was horrible"])
    predictions = inference_model(raw_text_data)

    logging.info(f"Prediction: {float(predictions[0] * 100):.2f}%")


if __name__ == "__main__":
    """
    Entry point of the script. Executes the main function.
    """
    logging.info("Starting the sentiment analysis pipeline...")
    main()
