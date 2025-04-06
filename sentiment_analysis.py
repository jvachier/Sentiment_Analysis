from modules.load_data import DataLoader
from modules.model import SentimentModelKeras
from modules.text_vecto import TextVectorizer
import os
import tensorflow as tf

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
    data_loader = DataLoader(data_path="./data/tripadvisor_hotel_reviews.csv")
    ds_raw, ds_raw_train, ds_raw_valid, ds_raw_test, target = data_loader.load_data()

    # Initialize the TextVectorization layer and adapt it to the training data
    with tf.device("/CPU:0"):
        # text_vectorizer = TextVectorizer(max_tokens=20000, output_sequence_length=500)
        # text_vectorizer.adapt(ds_raw_train)
        # vectorized_dataset = text_vectorizer.vectorize_datasets(
        #     ds_raw_train, ds_raw_valid, ds_raw_test
        # )
        # train_data, valid_data, test_data = vectorized_dataset
        text_vec = tf.keras.layers.TextVectorization(
            max_tokens=20000, output_mode="int", output_sequence_length=500
        )
        train_texts = ds_raw_train.map(lambda text, label: text)
        text_vec.adapt(train_texts)

        # Map the datasets using the TextVectorization layer
        ds_train = ds_raw_train.map(lambda text, label: (text_vec(text), label))
        ds_valid = ds_raw_valid.map(lambda text, label: (text_vec(text), label))
        ds_test = ds_raw_test.map(lambda text, label: (text_vec(text), label))

        # Batch and pad the datasets
        train_data = ds_train.padded_batch(32, padded_shapes=([-1], []))
        valid_data = ds_valid.padded_batch(32, padded_shapes=([-1], []))
        test_data = ds_test.padded_batch(32, padded_shapes=([-1], []))

    # Initialize the sentiment analysis model
    sentiment_keras = SentimentModelKeras()

    # Perform hyperparameter optimization using Optuna if enabled
    if OPTUNA and os.path.isfile("./models/optuna_model_binary.json") is False:
        sentiment_keras.Optuna(train_data, valid_data, test_data)

    # Train the model if it is not already saved
    if os.path.isfile("./models/sentiment_keras_binary.keras") is False:
        model = sentiment_keras.get_model()
        sentiment_keras.train_and_evaluate(model, train_data, valid_data, test_data)
    else:
        model = tf.keras.models.load_model("./models/sentiment_keras_binary.keras")

    # Create and save the inference model if it is not already saved
    if os.path.isfile("./models/inference_model.keras") is False:
        # inference_model = sentiment_keras.inference_model(model, text_vectorizer)
        inference_model = sentiment_keras.inference_model(model, text_vec)
        inference_model.save("./models/inference_model.keras")
    else:
        inference_model = tf.keras.models.load_model("./models/inference_model.keras")

    # Test the inference model with a sample input
    raw_text_data = tf.convert_to_tensor(["I hate this hotel, it was horrible"])
    predictions = inference_model(raw_text_data)

    print(f"{float(predictions[0] * 100):.2f}%")


if __name__ == "__main__":
    """
    Entry point of the script. Executes the main function.
    """
    main()
