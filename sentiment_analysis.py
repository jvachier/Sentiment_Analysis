from modules.load_data import DataLoader
from modules.model import SentimentModelKeras
import os
import tensorflow as tf


def main():
    data_loader = DataLoader(data_path="./data/tripadvisor_hotel_reviews.csv")
    ds_raw, ds_raw_train, ds_raw_valid, ds_raw_test, target = data_loader.load_data()

    with tf.device("/CPU:0"):
        text_vec = tf.keras.layers.TextVectorization(
            max_tokens=20000, output_mode="int", output_sequence_length=500
        )
        # Adapt the TextVectorization layer to the training data
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
    sentiment_keras = SentimentModelKeras()
    if os.path.isfile("./models/sentiment_keras_binary.keras") is False:
        model = sentiment_keras.get_model()
        sentiment_keras.train_and_evaluate(model, train_data, valid_data, test_data)
    else:
        model = tf.keras.models.load_model("./models/sentiment_keras_binary.keras")

    if os.path.isfile("./models/inference_model.keras") is False:
        inference_model = sentiment_keras.inference_model(model, text_vec)
        inference_model.save("./models/inference_model.keras")
    else:
        inference_model = tf.keras.models.load_model("./models/inference_model.keras")

    # test
    raw_text_data = tf.convert_to_tensor(["I hate this hotel, it was horrible"])
    predictions = inference_model(raw_text_data)

    print(f"{float (predictions[0]*100):.2f}%")


if __name__ == "__main__":
    main()
