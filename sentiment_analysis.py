import tensorflow as tf

# import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
from collections import Counter

# def pre_process_data():
#     return


def encode_map_fn(text, label):
    return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))


def encode(text_tensor, label):
    text = text_tensor.numpy().decode("utf-8")
    encoded_text = tokenizer.texts_to_sequences([text])[0]
    return encoded_text, label


def main():
    global tokenizer  # Use the global tokenizer variable
    df = pd.read_csv("./data/tripadvisor_hotel_reviews.csv", encoding="utf-8")
    target = df.pop("Rating")
    ds_raw = tf.data.Dataset.from_tensor_slices((df["Review"].values, target.values))
    tf.random.set_seed(1)

    ds_raw = ds_raw.shuffle(len(df), reshuffle_each_iteration=False)

    ds_raw_test = ds_raw.take(int(len(df) * 0.5))
    ds_raw_train_valid = ds_raw.skip(int(len(df) * 0.5))
    ds_raw_train = ds_raw_train_valid.take(5000)
    ds_raw_valid = ds_raw_train_valid.skip(5000)

    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    reviews = [review.numpy().decode("utf-8") for review, _ in ds_raw]
    tokenizer.fit_on_texts(reviews)
    token_counts = Counter(
        token for text in reviews for token in tokenizer.texts_to_sequences([text])[0]
    )

    # print("Vocab-size:", len(tokenizer.word_index))
    # print("Token counts:", token_counts)

    ds_train = ds_raw_train.map(lambda text, label: encode_map_fn(text, label))
    ds_valid = ds_raw_valid.map(lambda text, label: encode_map_fn(text, label))
    ds_test = ds_raw_test.map(lambda text, label: encode_map_fn(text, label))

    # for example in ds_train.shuffle(100).take(5):
    #     print("Sequence length:", example[0].shape)

    ## batching the datasets
    train_data = ds_train.padded_batch(16, padded_shapes=([-1], []))

    valid_data = ds_valid.padded_batch(16, padded_shapes=([-1], []))

    test_data = ds_test.padded_batch(16, padded_shapes=([-1], []))

    embedding_dim = 20
    vocab_size = len(token_counts) + 2
    num_classes = target.nunique()  # Number of unique classes in the target

    ## build the model
    bi_lstm_model = tf.keras.Sequential()
    bi_lstm_model.add(
        tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embedding_dim, name="embed-layer"
        )
    )
    bi_lstm_model.add(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True, name="lstm-layer"),
            name="bidir-lstm1",
        )
    )
    bi_lstm_model.add(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=False, name="lstm-layer"),
            name="bidir-lstm2",
        )
    )
    bi_lstm_model.add(tf.keras.layers.Dense(64, activation="gelu"))
    bi_lstm_model.add(tf.keras.layers.Dense(32, activation="gelu"))
    bi_lstm_model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
    bi_lstm_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    bi_lstm_model.summary()

    history = bi_lstm_model.fit(train_data, validation_data=valid_data, epochs=10)

    ## evaluate on the test data
    test_results = bi_lstm_model.evaluate(test_data)
    print("Test Acc.: {:.2f}%".format(test_results[1] * 100))

    # tokenizer = tf.keras.preprocessing.text.Tokenizer()
    # tokenizer.fit_on_texts(df['Review'])
    # token_counts = Counter(token for text in df['Review'] for token in tokenizer.texts_to_sequences([text])[0])

    # print('Vocab-size:', len(tokenizer.word_index))
    # print('Token counts:', token_counts)

    # tf_keras_encoded = tokenizer.texts_to_sequences(df['Review'])
    # print(tf_keras_encoded)

    # tf.keras.models.Sequential()


if __name__ == "__main__":
    main()
