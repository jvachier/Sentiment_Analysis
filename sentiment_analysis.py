import tensorflow as tf

# import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
from collections import Counter


# def pre_process_data():
#     return


def main():
    df = pd.read_csv("./data/tripadvisor_hotel_reviews.csv", encoding="utf-8")
    df.info()
    target = df.pop("Rating")
    ds_raw = tf.data.Dataset.from_tensor_slices((df["Review"].values, target.values))
    tf.random.set_seed(1)

    ds_raw = ds_raw.shuffle(len(df), reshuffle_each_iteration=False)

    ds_raw_test = ds_raw.take(int(len(df) * 0.2))
    ds_raw_train_valid = ds_raw.skip(int(len(df) * 0.2))
    ds_raw_train = ds_raw_train_valid.take(int(len(df) * 0.8))
    ds_raw_valid = ds_raw_train_valid.skip(int(len(df) * 0.8))

    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    reviews = [review.numpy().decode("utf-8") for review, _ in ds_raw]
    tokenizer.fit_on_texts(reviews)
    token_counts = Counter(
        token for text in reviews for token in tokenizer.texts_to_sequences([text])[0]
    )

    print("Vocab-size:", len(tokenizer.word_index))
    print("Token counts:", token_counts)

    tf_keras_encoded = tokenizer.texts_to_sequences(reviews)
    print(tf_keras_encoded)

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
