from modules.load_data import DataLoader
from modules.data_preprocess import TextPreprocessor
from modules.model import SentimentModel
# def preprocess_text(text):
#     # Tokenize the text
#     tokens = word_tokenize(text)
#     # Convert to lower case
#     tokens = [token.lower() for token in tokens]
#     # Remove punctuation
#     tokens = [token for token in tokens if token.isalnum()]
#     # Remove stop words
#     tokens = [token for token in tokens if token not in stop_words]
#     # Lemmatize the tokens
#     tokens = [lemmatizer.lemmatize(token) for token in tokens]
#     return " ".join(tokens)


# def encode_map_fn(text, label):
#     return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))


# def encode(text_tensor, label):
#     text = text_tensor.numpy().decode("utf-8")
#     text = preprocess_text(text)  # Preprocess the text
#     encoded_text = tokenizer.texts_to_sequences([text])[0]
#     return encoded_text, label


# def main():
#     global tokenizer  # Use the global tokenizer variable
#     df = pd.read_csv("./data/tripadvisor_hotel_reviews.csv", encoding="utf-8")
#     target = df.pop("Rating")
#     ds_raw = tf.data.Dataset.from_tensor_slices((df["Review"].values, target.values))

#     tf.random.set_seed(1)

#     ds_raw = ds_raw.shuffle(len(df), reshuffle_each_iteration=False)

#     ds_raw_test = ds_raw.take(int(len(df) * 0.2))
#     ds_raw_train_valid = ds_raw.skip(int(len(df) * 0.2))
#     ds_raw_train = ds_raw_train_valid.take(int(len(ds_raw_train_valid) * 0.8))
#     ds_raw_valid = ds_raw_train_valid.skip(int(len(ds_raw_train_valid) * 0.8))

#     tokenizer = tf.keras.preprocessing.text.Tokenizer()
#     reviews = [review.numpy().decode("utf-8") for review, _ in ds_raw]
#     reviews = [preprocess_text(review) for review in reviews]  # Preprocess the reviews
#     tokenizer.fit_on_texts(reviews)
#     token_counts = Counter(
#         token for text in reviews for token in tokenizer.texts_to_sequences([text])[0]
#     )

#     # print("Vocab-size:", len(tokenizer.word_index))
#     # print("Token counts:", token_counts)

#     ds_train = ds_raw_train.map(lambda text, label: encode_map_fn(text, label))
#     ds_valid = ds_raw_valid.map(lambda text, label: encode_map_fn(text, label))
#     ds_test = ds_raw_test.map(lambda text, label: encode_map_fn(text, label))

#     # for example in ds_train.shuffle(100).take(5):
#     #     print("Sequence length:", example[0].shape)

#     ## batching the datasets
#     train_data = ds_train.padded_batch(32, padded_shapes=([-1], []))
#     valid_data = ds_valid.padded_batch(32, padded_shapes=([-1], []))
#     test_data = ds_test.padded_batch(32, padded_shapes=([-1], []))

#     embedding_dim = 50
#     vocab_size = len(token_counts) + 2
#     num_classes = target.nunique()  # Number of unique classes in the target

#     ## build the model
#     bi_lstm_model = tf.keras.Sequential()
#     bi_lstm_model.add(tf.keras.Input(shape=(None,), dtype="int64"))
#     bi_lstm_model.add(
#         tf.keras.layers.Embedding(
#             input_dim=vocab_size, output_dim=embedding_dim, name="embed-layer"
#         )
#     )
#     bi_lstm_model.add(
#         tf.keras.layers.Bidirectional(
#             tf.keras.layers.LSTM(128, return_sequences=True, name="lstm-layer"),
#             name="bidir-lstm1",
#         )
#     )
#     bi_lstm_model.add(
#         tf.keras.layers.Bidirectional(
#             tf.keras.layers.LSTM(128, return_sequences=False, name="lstm-layer"),
#             name="bidir-lstm2",
#         )
#     )
#     bi_lstm_model.add(tf.keras.layers.Dropout(0.5))
#     bi_lstm_model.add(tf.keras.layers.Dense(128, activation="gelu"))
#     bi_lstm_model.add(tf.keras.layers.Dropout(0.5))
#     bi_lstm_model.add(tf.keras.layers.Dense(64, activation="gelu"))
#     bi_lstm_model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
#     bi_lstm_model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#         metrics=["accuracy"],
#     )

#     bi_lstm_model.summary()

#     history = bi_lstm_model.fit(train_data, validation_data=valid_data, epochs=20)

#     ## evaluate on the test data
#     test_results = bi_lstm_model.evaluate(test_data)
#     print("Test Acc.: {:.2f}%".format(test_results[1] * 100))


def main():
    data_loader = DataLoader(data_path="./data/tripadvisor_hotel_reviews.csv")
    ds_raw, ds_raw_train, ds_raw_valid, ds_raw_test, target = data_loader.load_data()

    text_preprocessor = TextPreprocessor()
    token_counts = text_preprocessor.fit_tokenizer(ds_raw)

    ds_train = ds_raw_train.map(
        lambda text, label: text_preprocessor.encode_map_fn(text, label)
    )
    ds_valid = ds_raw_valid.map(
        lambda text, label: text_preprocessor.encode_map_fn(text, label)
    )
    ds_test = ds_raw_test.map(
        lambda text, label: text_preprocessor.encode_map_fn(text, label)
    )

    train_data = ds_train.padded_batch(32, padded_shapes=([-1], []))
    valid_data = ds_valid.padded_batch(32, padded_shapes=([-1], []))
    test_data = ds_test.padded_batch(32, padded_shapes=([-1], []))

    vocab_size = len(token_counts) + 2
    num_classes = target.nunique()

    sentiment_model = SentimentModel()
    model = sentiment_model.build_model(vocab_size, num_classes)
    sentiment_model.train_and_evaluate(model, train_data, valid_data, test_data)


if __name__ == "__main__":
    main()
