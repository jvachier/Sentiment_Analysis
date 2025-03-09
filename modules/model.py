import tensorflow as tf


class SentimentModel:
    def __init__(
        self,
        embedding_dim=50,
        lstm_units=128,
        dropout_rate=0.5,
        learning_rate=1e-5,
        batch_size=32,
        epochs=20,
    ):
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def build_model(self, vocab_size, num_classes):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(None,), dtype="int64"))
        model.add(
            tf.keras.layers.Embedding(
                input_dim=vocab_size, output_dim=self.embedding_dim, name="embed-layer"
            )
        )
        model.add(
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    self.lstm_units, return_sequences=True, name="lstm-layer"
                ),
                name="bidir-lstm1",
            )
        )
        model.add(
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    self.lstm_units, return_sequences=False, name="lstm-layer"
                ),
                name="bidir-lstm2",
            )
        )
        model.add(tf.keras.layers.Dropout(self.dropout_rate))
        model.add(tf.keras.layers.Dense(128, activation="gelu"))
        model.add(tf.keras.layers.Dropout(self.dropout_rate))
        model.add(tf.keras.layers.Dense(64, activation="gelu"))
        model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )
        return model

    def train_and_evaluate(self, model, train_data, valid_data, test_data):
        model.summary()
        history = model.fit(train_data, validation_data=valid_data, epochs=self.epochs)
        test_results = model.evaluate(test_data)
        print("Test Acc.: {:.2f}%".format(test_results[1] * 100))
