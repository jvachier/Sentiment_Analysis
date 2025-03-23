import tensorflow as tf
from transformers import TFBertModel, BertTokenizer  # keras 2
import optuna
import json
import numpy as np


class SentimentModelKeras:
    def __init__(
        self,
        embedding_dim=50,
        lstm_units=128,
        dropout_rate=0.5,
        learning_rate=0.0008659430202504234,
        epochs=5,
        max_token=20000,
    ):
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_token = max_token

    def get_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(None,), dtype="int32"))
        model.add(
            tf.keras.layers.Embedding(
                input_dim=self.max_token,
                output_dim=self.embedding_dim,
                name="embed-layer",
            )
        )
        model.add(
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(80, return_sequences=True, name="lstm-layer"),
                name="bidir-lstm1",
            )
        )
        model.add(
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(121, return_sequences=False, name="lstm-layer"),
                name="bidir-lstm2",
            )
        )
        model.add(tf.keras.layers.Dropout(self.dropout_rate))
        model.add(tf.keras.layers.Dense(67, activation="gelu"))
        model.add(tf.keras.layers.Dense(75, activation="gelu"))
        model.add(tf.keras.layers.Dropout(self.dropout_rate))
        model.add(tf.keras.layers.Dense(32, activation="gelu"))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        model.compile(
            optimizer=tf.keras.optimizers.legacy.RMSprop(
                learning_rate=self.learning_rate
            ),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )
        return model

    def inference_model(self, model, text_vec):
        inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
        process_inputs = text_vec(inputs)
        outputs = model(process_inputs)
        inference_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return inference_model

    def train_and_evaluate(self, model, train_data, valid_data, test_data):
        model.summary()
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            mode="min",
            verbose=0,
            restore_best_weights=True,
        )
        callbacks_model = tf.keras.callbacks.ModelCheckpoint(
            filepath="./models/sentiment_keras_binary.keras",
            save_best_only=True,
        )
        with tf.device("/device:GPU:0"):
            model.fit(
                train_data,
                validation_data=valid_data,
                epochs=self.epochs,
                callbacks=[early_stopping_callback, callbacks_model],
            )
            test_results = model.evaluate(test_data)
        print("Test Acc.: {:.2f}%".format(test_results[1] * 100))
