import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
import optuna
import json


class SentimentModelBert:
    """
    A class to define, train, and evaluate a sentiment analysis model using BERT.

    Attributes:
        model_name (str): The name of the pre-trained BERT model.
        max_length (int): The maximum sequence length for tokenization.
        learning_rate (float): The learning rate for the optimizer.
        batch_size (int): The batch size for training.
        epochs (int): The number of training epochs.
        tokenizer (BertTokenizer): The tokenizer for the BERT model.
    """

    def __init__(
        self,
        model_name="bert-base-uncased",
        max_length=128,
        learning_rate=1e-4,
        batch_size=32,
        epochs=5,
    ):
        """
        Initialize the SentimentModelBert class.

        Args:
            model_name (str): The name of the pre-trained BERT model.
            max_length (int): The maximum sequence length for tokenization.
            learning_rate (float): The learning rate for the optimizer.
            batch_size (int): The batch size for training.
            epochs (int): The number of training epochs.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def encodeBB(self, text_tensor, label_tensor):
        """
        Encode text and label tensors into BERT-compatible input.

        Args:
            text_tensor (tf.Tensor): The input text tensor.
            label_tensor (tf.Tensor): The label tensor.

        Returns:
            tuple: Encoded input IDs, attention masks, and labels.
        """
        text = text_tensor.numpy().decode("utf-8")
        encoded_text = self.encodeB(text)
        return (
            encoded_text["input_ids"][0],
            encoded_text["attention_mask"][0],
            label_tensor,
        )

    def encode_map_fnB(self, text, label):
        """
        Map function to encode text and label into BERT-compatible input.

        Args:
            text (tf.Tensor): The input text tensor.
            label (tf.Tensor): The label tensor.

        Returns:
            tuple: Encoded input IDs, attention masks, and labels.
        """
        return tf.py_function(
            self.encodeBB, inp=[text, label], Tout=(tf.int32, tf.int32, tf.int64)
        )

    def prepare_data(self, dataset, batch_size):
        """
        Prepare the dataset for training by encoding and batching.

        Args:
            dataset (tf.data.Dataset): The input dataset.
            batch_size (int): The batch size for training.

        Returns:
            tf.data.Dataset: The prepared dataset.
        """
        dataset = dataset.map(lambda text, label: self.encode_map_fnB(text, label))
        dataset = dataset.map(
            lambda input_ids, attention_mask, label: (
                {"input_ids": input_ids, "attention_mask": attention_mask},
                label,
            )
        )
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=({"input_ids": [None], "attention_mask": [None]}, []),
        )
        return dataset

    def encodeB(self, texts):
        """
        Tokenize the input text using the BERT tokenizer.

        Args:
            texts (str or list): The input text(s) to tokenize.

        Returns:
            dict: Tokenized input IDs and attention masks.
        """
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="tf",
        )

    def build_model(self, num_classes):
        """
        Build and compile the BERT-based sentiment analysis model.

        Args:
            num_classes (int): The number of output classes.

        Returns:
            tf.keras.Model: The compiled BERT model.
        """
        bert_model = TFBertModel.from_pretrained(
            self.model_name, output_hidden_states=False
        )
        input_ids = tf.keras.layers.Input(
            shape=(self.max_length,), dtype=tf.int32, name="input_ids"
        )
        attention_mask = tf.keras.layers.Input(
            shape=(self.max_length,), dtype=tf.int32, name="attention_mask"
        )

        bert_output = bert_model(input_ids, attention_mask=attention_mask)
        cls_token = bert_output.last_hidden_state[:, 0, :]
        dropout = tf.keras.layers.Dropout(0.3)(cls_token)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(dropout)

        model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.legacy.RMSprop(
                learning_rate=self.learning_rate
            ),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )
        return model

    def train_and_evaluate(self, model, train_data, valid_data, test_data):
        """
        Train and evaluate the BERT model.

        Args:
            model (tf.keras.Model): The BERT model to train.
            train_data (tf.data.Dataset): The training dataset.
            valid_data (tf.data.Dataset): The validation dataset.
            test_data (tf.data.Dataset): The test dataset.
        """
        model.summary()
        with tf.device("/device:GPU:0"):
            model.fit(train_data, validation_data=valid_data, epochs=self.epochs)
        test_results = model.evaluate(test_data)
        print("Test Acc.: {:.2f}%".format(test_results[1] * 100))


class SentimentModel:
    """
    A class to define, train, and evaluate a sentiment analysis model using LSTM layers.

    Attributes:
        embedding_dim (int): Dimension of the embedding layer.
        lstm_units (int): Number of units in the LSTM layers.
        dropout_rate (float): Dropout rate for regularization.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
    """

    def __init__(
        self,
        embedding_dim=50,
        lstm_units=128,
        dropout_rate=0.5,
        learning_rate=0.0008659430202504234,
        epochs=10,
    ):
        """
        Initialize the SentimentModel class with hyperparameters.

        Args:
            embedding_dim (int): Dimension of the embedding layer.
            lstm_units (int): Number of units in the LSTM layers.
            dropout_rate (float): Dropout rate for regularization.
            learning_rate (float): Learning rate for the optimizer.
            epochs (int): Number of training epochs.
        """
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs

    def build_model(self, vocab_size, num_classes):
        """
        Build and compile the LSTM-based sentiment analysis model.

        Args:
            vocab_size (int): Size of the vocabulary.
            num_classes (int): Number of output classes.

        Returns:
            tf.keras.Model: The compiled LSTM model.
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(None,), dtype="int32"))
        model.add(
            tf.keras.layers.Embedding(
                input_dim=vocab_size, output_dim=self.embedding_dim, name="embed-layer"
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

    def train_and_evaluate(self, model, train_data, valid_data, test_data):
        """
        Train and evaluate the LSTM model.

        Args:
            model (tf.keras.Model): The LSTM model to train.
            train_data (tf.data.Dataset): The training dataset.
            valid_data (tf.data.Dataset): The validation dataset.
            test_data (tf.data.Dataset): The test dataset.
        """
        model.summary()
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            mode="min",
            verbose=0,
            restore_best_weights=True,
        )
        with tf.device("/device:GPU:0"):
            model.fit(
                train_data,
                validation_data=valid_data,
                epochs=self.epochs,
                callbacks=[early_stopping_callback],
            )
            model.save("./models/sentiment_binary.keras")
            test_results = model.evaluate(test_data)
        print("Test Acc.: {:.2f}%".format(test_results[1] * 100))

    def evaluate(self, test_data):
        """
        Evaluate the saved LSTM model on the test dataset.

        Args:
            test_data (tf.data.Dataset): The test dataset.
        """
        model = tf.keras.models.load_model("./models/sentiment_binary.keras")
        test_results = model.evaluate(test_data)
        print("Test Acc.: {:.2f}%".format(test_results[1] * 100))

    def evaluate_text(self, test_data):
        """
        Evaluate the saved LSTM model and return the accuracy.

        Args:
            test_data (tf.data.Dataset): The test dataset.

        Returns:
            float: The accuracy of the model on the test dataset.
        """
        model = tf.keras.models.load_model("./models/sentiment_binary.keras")
        test_results = model.evaluate(test_data)
        return test_results[1]

    def predict_text(self, predict_data):
        """
        Predict sentiment for the given data using the saved LSTM model.

        Args:
            predict_data (tf.data.Dataset): The dataset for prediction.

        Returns:
            tuple: Predicted classes and probabilities.
        """
        model = tf.keras.models.load_model("./models/sentiment_binary.keras")
        predictions = model.predict(predict_data)
        y_classes = predictions.argmax(axis=-1)
        return y_classes, predictions

    def Optuna(self, vocab_size, num_classes, train_data, valid_data, test_data):
        """
        Perform hyperparameter optimization using Optuna.

        Args:
            vocab_size (int): Size of the vocabulary.
            num_classes (int): Number of output classes.
            train_data (tf.data.Dataset): The training dataset.
            valid_data (tf.data.Dataset): The validation dataset.
            test_data (tf.data.Dataset): The test dataset.
        """

        def _objective(trial):
            """
            Objective function for Optuna to optimize the model's hyperparameters.

            Args:
                trial (optuna.trial.Trial): An Optuna trial object.

            Returns:
                float: Validation accuracy of the model.
            """
            tf.keras.backend.clear_session()
            model = tf.keras.Sequential()
            model.add(tf.keras.Input(shape=(None,), dtype="int32"))
            model.add(
                tf.keras.layers.Embedding(
                    input_dim=vocab_size,
                    output_dim=self.embedding_dim,
                )
            )
            n_layers_bidirectional = trial.suggest_int("n_units_bidirectional", 1, 3)
            for i in range(n_layers_bidirectional):
                num_hidden_bidirectional = trial.suggest_int(
                    "n_units_bidirectional_l{}".format(i), 64, 128, log=True
                )
                model.add(
                    tf.keras.layers.Bidirectional(
                        tf.keras.layers.LSTM(
                            num_hidden_bidirectional,
                            return_sequences=True,
                        ),
                    )
                )
            num_hidden_lstm = trial.suggest_int(
                "n_units_lstm_l{}".format(i), 64, 128, log=True
            )
            model.add(
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        num_hidden_lstm,
                        return_sequences=False,
                    ),
                )
            )

            model.add(tf.keras.layers.Dropout(self.dropout_rate))
            n_layers_nn = trial.suggest_int("n_layers_nn", 1, 2)
            for i in range(n_layers_nn):
                num_hidden_nn = trial.suggest_int(
                    "n_units_nn_l{}".format(i), 64, 128, log=True
                )
                model.add(tf.keras.layers.Dense(num_hidden_nn, activation="gelu"))

            model.add(tf.keras.layers.Dropout(self.dropout_rate))
            model.add(tf.keras.layers.Dense(32, activation="gelu"))
            model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            model.compile(
                optimizer=tf.keras.optimizers.legacy.RMSprop(
                    learning_rate=learning_rate
                ),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=["accuracy"],
            )

            early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                mode="min",
                verbose=0,
                restore_best_weights=True,
            )
            with tf.device("/device:GPU:0"):
                model.fit(
                    train_data,
                    validation_data=valid_data,
                    epochs=int(self.epochs / 2),
                    callbacks=[early_stopping_callback],
                    verbose=1,
                )
            # Evaluate the model accuracy on the validation set.
            score = model.evaluate(test_data, verbose=1)
            return score[1]

        study = optuna.create_study(direction="maximize")
        study.optimize(
            _objective,
            n_trials=5,
        )
        with open("./models/optuna_model_binary.json", "w") as outfile:
            json.dump(study.best_params, outfile)
