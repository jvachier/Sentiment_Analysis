import tensorflow as tf
import optuna
import json
import logging

logging.basicConfig(level=logging.INFO)


class ModelTrainer:
    """
    A class to train, evaluate, and optimize sentiment analysis models.
    """

    def __init__(self, learning_rate=0.0008659430202504234, epochs=5):
        """
        Initialize the ModelTrainer class with hyperparameters.

        Args:
            learning_rate (float): Learning rate for the optimizer.
            epochs (int): Number of training epochs.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train_and_evaluate(
        self,
        model: tf.keras.Model,
        train_data: tf.data.Dataset,
        valid_data: tf.data.Dataset,
        test_data: tf.data.Dataset,
    ) -> None:
        """
        Train and evaluate the sentiment analysis model.

        Args:
            model (tf.keras.Model): The Keras model to train.
            train_data (tf.data.Dataset): Training dataset.
            valid_data (tf.data.Dataset): Validation dataset.
            test_data (tf.data.Dataset): Test dataset.
        """
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
        with tf.device(
            "/device:GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
        ):
            model.fit(
                train_data,
                validation_data=valid_data,
                epochs=self.epochs,
                callbacks=[early_stopping_callback, callbacks_model],
            )
            test_results = model.evaluate(test_data)
        logging.info("Test Accuracy: {:.2f}%".format(test_results[1] * 100))

    def inference_model(
        self, model: tf.keras.Model, text_vec: tf.keras.layers.TextVectorization
    ) -> tf.keras.Model:
        """
        Create an inference model for predicting sentiment.

        Args:
            model (tf.keras.Model): The trained Keras model.
            text_vec (tf.keras.layers.TextVectorization): Text vectorization layer.

        Returns:
            tf.keras.Model: An inference model for sentiment prediction.
        """
        inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
        process_inputs = text_vec(inputs)
        outputs = model(process_inputs)
        inference_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return inference_model


class OptunaOptimizer:
    """
    A class to perform hyperparameter optimization using Optuna.
    """

    def __init__(self, max_token=20000, embedding_dim=50, dropout_rate=0.5):
        """
        Initialize the OptunaOptimizer class with model parameters.

        Args:
            max_token (int): Maximum number of tokens for the embedding layer.
            embedding_dim (int): Dimension of the embedding layer.
            dropout_rate (float): Dropout rate for regularization.
        """
        self.max_token = max_token
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate

    def Optuna(
        self,
        train_data: tf.data.Dataset,
        valid_data: tf.data.Dataset,
        test_data: tf.data.Dataset,
        n_trials: int = 5,
    ) -> None:
        """
        Perform hyperparameter optimization using Optuna.

        Args:
            train_data (tf.data.Dataset): Training dataset.
            valid_data (tf.data.Dataset): Validation dataset.
            test_data (tf.data.Dataset): Test dataset.
            n_trials (int): Number of trials for Optuna optimization.
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
                    input_dim=self.max_token,
                    output_dim=self.embedding_dim,
                )
            )
            n_layers_bidirectional = trial.suggest_int("n_units_bidirectional", 1, 3)
            for i in range(n_layers_bidirectional):
                num_hidden_bidirectional = trial.suggest_int(
                    f"n_units_bidirectional_l{i}", 64, 128, log=True
                )
                model.add(
                    tf.keras.layers.Bidirectional(
                        tf.keras.layers.LSTM(
                            num_hidden_bidirectional,
                            return_sequences=True,
                        ),
                    )
                )
            num_hidden_lstm = trial.suggest_int(f"n_units_lstm_l{i}", 64, 128, log=True)
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
                num_hidden_nn = trial.suggest_int(f"n_units_nn_l{i}", 64, 128, log=True)
                model.add(tf.keras.layers.Dense(num_hidden_nn, activation="gelu"))

            model.add(tf.keras.layers.Dropout(self.dropout_rate))
            model.add(tf.keras.layers.Dense(32, activation="gelu"))
            model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            model.compile(
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
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
            with tf.device(
                "/device:GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
            ):
                model.fit(
                    train_data,
                    validation_data=valid_data,
                    epochs=5,
                    callbacks=[early_stopping_callback],
                    verbose=1,
                )
            # Evaluate the model accuracy on the validation set.
            score = model.evaluate(test_data, verbose=1)
            return score[1]

        study = optuna.create_study(direction="maximize")
        study.optimize(
            _objective,
            n_trials=n_trials,
        )
        with open("./models/optuna_model_binary.json", "w") as outfile:
            json.dump(study.best_params, outfile)


class ModelBuilder:
    """
    A class to define and build sentiment analysis models using Keras.
    """

    def __init__(
        self,
        embedding_dim=50,
        lstm_units=128,
        dropout_rate=0.5,
        max_token=20000,
    ):
        """
        Initialize the ModelBuilder class with hyperparameters.

        Args:
            embedding_dim (int): Dimension of the embedding layer.
            lstm_units (int): Number of units in the LSTM layers.
            dropout_rate (float): Dropout rate for regularization.
            max_token (int): Maximum number of tokens for the embedding layer.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.max_token = max_token

    def get_model_api(self) -> tf.keras.Model:
        """
        Build and compile the sentiment analysis model using the Functional API.

        Returns:
            tf.keras.Model: A compiled Keras model.
        """
        inputs = tf.keras.Input(shape=(None,), dtype="int32")
        embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.max_token,
            output_dim=self.embedding_dim,
            name="embed-layer",
        )(inputs)
        lstm_layer1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(80, return_sequences=True, name="lstm-layer")
        )(embedding_layer)
        lstm_layer2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(121, return_sequences=False, name="lstm-layer")
        )(lstm_layer1)
        dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)(lstm_layer2)
        dense_layer1 = tf.keras.layers.Dense(67, activation="gelu")(dropout_layer)
        dense_layer2 = tf.keras.layers.Dense(75, activation="gelu")(dense_layer1)
        dropout_layer2 = tf.keras.layers.Dropout(self.dropout_rate)(dense_layer2)
        dense_layer3 = tf.keras.layers.Dense(32, activation="gelu")(dropout_layer2)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(dense_layer3)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )
        return model

    def get_model_sequential(self) -> tf.keras.Model:
        """
        Build and compile the sentiment analysis model using the Sequential API.

        Returns:
            tf.keras.Model: A compiled Keras model.
        """
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
            optimizer=tf.keras.optimizers.RMSprop(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )
        return model

    def get_config(self) -> dict:
        """
        Retrieve the configuration of the model.

        Returns:
            dict: A dictionary containing the model's configuration.
        """
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "lstm_units": self.lstm_units,
                "dropout_rate": self.dropout_rate,
                "max_token": self.max_token,
            }
        )
        return config
