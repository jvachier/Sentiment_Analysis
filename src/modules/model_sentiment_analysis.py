import tensorflow as tf
import optuna
import json
import logging
from src.modules.utils import ModelPaths, OptunaPaths


class ModelTrainer:
    """
    A class to train, evaluate, and optimize sentiment analysis models.
    """

    def __init__(self, config_path: str):
        """
        Initialize the ModelTrainer class with hyperparameters.

        Args:
            learning_rate (float): Learning rate for the optimizer.
            epochs (int): Number of training epochs.
        """
        with open(config_path, "r") as config_file:
            self.config = json.load(config_file)

        self.learning_rate = self.config["learning_rate"]
        self.epochs = self.config["epochs"]

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
            filepath=ModelPaths.TRAINED_MODEL.value,
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

    def __init__(self, config_path: str):
        """
        Initialize the OptunaOptimizer class with model parameters and configuration.

        Args:
            config_path (str): Path to the JSON configuration file.
        """

        # Load Optuna configuration from JSON file
        with open(config_path, "r") as config_file:
            self.config = json.load(config_file)

        # Extract embedding_dim and dropout_rate from the configuration
        self.embedding_dim = self.config["embedding_dim"]
        self.dropout_rate = self.config["dropout_rate"]
        self.max_token = self.config["max_token"]

    def optimize(
        self,
        train_data: tf.data.Dataset,
        valid_data: tf.data.Dataset,
        test_data: tf.data.Dataset,
    ) -> None:
        """
        Perform hyperparameter optimization using Optuna.

        Args:
            train_data (tf.data.Dataset): Training dataset.
            valid_data (tf.data.Dataset): Validation dataset.
            test_data (tf.data.Dataset): Test dataset.
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

        # Create an Optuna study
        study = optuna.create_study(
            direction=self.config["direction"],
            study_name=self.config["study_name"],
            storage=self.config["storage"],
            load_if_exists=True,
        )
        study.optimize(
            _objective,
            n_trials=self.config["n_trials"],
        )

        # Save the best parameters to a JSON file
        with open(OptunaPaths.OPTUNA_MODEL.value, "w") as outfile:
            json.dump(study.best_params, outfile)


class ModelBuilder:
    """
    A class to define and build sentiment analysis models using Keras.
    """

    def __init__(self, config_path: str):
        """
        Initialize the ModelBuilder class with hyperparameters.

        Args:
            embedding_dim (int): Dimension of the embedding layer.
            lstm_units (int): Number of units in the LSTM layers.
            dropout_rate (float): Dropout rate for regularization.
            max_token (int): Maximum number of tokens for the embedding layer.
        """
        super().__init__()
        with open(config_path, "r") as config_file:
            self.config = json.load(config_file)

        self.embedding_dim = self.config["embedding_dim"]
        self.dropout_rate = self.config["dropout_rate"]
        self.lstm_units = self.config["lstm_units"]
        self.max_token = self.config["max_token"]

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
