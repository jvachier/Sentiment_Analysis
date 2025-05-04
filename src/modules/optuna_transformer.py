import optuna
import tensorflow as tf
from modules.data_processor import DatasetProcessor, TextPreprocessor
from modules.transformer_components import (
    PositionalEmbedding,
    TransformerEncoder,
    TransformerDecoder,
    evaluate_bleu,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def build_transformer_model(trial, preprocessor):
    """
    Build a Transformer model with hyperparameters suggested by Optuna.

    Args:
        trial (optuna.trial.Trial): The trial object for hyperparameter optimization.
        preprocessor (TextPreprocessor): Preprocessor object containing sequence length and vocabulary size.

    Returns:
        tf.keras.Model: The compiled Transformer model.
    """
    # Hyperparameters to optimize
    embed_dim = trial.suggest_categorical("embed_dim", [64, 128])
    dense_dim = trial.suggest_int("dense_dim", 512, 2048, step=512)
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)

    sequence_length = preprocessor.sequence_length
    vocab_size = preprocessor.vocab_size

    # Build the Transformer model
    encoder_inputs = tf.keras.Input(shape=(None,), dtype="int32", name="english")
    encoder_embeddings = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(
        encoder_inputs
    )
    encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(
        encoder_embeddings
    )

    decoder_inputs = tf.keras.Input(shape=(None,), dtype="int32", name="french")
    decoder_embeddings = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(
        decoder_inputs
    )
    decoder_outputs = TransformerDecoder(embed_dim, dense_dim, num_heads)(
        decoder_embeddings, encoder_outputs
    )
    dropout_outputs = tf.keras.layers.Dropout(dropout_rate)(decoder_outputs)
    final_outputs = tf.keras.layers.Dense(vocab_size, activation="softmax")(
        dropout_outputs
    )

    transformer = tf.keras.Model([encoder_inputs, decoder_inputs], final_outputs)

    # Compile the model
    transformer.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return transformer


def objective(trial):
    """
    Objective function for Optuna to optimize the Transformer model using BLEU score.

    Args:
        trial (optuna.trial.Trial): The trial object for hyperparameter optimization.

    Returns:
        float: BLEU score of the model on the validation dataset.
    """
    # Load and preprocess the dataset
    processor = DatasetProcessor(file_path="src/data/en-fr.parquet")
    processor.load_data()
    processor.process_data()
    data_splits = processor.shuffle_and_split()
    train_df, val_df = data_splits["train"], data_splits["validation"]

    preprocessor = TextPreprocessor()
    preprocessor.adapt(train_df)

    train_ds = preprocessor.make_dataset(train_df)
    val_ds = preprocessor.make_dataset(val_df)

    # Build the model
    model = build_transformer_model(trial, preprocessor)

    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=2,
            mode="min",
            verbose=1,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            mode="min",
            verbose=1,
        ),
    ]

    # Train the model
    device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
    with tf.device(device):
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=3,  # Use fewer epochs for faster optimization
            verbose=1,
            callbacks=callbacks,
        )

    # Calculate BLEU score on the validation dataset
    bleu_score = evaluate_bleu(model, val_ds, preprocessor)
    return bleu_score


def main():
    """
    Main function to run the Optuna optimization.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)

    logging.info("Best trial:")
    logging.info(f"Value (BLEU Score): {study.best_trial.value}")
    logging.info("Params:")
    for key, value in study.best_trial.params.items():
        logging.info(f"    {key}: {value}")

    # Save the best hyperparameters
    best_params = study.best_trial.params
    with open("src/models/optuna_transformer_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)


if __name__ == "__main__":
    main()
