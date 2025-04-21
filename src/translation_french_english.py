import logging
from modules.data_processor import DatasetProcessor, TextPreprocessor
from modules.transformer_components import (
    evaluate_bleu,
    PositionalEmbedding,
    TransformerEncoder,
    TransformerDecoder,
)
from modules.utils import ModelPaths
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def transformer_model(
    transformer_model_path,
    preprocessor: TextPreprocessor,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
) -> tf.keras.Model:
    """
    Build, compile, and train a Transformer model for English-to-French translation.

    This function either loads a pre-trained Transformer model from a saved file
    or builds a new model if no saved model is found. The model is trained on the
    provided training and validation datasets.

    Args:
        preprocessor (TextPreprocessor): Preprocessor object containing sequence length
            and vocabulary size information.
        train_ds (tf.data.Dataset): Training dataset.
        val_ds (tf.data.Dataset): Validation dataset.

    Returns:
        tf.keras.Model: The trained Transformer model.
    """
    if os.path.isfile(transformer_model_path):
        # Load the saved model
        logging.info("Loading the saved Transformer model.")
        return tf.keras.models.load_model(
            "src/models/transformer_best_model.keras",
            custom_objects={
                "PositionalEmbedding": PositionalEmbedding,
                "TransformerEncoder": TransformerEncoder,
                "TransformerDecoder": TransformerDecoder,
            },
        )
    # Define model parameters
    embed_dim = 128
    dense_dim = 2048
    num_heads = 8
    sequence_length = preprocessor.sequence_length
    vocab_size = preprocessor.vocab_size

    # Build the Transformer model
    encoder_inputs = tf.keras.Input(shape=(None,), dtype="int32", name="english")
    x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
    encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)

    decoder_inputs = tf.keras.Input(shape=(None,), dtype="int32", name="french")
    x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
    x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    decoder_outputs = tf.keras.layers.Dense(vocab_size, activation="softmax")(x)

    transformer = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Compile the model
    transformer.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # Display the model summary
    transformer.summary()

    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            filepath="src/models/transformer_best_model.keras",
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1,
        ),
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
    logging.info("Starting model training.")
    with tf.device("/GPU:0"):
        transformer.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10,
            verbose=1,
            callbacks=callbacks,
        )
    return transformer


def test_translation(transformer, preprocessor, input_sentence="Hello") -> None:
    """
    Test the Transformer model by translating an input sentence.

    Args:
        transformer (tf.keras.Model): The trained Transformer model.
        preprocessor (TextPreprocessor): Preprocessor object for tokenization and vectorization.
        input_sentence (str): The input sentence to translate.

    Returns:
        None
    """
    en_vocab = preprocessor.target_vectorization.get_vocabulary()
    en_index_lookup = dict(zip(range(len(en_vocab)), en_vocab))

    tokenized_input_sentence = preprocessor.source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(20):
        tokenized_target_sentence = preprocessor.target_vectorization(
            [decoded_sentence]
        )[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = en_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break

    logging.info(f"Input Sentence: {input_sentence}")
    logging.info(f"Translated Sentence: {decoded_sentence}")


def main() -> None:
    logging.info("Initializing dataset processor.")
    processor = DatasetProcessor(file_path="src/data/en-fr.parquet")
    processor.load_data()
    processor.process_data()
    data_splits = processor.shuffle_and_split()
    train_df, val_df, test_df = (
        data_splits["train"],
        data_splits["validation"],
        data_splits["test"],
    )

    logging.info("Initializing text preprocessor.")
    preprocessor = TextPreprocessor()
    preprocessor.adapt(train_df)

    # Create TensorFlow datasets
    train_ds = preprocessor.make_dataset(train_df)
    val_ds = preprocessor.make_dataset(val_df)
    test_ds = preprocessor.make_dataset(test_df)

    transformer_model_path = ModelPaths.TRANSFORMER_MODEL.value

    transformer = transformer_model(
        transformer_model_path, preprocessor, train_ds, val_ds
    )

    # Test the translation
    test_translation(
        transformer,
        preprocessor,
        input_sentence="How are you?",
    )

    # Evaluate the model
    logging.info("Evaluating the model on the test dataset.")
    results = transformer.evaluate(test_ds)
    logging.info(f"Test loss: {results[0]}, Test accuracy: {results[1]}")

    # Calculate BLEU score
    bleu_score = evaluate_bleu(transformer, test_ds, preprocessor)
    logging.info(f"BLEU score on the test dataset: {bleu_score:.4f}")


if __name__ == "__main__":
    main()
