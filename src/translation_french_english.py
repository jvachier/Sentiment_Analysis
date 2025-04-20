from modules.data_processor import DatasetProcessor, TextPreprocessor
from modules.transformer_components import (
    PositionalEmbedding,
    TransformerEncoder,
    TransformerDecoder,
)
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Initialize the DatasetProcessor
processor = DatasetProcessor(file_path="src/data/en-fr.parquet")
processor.load_data()
processor.process_data()
data_splits = processor.shuffle_and_split()
train_df, val_df, test_df = (
    data_splits["train"],
    data_splits["validation"],
    data_splits["test"],
)

# Initialize the TextPreprocessor
preprocessor = TextPreprocessor()
preprocessor.adapt(train_df)

# Create TensorFlow datasets
train_ds = preprocessor.make_dataset(train_df)
val_ds = preprocessor.make_dataset(val_df)
test_ds = preprocessor.make_dataset(test_df)


# Define Transformer model
embed_dim = 64  # 256
dense_dim = 2048
num_heads = 3  # 8
sequence_length = preprocessor.sequence_length
vocab_size = preprocessor.vocab_size

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
    # Save the best model during training
    ModelCheckpoint(
        filepath="src/models/transformer_best_model.keras",
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=1,
    ),
    # Stop training early if validation loss doesn't improve
    EarlyStopping(
        monitor="val_loss",
        patience=2,
        mode="min",
        verbose=1,
        restore_best_weights=True,
    ),
    # Reduce learning rate when validation loss plateaus
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        mode="min",
        verbose=1,
    ),
]

# Train the model with callbacks
with tf.device("/GPU:0"):
    transformer.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,  # Increased epochs for better training
        verbose=1,
        callbacks=callbacks,
    )

# Evaluate the model
results = transformer.evaluate(test_ds)
print("Test loss, Test accuracy:", results)


# BLEU Metric Evaluation
def evaluate_bleu(model, dataset):
    """
    Evaluate the BLEU score for the model on the given dataset.

    Args:
        model (tf.keras.Model): The trained Transformer model.
        dataset (tf.data.Dataset): The dataset to evaluate.

    Returns:
        float: The BLEU score for the dataset.
    """
    references = []
    candidates = []
    smoothing_function = SmoothingFunction().method1

    # Get the vocabulary from the target vectorization layer
    vocab = preprocessor.target_vectorization.get_vocabulary()
    index_to_word = {i: word for i, word in enumerate(vocab)}

    for batch in dataset:
        inputs, targets = batch
        # Generate predictions
        predictions = model.predict(inputs, verbose=0)

        # Decode predictions and targets
        for i in range(len(predictions)):
            # Decode predicted sentence
            pred_tokens = predictions[i].argmax(axis=-1)  # Get token IDs
            pred_sentence = " ".join(
                [
                    index_to_word[token] for token in pred_tokens if token != 0
                ]  # Ignore padding tokens
            )

            # Decode reference sentence
            ref_tokens = targets[i].numpy()  # Get token IDs
            ref_sentence = " ".join(
                [
                    index_to_word[token] for token in ref_tokens if token != 0
                ]  # Ignore padding tokens
            )

            candidates.append(pred_sentence)
            references.append([ref_sentence])

    # Calculate BLEU score
    bleu_score = corpus_bleu(
        references, candidates, smoothing_function=smoothing_function
    )
    return bleu_score


# Calculate BLEU score on the test dataset
bleu_score = evaluate_bleu(transformer, test_ds)
print(f"BLEU score on the test dataset: {bleu_score:.4f}")
