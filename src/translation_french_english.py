from modules.data_processor import DatasetProcessor, TextPreprocessor
from modules.transformer_components import (
    PositionalEmbedding,
    TransformerEncoder,
    TransformerDecoder,
)
import tensorflow as tf

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
dense_dim = 512  # 2048
num_heads = 3  # 8
sequence_length = preprocessor.sequence_length
vocab_size = 5000  # preprocessor.vocab_size

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

# Train the model
with tf.device("/GPU:0"):
    transformer.fit(train_ds, validation_data=val_ds, epochs=2)
