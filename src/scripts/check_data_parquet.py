import polars as pl
import tensorflow as tf
import random
import string
import re
from tensorflow.keras import layers

# Set a seed for reproducibility
random.seed(42)

# Read the Parquet file using Polars
df = pl.read_parquet("src/data/en-fr.parquet")

# Define the delimiters for splitting
delimiters = r"|"

# Split the 'en' column into rows based on delimiters
if "en" in df.columns:
    en_split = df.select(pl.col("en").str.split(delimiters)).explode("en")

# Split the 'fr' column into rows based on delimiters
if "fr" in df.columns:
    fr_split = df.select(pl.col("fr").str.split(delimiters)).explode("fr")

# Combine the split results into a new DataFrame
split_df = pl.concat([en_split, fr_split], how="horizontal")

# Remove rows with null values
split_df = split_df.drop_nulls()  # Drops rows with null values in any column

# Add start and end tokens to French sentences directly in Polars
split_df = split_df.with_columns(
    pl.col("fr").apply(lambda fr: f"[start] {fr} [end]").alias("fr")
)

print("split done")

# Shuffle the DataFrame
split_df = split_df.sample(frac=1, seed=42)  # Shuffle with a seed for reproducibility

# Split the dataset into training, validation, and test sets
num_val_samples = int(0.15 * len(split_df))
num_train_samples = len(split_df) - 2 * num_val_samples

train_df = split_df[:num_train_samples]
val_df = split_df[num_train_samples : num_train_samples + num_val_samples]
test_df = split_df[num_train_samples + num_val_samples :]

# Define custom standardization function
strip_chars = string.punctuation
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")


def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")


# Define TextVectorization layers
vocab_size = 15000
sequence_length = 20

source_vectorization = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)
target_vectorization = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)

# Adapt the TextVectorization layers to the training data
source_vectorization.adapt(train_df["en"].to_list())
target_vectorization.adapt(train_df["fr"].to_list())

print("text vecto and adapt done")

# Define batch size
batch_size = 64


# Format dataset for training
def format_dataset(eng, fr):
    eng = source_vectorization(eng)
    fr = target_vectorization(fr)
    return (
        {
            "english": eng,
            "french": fr[:, :-1],  # Input sequence for the decoder
        },
        fr[:, 1:],  # Target sequence for the decoder
    )


# Create a function to make datasets
def make_dataset(df):
    dataset = tf.data.Dataset.from_tensor_slices(
        (df["en"].to_list(), df["fr"].to_list())
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.shuffle(2048).prefetch(tf.data.AUTOTUNE)


print("Start to build the dataset")
# Create training, validation, and test datasets
train_ds = make_dataset(train_df)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)

# Display a batch of vectorized data
for inputs, targets in train_ds.take(1):
    print(f"inputs['english'].shape: {inputs['english'].shape}")
    print(f"inputs['french'].shape: {inputs['french'].shape}")
    print(f"targets.shape: {targets.shape}")
