from data_processor import DatasetProcessor
from data_processor import TextPreprocessor

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

# Display a batch of vectorized data
for inputs, targets in train_ds.take(1):
    print(f"inputs['english'].shape: {inputs['english'].shape}")
    print(f"inputs['french'].shape: {inputs['french'].shape}")
    print(f"targets.shape: {targets.shape}")
