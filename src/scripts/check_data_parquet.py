import polars as pl

# Read the Parquet file using Polars
df = pl.read_parquet("src/data/en-fr.parquet")

# Display DataFrame information
print(
    df.describe()
)  # Polars does not have `info()`, but `describe()` provides summary statistics

# Display the first few rows
print(df.head())
