import polars as pl

# Read the Parquet file using Polars
df = pl.read_parquet("src/data/en-fr.parquet")

# Display DataFrame information
print(
    df.describe()
)  # Polars does not have `info()`, but `describe()` provides summary statistics


# Define the delimiters for splitting
delimiters = r"|"

# Split the 'en' column into rows based on delimiters
if "en" in df.columns:
    en_split = df.select(pl.col("en").str.split(delimiters)).explode("en")

# Split the 'fr' column into rows based on delimiters
if "fr" in df.columns:
    fr_split = df.select(pl.col("fr").str.split(delimiters)).explode("fr")
split_df = pl.concat([en_split, fr_split], how="horizontal")
# Display the first few rows
print(split_df.head(20))
