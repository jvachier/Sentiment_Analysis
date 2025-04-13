import numpy as np
import pandas as pd
import logging


class MemoryReducer:
    """
    A class to reduce memory usage of a pandas DataFrame by optimizing data types.
    """

    def __init__(self, verbose=False):
        """
        Initialize the MemoryReducer class.

        Args:
            verbose (bool): Whether to log memory reduction details.
        """
        self.verbose = verbose

    def reduce(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce memory usage of a pandas DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to optimize.

        Returns:
            pd.DataFrame: The optimized DataFrame.
        """
        numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
        start_mem = df.memory_usage().sum() / 1024**2

        if self.verbose:
            logging.info(f"Initial memory usage: {start_mem:.2f} MB")

        for col in df.columns:
            col_type = df[col].dtypes

            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif (
                        c_min > np.iinfo(np.int16).min
                        and c_max < np.iinfo(np.int16).max
                    ):
                        df[col] = df[col].astype(np.int16)
                    elif (
                        c_min > np.iinfo(np.int32).min
                        and c_max < np.iinfo(np.int32).max
                    ):
                        df[col] = df[col].astype(np.int32)
                    elif (
                        c_min > np.iinfo(np.int64).min
                        and c_max < np.iinfo(np.int64).max
                    ):
                        df[col] = df[col].astype(np.int64)
                else:
                    if (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                    ):
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            elif col_type == "object":
                # Convert object columns to category if unique values are less than 50% of total rows
                num_unique_values = df[col].nunique()
                num_total_values = len(df[col])
                if num_unique_values / num_total_values < 0.5:
                    df[col] = df[col].astype("category")
                else:
                    df[col] = df[col].astype("string")

        end_mem = df.memory_usage().sum() / 1024**2
        if self.verbose:
            logging.info(
                f"Memory usage decreased to {end_mem:.2f} MB "
                f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
            )
        return df
