import pandas as pd
import tensorflow as tf


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        df = pd.read_csv(self.data_path, encoding="utf-8")
        df.loc[df.Rating < 3, "Label"] = 0
        df.loc[df.Rating == 3, "Label"] = 1
        df.loc[df.Rating > 3, "Label"] = 2
        df["Label"] = df["Label"].astype(int)
        df.drop(columns=["Rating"], inplace=True)

        target = df.pop("Label")
        ds_raw = tf.data.Dataset.from_tensor_slices(
            (df["Review"].values, target.values)
        )
        ds_raw = ds_raw.shuffle(len(df), reshuffle_each_iteration=False, seed=42)

        ds_raw_test = ds_raw.take(int(len(df) * 0.3))
        ds_raw_train_valid = ds_raw.skip(int(len(df) * 0.3))
        ds_raw_train = ds_raw_train_valid.take(int(len(ds_raw_train_valid) * 0.7))
        ds_raw_valid = ds_raw_train_valid.skip(int(len(ds_raw_train_valid) * 0.7))

        return ds_raw, ds_raw_train, ds_raw_valid, ds_raw_test, target
