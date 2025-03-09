import pandas as pd
import tensorflow as tf


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        df = pd.read_csv(self.data_path, encoding="utf-8")
        target = df.pop("Rating")
        ds_raw = tf.data.Dataset.from_tensor_slices(
            (df["Review"].values, target.values)
        )
        ds_raw = ds_raw.shuffle(len(df), reshuffle_each_iteration=False)

        ds_raw_test = ds_raw.take(int(len(df) * 0.2))
        ds_raw_train_valid = ds_raw.skip(int(len(df) * 0.2))
        ds_raw_train = ds_raw_train_valid.take(int(len(ds_raw_train_valid) * 0.8))
        ds_raw_valid = ds_raw_train_valid.skip(int(len(ds_raw_train_valid) * 0.8))

        return ds_raw, ds_raw_train, ds_raw_valid, ds_raw_test, target
