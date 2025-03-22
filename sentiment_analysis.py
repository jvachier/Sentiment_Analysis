from modules.load_data import DataLoader
from modules.data_preprocess import TextPreprocessor
from modules.model import SentimentModel, SentimentModelBert
import os
import tensorflow as tf
import pandas as pd
import numpy as np

BERT = False
OPTUNA = False


def main():
    data_loader = DataLoader(data_path="./data/tripadvisor_hotel_reviews.csv")
    ds_raw, ds_raw_train, ds_raw_valid, ds_raw_test, target = data_loader.load_data()

    # Sentiment with BERT
    if BERT:
        sentiment_model = SentimentModelBert()
        train_data = sentiment_model.prepare_data(
            ds_raw_train, sentiment_model.batch_size
        )
        valid_data = sentiment_model.prepare_data(
            ds_raw_valid, sentiment_model.batch_size
        )
        test_data = sentiment_model.prepare_data(
            ds_raw_test, sentiment_model.batch_size
        )

        num_classes = target.nunique()

        model = sentiment_model.build_model(num_classes)
        sentiment_model.train_and_evaluate(model, train_data, valid_data, test_data)
    # Sentiment without BERT
    else:
        text_preprocessor = TextPreprocessor()
        token_counts = text_preprocessor.fit_tokenizer(ds_raw)
        ds_train = ds_raw_train.map(
            lambda text, label: text_preprocessor.encode_map_fn(text, label)
        )
        ds_valid = ds_raw_valid.map(
            lambda text, label: text_preprocessor.encode_map_fn(text, label)
        )
        ds_test = ds_raw_test.map(
            lambda text, label: text_preprocessor.encode_map_fn(text, label)
        )

        train_data = ds_train.padded_batch(32, padded_shapes=([-1], []))
        valid_data = ds_valid.padded_batch(32, padded_shapes=([-1], []))
        test_data = ds_test.padded_batch(32, padded_shapes=([-1], []))

        vocab_size = len(token_counts) + 2
        num_classes = target.nunique()

        sentiment_model = SentimentModel()
        if OPTUNA:
            sentiment_model.Optuna(
                vocab_size, num_classes, train_data, valid_data, test_data
            )
        else:
            if os.path.isfile("./models/sentiment_binary.keras") is False:
                model = sentiment_model.build_model(vocab_size, num_classes)
                sentiment_model.train_and_evaluate(
                    model, train_data, valid_data, test_data
                )
            else:
                sentiment_model.evaluate(test_data)
                # df = pd.read_csv(
                #     "./data/tripadvisor_hotel_reviews.csv", encoding="utf-8"
                # )
                # df.loc[df.Rating < 3, "Label"] = 0
                # df.loc[df.Rating == 3, "Label"] = 1
                # df.loc[df.Rating > 3, "Label"] = 2
                # df["Label"] = df["Label"].astype(int)
                # list_eval = []
                # list_predic = []
                # for i in range(0, 100):
                #     a = np.random.randint(0, 1000)
                #     example_text = df["Review"].iloc[a]
                #     rating = df["Rating"].iloc[a]
                #     label = df["Label"].iloc[a]
                #     print(label)
                #     df_test = pd.DataFrame({"Review": [example_text], "Label": label})
                #     target = df_test.pop("Label")
                #     ds_raw = tf.data.Dataset.from_tensor_slices(
                #         (df_test["Review"].values, target.values)
                #     )
                #     text_preprocessor.fit_tokenizer(ds_raw)

                #     ds_final = ds_raw.map(
                #         lambda text, label: text_preprocessor.encode_map_fn(text, label)
                #     )

                #     final = ds_final.padded_batch(32, padded_shapes=([-1], []))

                #     # Preprocess the example text
                #     preprocessed_text = text_preprocessor.preprocess_text(example_text)

                #     # ds_raw = tf.data.Dataset.from_tensor_slices(
                #     #     (example_text.values, label.values)
                #     # )
                #     # text_preprocessor.fit_tokenizer(ds_raw)

                #     # ds_final = ds_raw.map(
                #     #     lambda text, label: text_preprocessor.encode_map_fn(text, label)
                #     # )

                #     # final = ds_final.padded_batch(32, padded_shapes=([-1], []))

                #     # # Preprocess the example text
                #     preprocessed_text = text_preprocessor.preprocess_text(example_text)
                #     encoded_text = text_preprocessor.tokenizer.texts_to_sequences(
                #         [preprocessed_text]
                #     )
                #     padded_text = tf.keras.preprocessing.sequence.pad_sequences(
                #         encoded_text, maxlen=None
                #     )
                #     # Make predictions
                #     y_classes, predictions = sentiment_model.predict_text(padded_text)
                #     test_results = sentiment_model.evaluate_text(final)
                #     list_eval.append(test_results)
                #     list_predic.append(y_classes)
                # print(np.mean(np.array(list_eval)))


if __name__ == "__main__":
    main()
