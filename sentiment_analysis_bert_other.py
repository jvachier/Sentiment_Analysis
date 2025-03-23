from modules.load_data import DataLoader
from modules.data_preprocess import TextPreprocessor
from modules.model_bert_other import SentimentModel, SentimentModelBert
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


if __name__ == "__main__":
    main()
