from src.modules.load_data import DataLoader
from src.modules.model_bert_other import SentimentModelBert


def main():
    """
    Main function to execute the sentiment analysis pipeline.

    This function performs the following steps:
    1. Loads the dataset using the DataLoader class.
    2. Prepares the training, validation, and test datasets for the sentiment analysis model.
    3. Builds a BERT-based sentiment analysis model.
    4. Trains the model and evaluates its performance on the validation and test datasets.

    The dataset is expected to be located at "./data/tripadvisor_hotel_reviews.csv".

    Classes and Methods Used:
    - DataLoader: Handles loading and splitting the dataset.
    - SentimentModelBert: Prepares data, builds the model, and handles training and evaluation.

    Variables:
    - data_loader: Instance of DataLoader to load and preprocess the dataset.
    - ds_raw, ds_raw_train, ds_raw_valid, ds_raw_test: Raw and split datasets.
    - target: Target column from the dataset.
    - sentiment_model: Instance of SentimentModelBert for model-related operations.
    - train_data, valid_data, test_data: Prepared datasets for training, validation, and testing.
    - num_classes: Number of unique target classes in the dataset.
    - model: The BERT-based sentiment analysis model.
    """
    data_loader = DataLoader(data_path="./data/tripadvisor_hotel_reviews.csv")
    ds_raw, ds_raw_train, ds_raw_valid, ds_raw_test, target = data_loader.load_data()

    sentiment_model = SentimentModelBert()
    train_data = sentiment_model.prepare_data(ds_raw_train, sentiment_model.batch_size)
    valid_data = sentiment_model.prepare_data(ds_raw_valid, sentiment_model.batch_size)
    test_data = sentiment_model.prepare_data(ds_raw_test, sentiment_model.batch_size)

    num_classes = target.nunique()

    model = sentiment_model.build_model(num_classes)
    sentiment_model.train_and_evaluate(model, train_data, valid_data, test_data)


if __name__ == "__main__":
    main()
