import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
import optuna
import json


class SentimentModelBert:
    """
    A class to define, train, and evaluate a sentiment analysis model using BERT.

    Attributes:
        model_name (str): The name of the pre-trained BERT model.
        max_length (int): The maximum sequence length for tokenization.
        learning_rate (float): The learning rate for the optimizer.
        batch_size (int): The batch size for training.
        epochs (int): The number of training epochs.
        tokenizer (BertTokenizer): The tokenizer for the BERT model.
    """

    def __init__(
        self,
        model_name="bert-base-uncased",
        max_length=128,
        learning_rate=1e-4,
        batch_size=32,
        epochs=5,
    ):
        """
        Initialize the SentimentModelBert class.

        Args:
            model_name (str): The name of the pre-trained BERT model.
            max_length (int): The maximum sequence length for tokenization.
            learning_rate (float): The learning rate for the optimizer.
            batch_size (int): The batch size for training.
            epochs (int): The number of training epochs.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def encodeBB(self, text_tensor, label_tensor):
        """
        Encode text and label tensors into BERT-compatible input.

        Args:
            text_tensor (tf.Tensor): The input text tensor.
            label_tensor (tf.Tensor): The label tensor.

        Returns:
            tuple: Encoded input IDs, attention masks, and labels.
        """
        text = text_tensor.numpy().decode("utf-8")
        encoded_text = self.encodeB(text)
        return (
            encoded_text["input_ids"][0],
            encoded_text["attention_mask"][0],
            label_tensor,
        )

    def encode_map_fnB(self, text, label):
        """
        Map function to encode text and label into BERT-compatible input.

        Args:
            text (tf.Tensor): The input text tensor.
            label (tf.Tensor): The label tensor.

        Returns:
            tuple: Encoded input IDs, attention masks, and labels.
        """
        return tf.py_function(
            self.encodeBB, inp=[text, label], Tout=(tf.int32, tf.int32, tf.int64)
        )

    def prepare_data(self, dataset, batch_size):
        """
        Prepare the dataset for training by encoding and batching.

        Args:
            dataset (tf.data.Dataset): The input dataset.
            batch_size (int): The batch size for training.

        Returns:
            tf.data.Dataset: The prepared dataset.
        """
        dataset = dataset.map(lambda text, label: self.encode_map_fnB(text, label))
        dataset = dataset.map(
            lambda input_ids, attention_mask, label: (
                {"input_ids": input_ids, "attention_mask": attention_mask},
                label,
            )
        )
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=({"input_ids": [None], "attention_mask": [None]}, []),
        )
        return dataset

    def encodeB(self, texts):
        """
        Tokenize the input text using the BERT tokenizer.

        Args:
            texts (str or list): The input text(s) to tokenize.

        Returns:
            dict: Tokenized input IDs and attention masks.
        """
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="tf",
        )

    def build_model(self, num_classes):
        """
        Build and compile the BERT-based sentiment analysis model.

        Args:
            num_classes (int): The number of output classes.

        Returns:
            tf.keras.Model: The compiled BERT model.
        """
        bert_model = TFBertModel.from_pretrained(
            self.model_name, output_hidden_states=False
        )
        input_ids = tf.keras.layers.Input(
            shape=(self.max_length,), dtype=tf.int32, name="input_ids"
        )
        attention_mask = tf.keras.layers.Input(
            shape=(self.max_length,), dtype=tf.int32, name="attention_mask"
        )

        bert_output = bert_model(input_ids, attention_mask=attention_mask)
        cls_token = bert_output.last_hidden_state[:, 0, :]
        dropout = tf.keras.layers.Dropout(0.3)(cls_token)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(dropout)

        model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.legacy.RMSprop(
                learning_rate=self.learning_rate
            ),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )
        return model

    def train_and_evaluate(self, model, train_data, valid_data, test_data):
        """
        Train and evaluate the BERT model.

        Args:
            model (tf.keras.Model): The BERT model to train.
            train_data (tf.data.Dataset): The training dataset.
            valid_data (tf.data.Dataset): The validation dataset.
            test_data (tf.data.Dataset): The test dataset.
        """
        model.summary()
        with tf.device("/device:GPU:0"):
            model.fit(train_data, validation_data=valid_data, epochs=self.epochs)
        test_results = model.evaluate(test_data)
        print("Test Acc.: {:.2f}%".format(test_results[1] * 100))
