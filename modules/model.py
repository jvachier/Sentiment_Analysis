import tensorflow as tf
from transformers import TFBertModel, BertTokenizer


class SentimentModelBert:
    def __init__(
        self,
        model_name="bert-base-uncased",
        max_length=128,
        learning_rate=1e-5,
        batch_size=32,
        epochs=20,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def encodeBB(self, text_tensor, label_tensor):
        text = text_tensor.numpy().decode("utf-8")
        encoded_text = self.encodeB(text)
        return (
            encoded_text["input_ids"][0],
            encoded_text["attention_mask"][0],
            label_tensor,
        )

    def encode_map_fnB(self, text, label):
        return tf.py_function(
            self.encodeBB, inp=[text, label], Tout=(tf.int32, tf.int32, tf.int64)
        )

    def prepare_data(self, dataset, batch_size):
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
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="tf",
        )

    def build_model(self, num_classes):
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
        output = tf.keras.layers.Dense(num_classes, activation="softmax")(dropout)

        model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )
        return model

    def train_and_evaluate(self, model, train_data, valid_data, test_data):
        model.summary()
        model.fit(train_data, validation_data=valid_data, epochs=self.epochs)
        test_results = model.evaluate(test_data)
        print("Test Acc.: {:.2f}%".format(test_results[1] * 100))


class SentimentModel:
    def __init__(
        self,
        embedding_dim=50,
        lstm_units=128,
        dropout_rate=0.5,
        learning_rate=1e-4,
        batch_size=32,
        epochs=10,
    ):
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def build_model(self, vocab_size, num_classes):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(None,), dtype="int32"))
        model.add(
            tf.keras.layers.Embedding(
                input_dim=vocab_size, output_dim=self.embedding_dim, name="embed-layer"
            )
        )
        model.add(
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    self.lstm_units, return_sequences=True, name="lstm-layer"
                ),
                name="bidir-lstm1",
            )
        )
        model.add(
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    self.lstm_units, return_sequences=False, name="lstm-layer"
                ),
                name="bidir-lstm2",
            )
        )
        model.add(tf.keras.layers.Dropout(self.dropout_rate))
        model.add(tf.keras.layers.Dense(128, activation="gelu"))
        model.add(tf.keras.layers.Dropout(self.dropout_rate))
        model.add(tf.keras.layers.Dense(64, activation="gelu"))
        model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )
        return model

    def train_and_evaluate(self, model, train_data, valid_data, test_data):
        model.summary()
        model.fit(train_data, validation_data=valid_data, epochs=self.epochs)
        test_results = model.evaluate(test_data)
        print("Test Acc.: {:.2f}%".format(test_results[1] * 100))
