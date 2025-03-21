import os
import vosk
import pyaudio
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import threading
import base64
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from modules.data_preprocess import TextPreprocessorNoLabel
from modules.model import SentimentModel
from io import StringIO

# Load the sentiment analysis model
sentiment_model = SentimentModel()
model = tf.keras.models.load_model("./models/sentiment.keras")


class SpeechToText:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            print(
                f"Please download a Vosk model from https://alphacephei.com/vosk/models and unpack as '{model_path}' in the current folder."
            )
            exit(1)

        self.model = vosk.Model(model_path)
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=16000,
        )
        self.stream.start_stream()
        self.rec = vosk.KaldiRecognizer(self.model, 16000)
        self.recognized_text = []
        self.recording = False

    def start_recording(self):
        self.recognized_text = []
        self.recording = True
        threading.Thread(target=self.record_audio).start()

    def stop_recording(self):
        self.recording = False

    def record_audio(self):
        try:
            while self.recording:
                data = self.stream.read(4000, exception_on_overflow=False)
                if len(data) == 0:
                    break
                if self.rec.AcceptWaveform(data):
                    result = json.loads(self.rec.Result())
                    text = result.get("text", "")
                    self.recognized_text.append(text)
                    print(f"Recognized: {text}")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            final_result = json.loads(self.rec.FinalResult())
            text = final_result.get("text", "")
            self.recognized_text.append(text)
            print(f"Final result: {text}")
            with open("recognized_text.txt", "w") as text_file:
                text_file.write("\n".join(self.recognized_text))
            print("Text written to recognized_text.txt")

    def get_recognized_text(self):
        return " ".join(self.recognized_text)

    def predict_sentiment(self, text):
        tf.keras.backend.clear_session()
        # # Preprocess the text using TextPreprocessorNoLabel
        text_preprocessor = TextPreprocessorNoLabel()
        # preprocessed_text = text_preprocessor.preprocess_text(text)
        # # Tokenize and pad the text as required by the model
        # tokenizer = tf.keras.preprocessing.text.Tokenizer()
        # tokenizer.fit_on_texts([preprocessed_text])
        # tokenized_text = tokenizer.texts_to_sequences([preprocessed_text])
        # padded_text = tf.keras.preprocessing.sequence.pad_sequences(
        #     tokenized_text, maxlen=100
        # )
        # Convert string to StringIO object
        data_io = StringIO(text)
        # Read the string as if it were a CSV file
        df = pd.read_csv(data_io, header=None, columns=["Review"])
        ds_test = df.map(lambda text: text_preprocessor.encode_map_fn(text))
        test_data = ds_test.padded_batch(32, padded_shapes=([-1]))
        print(test_data)

        # prediction = model.predict(padded_text)
        prediction = model.predict(test_data)
        negative, neutral, positive = prediction[0]
        print(negative, neutral, positive)
        if positive > 0.6:
            sentiment = "Positive"
        elif negative > 0.6:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        return sentiment


# Initialize the SpeechToText class
speech_to_text = SpeechToText(model_path="./vosk-model-en-us-0.22")
# speech_to_text = SpeechToText(model_path="vosk-model-small-sv-rhasspy-0.15")

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Speech to Text"),
        html.Button("Start Recording", id="start-record-button", n_clicks=0),
        html.Button("Stop Recording", id="stop-record-button", n_clicks=0),
        html.Div(id="recognized-text"),
        html.Div(id="sentiment-result"),
        html.A(
            "Download Recognized Text",
            id="download-link",
            download="recognized_text.txt",
            href="",
            target="_blank",
        ),
        dcc.Store(id="recording-state", data=False),
    ]
)


@app.callback(
    [
        Output("recognized-text", "children"),
        Output("sentiment-result", "children"),
        Output("download-link", "href"),
        Output("recording-state", "data"),
    ],
    [Input("start-record-button", "n_clicks"), Input("stop-record-button", "n_clicks")],
    [State("recording-state", "data")],
)
def update_output(start_n_clicks, stop_n_clicks, recording_state):
    if start_n_clicks > stop_n_clicks and not recording_state:
        speech_to_text.start_recording()
        start_n_clicks = 0
        return "Recording...", "", "", True
    elif stop_n_clicks > start_n_clicks and recording_state:
        speech_to_text.stop_recording()
        recognized_text = speech_to_text.get_recognized_text()
        sentiment = speech_to_text.predict_sentiment(recognized_text)
        encoded_text = base64.b64encode(recognized_text.encode()).decode()
        href = f"data:text/plain;base64,{encoded_text}"
        stop_n_clicks = 0
        return recognized_text, f"Sentiment: {sentiment}", href, False
    return "", "", "", recording_state


if __name__ == "__main__":
    app.run_server(debug=True)
