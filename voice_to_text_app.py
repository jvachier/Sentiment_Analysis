import os
import sys
import vosk
import pyaudio
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import threading
import base64
import json
import tensorflow as tf
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# from modules.model_bert_other import SentimentModel
from modules.model import SentimentModelKeras


class SpeechToText:
    """
    A class to handle speech-to-text conversion and sentiment analysis.

    Attributes:
        model (vosk.Model): The Vosk speech recognition model.
        audio (pyaudio.PyAudio): PyAudio instance for audio input.
        stream (pyaudio.Stream): Audio stream for recording.
        rec (vosk.KaldiRecognizer): Vosk recognizer for speech-to-text conversion.
        recognized_text (list): List of recognized text segments.
        recording (bool): Flag to indicate if recording is active.
    """

    def __init__(self, model_path):
        """
        Initialize the SpeechToText class.

        Args:
            model_path (str): Path to the Vosk model directory.
        """
        if not os.path.exists(model_path):
            logging.error(
                f"Please download a Vosk model from https://alphacephei.com/vosk/models and unpack as '{model_path}' in the current folder."
            )
            sys.exit(1)

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
        """
        Start recording audio and process it for speech-to-text conversion.
        """
        self.recognized_text = []
        self.recording = True
        threading.Thread(target=self.record_audio).start()
        logging.info("Recording started.")

    def stop_recording(self):
        """
        Stop recording audio.
        """
        self.recording = False
        logging.info("Recording stopped.")

    def record_audio(self):
        """
        Record audio from the microphone and convert it to text using Vosk.
        """
        try:
            while self.recording:
                data = self.stream.read(4000, exception_on_overflow=False)
                if len(data) == 0:
                    break
                if self.rec.AcceptWaveform(data):
                    result = json.loads(self.rec.Result())
                    text = result.get("text", "")
                    self.recognized_text.append(text)
                    logging.info(f"Recognized: {text}")
        except Exception as e:
            logging.error(f"Error during recording: {e}")
        finally:
            final_result = json.loads(self.rec.FinalResult())
            text = final_result.get("text", "")
            self.recognized_text.append(text)
            logging.info(f"Final result: {text}")
            with open("recognized_text.txt", "w") as text_file:
                text_file.write("\n".join(self.recognized_text))
            logging.info("Text written to recognized_text.txt")

    def get_recognized_text(self):
        """
        Get the full recognized text.

        Returns:
            str: The concatenated recognized text.
        """
        return " ".join(self.recognized_text)

    def predict_sentiment(self, text):
        """
        Predict the sentiment of the given text using a pre-trained model.

        Args:
            text (str): The input text for sentiment analysis.

        Returns:
            str: The predicted sentiment ("Positive" or "Negative").
        """
        logging.info("Loading sentiment analysis model.")
        sentiment = SentimentModelKeras()
        inference_model = tf.keras.models.load_model("./models/inference_model.keras")
        raw_text_data = tf.convert_to_tensor([text])
        # Make predictions
        prediction = inference_model.predict(raw_text_data)
        sentiment = "Positive" if prediction[0] > 0.51 else "Negative"
        logging.info(f"Predicted sentiment: {sentiment}")
        return sentiment


# Initialize the SpeechToText class
speech_to_text = SpeechToText(model_path="./vosk-model-en-us-0.22")

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Speech to Text Sentiment Analysis"

# Define the layout of the Dash app
app.layout = html.Div(
    style={
        "fontFamily": "Arial, sans-serif",
        "margin": "0 auto",
        "maxWidth": "800px",
        "padding": "20px",
    },
    children=[
        html.H1("Speech to Text Sentiment Analysis", style={"textAlign": "center"}),
        html.Div(
            style={"textAlign": "center", "marginBottom": "20px"},
            children=[
                html.Button(
                    "Start Recording",
                    id="start-record-button",
                    n_clicks=0,
                    style={"marginRight": "10px"},
                ),
                html.Button("Stop Recording", id="stop-record-button", n_clicks=0),
            ],
        ),
        html.Div(
            id="recognized-text",
            style={"whiteSpace": "pre-wrap", "marginBottom": "20px"},
        ),
        html.Div(
            id="sentiment-result", style={"fontWeight": "bold", "marginBottom": "20px"}
        ),
        html.A(
            "Download Recognized Text",
            id="download-link",
            download="recognized_text.txt",
            href="",
            target="_blank",
            style={"display": "block", "textAlign": "center", "marginBottom": "20px"},
        ),
        dcc.Store(id="recording-state", data=False),
    ],
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
    """
    Update the app's output based on user interactions.

    Args:
        start_n_clicks (int): Number of clicks on the "Start Recording" button.
        stop_n_clicks (int): Number of clicks on the "Stop Recording" button.
        recording_state (bool): Current recording state.

    Returns:
        tuple: Updated recognized text, sentiment result, download link, and recording state.
    """
    if start_n_clicks > stop_n_clicks and not recording_state:
        speech_to_text.start_recording()
        start_n_clicks = 0
        return "Recording...", "", "", True
    elif stop_n_clicks > start_n_clicks and recording_state:
        speech_to_text.stop_recording()
        recognized_text = speech_to_text.get_recognized_text()
        sentiment = speech_to_text.predict_sentiment(text=recognized_text)
        encoded_text = base64.b64encode(recognized_text.encode()).decode()

        href = f"data:text/plain;base64,{encoded_text}"
        stop_n_clicks = 0
        return recognized_text, f"Sentiment: {sentiment}", href, False
    return "", "", "", recording_state


if __name__ == "__main__":
    """
    Entry point of the application. Starts the Dash server.
    """
    logging.info("Starting Dash server.")
    app.run_server(debug=True)
