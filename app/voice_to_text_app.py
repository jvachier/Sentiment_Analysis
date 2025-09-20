import base64
import logging
import sys
from pathlib import Path
from typing import Any, Tuple

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

# Add src directory to Python path
# src_path = Path(__file__).parent.parent / "src"
# sys.path.insert(0, str(src_path))

from modules.data_processor import DatasetProcessor, TextPreprocessor
from modules.speech_to_text import SpeechToText
from modules.utils import ModelPaths
from translation_french_english import (
    translation_test as test_translation,
    transformer_model,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize the SpeechToText class
speech_to_text = SpeechToText(model_path="./vosk-model-en-us-0.22")

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Speech to Text Translation and Sentiment Analysis"

# Load the Transformer model for translation
logging.info("Initializing dataset processor for translation.")
processor = DatasetProcessor(file_path="src/data/en-fr.parquet")
processor.load_data()
processor.process_data()
data_splits = processor.shuffle_and_split()
train_df, val_df, test_df = (
    data_splits["train"],
    data_splits["validation"],
    data_splits["test"],
)

logging.info("Initializing text preprocessor for translation.")
preprocessor = TextPreprocessor()
preprocessor.adapt(train_df)

# Create TensorFlow datasets
train_ds = preprocessor.make_dataset(train_df)
val_ds = preprocessor.make_dataset(val_df)

transformer_model_path = ModelPaths.TRANSFORMER_MODEL.value
transformer = transformer_model(transformer_model_path, preprocessor, train_ds, val_ds)

# Define the layout of the Dash app
app.layout = html.Div(
    style={
        "fontFamily": "Arial, sans-serif",
        "margin": "0 auto",
        "maxWidth": "800px",
        "padding": "20px",
    },
    children=[
        html.H1(
            "Speech to Text Translation and Sentiment Analysis",
            style={"textAlign": "center"},
        ),
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
            id="translated-text",
            style={
                "whiteSpace": "pre-wrap",
                "marginBottom": "20px",
                "fontWeight": "bold",
            },
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
        Output("translated-text", "children"),
        Output("sentiment-result", "children"),
        Output("download-link", "href"),
        Output("recording-state", "data"),
    ],
    [Input("start-record-button", "n_clicks"), Input("stop-record-button", "n_clicks")],
    [State("recording-state", "data")],
)
def update_output(
    start_n_clicks: int, stop_n_clicks: int, recording_state: bool
) -> Tuple[Any, Any, Any, Any, bool]:
    """
    Update the app's output based on user interactions.

    Args:
        start_n_clicks (int): Number of clicks on the "Start Recording" button.
        stop_n_clicks (int): Number of clicks on the "Stop Recording" button.
        recording_state (bool): Current recording state.

    Returns:
        tuple: Updated recognized text, translated text, sentiment result, download link, and recording state.
    """
    if start_n_clicks > stop_n_clicks and not recording_state:
        speech_to_text.start_recording()
        start_n_clicks = 0
        return "Recording...", "", "", "", True
    elif stop_n_clicks > start_n_clicks and recording_state:
        speech_to_text.stop_recording()
        recognized_text = speech_to_text.get_recognized_text()
        sentiment = speech_to_text.predict_sentiment(text=recognized_text)
        encoded_text = base64.b64encode(recognized_text.encode()).decode()

        # Perform translation
        translated_text = test_translation(
            transformer, preprocessor, input_sentence=recognized_text
        )

        href = f"data:text/plain;base64,{encoded_text}"
        stop_n_clicks = 0
        return (
            recognized_text,
            f"Translated: {translated_text}",
            f"Sentiment: {sentiment}",
            href,
            False,
        )
    return "", "", "", "", recording_state


if __name__ == "__main__":
    """
    Entry point of the application. Starts the Dash server.
    """
    logging.info("Starting Dash server.")
    app.run_server(debug=True)
