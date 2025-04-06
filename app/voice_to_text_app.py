import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import base64
import logging
from src.modules.speech_to_text import SpeechToText

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
