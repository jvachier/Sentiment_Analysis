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

model = tf.keras.models.load_model("./models/sentiment.keras")


class SpeechToText:
    def __init__(self, model_path):
        # if not os.path.exists(model_path):
        #     print(
        #         f"Please download a Vosk model from https://alphacephei.com/vosk/models and unpack as '{model_path}' in the current folder."
        #     )
        #     exit(1)

        # self.model = vosk.Model(model_path)
        # self.audio = pyaudio.PyAudio()
        # self.stream = self.audio.open(
        #     format=pyaudio.paInt16,
        #     channels=1,
        #     rate=16000,
        #     input=True,
        #     frames_per_buffer=16000,
        # )
        # self.stream.start_stream()
        # self.rec = vosk.KaldiRecognizer(self.model, 16000)
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

    def predict_sentiment(self, text=None):
        # Load the sentiment analysis model
        # sentiment_model = SentimentModel()

        # tf.keras.backend.clear_session()
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
        # text = "This is really bad!"
        # Convert string to StringIO object
        # data_io = StringIO(text)
        # Read the string as if it were a CSV file
        # df = pd.read_csv(data_io, header=None, names=["Review"])
        # token_counts = text_preprocessor.fit_tokenizer(df)
        # encoded_texts = df["Review"].map(
        #     lambda text: text_preprocessor.encode_map_fn(text)
        # )

        if text is None:
            # text = "Amzing place to stay. The staff is very friendly and helpful. The rooms are clean and comfortable."
            # text = (
            #     "nice hotel expensive parking got good deal stay hotel anniversary, "
            #     "arrived late evening took advice previous reviews did valet parking, "
            #     "check quick easy, little disappointed non-existent view room room "
            #     "clean nice size, bed comfortable woke stiff neck high pillows, not "
            #     "soundproof like heard music room night morning loud bangs doors "
            #     "opening closing hear people talking hallway, maybe just noisy "
            #     "neighbors, aveda bath products nice, did not goldfish stay nice "
            #     "touch taken advantage staying longer, location great walking distance "
            #     "shopping, overall nice experience having pay 40 parking night,"
            # )
            text = "disappointed say anticipating stay hotel monaco based reviews seen tripadvisor, definitely disppointment, decor room hotel envisioned nice, housekeeping staff impressive extremely polite cheery helpful, desk bellmen standard customer service, lots little things easily overlooked broken make mirror sagging post bathroom jiggly entrance handle room did n't feel secure handle nearly falling door, husband early morning flight mid-week asked cab called, suggested cab ride cost double private car flat-rate n't case, knew cost cab 30 car 38 bell staff insisted like 65 taxi ride, cab driver later explained bell staff kick referrals, preferred guests city and/or hotel hotel staff looking best interests, new years planned dining restaurant hotel new year day, came dressed dinner told desk closed holiday, asked nearby response good luck finding today, good luck, thanks help, feature needed hotel business centre work, downstairs service completely semi-funcitional, internet service out-sourced ended spending 50 money n't work no refund 2.5 hours vacation time trouble-shooting internet company toll free service line, asked desk times assistance told, worst experience time asked desk assistance got response included shrug say oh bad, majority staff attitude just simply did not care needed help dared intrude time got snotty response, occasions caught desk girls smirking thought turned n't, talk frustrating, honesty n't wait check, nice decor liked goldfish housekeeping staff great end hotel want dash door n't way not concerned taking greet pretty ignored desk area unless approached, time street westin, saw nothing smiling faces coming going hotel,"
            # text = "hotel stayed hotel monaco cruise, rooms generous decorated uniquely, hotel remodeled pacific bell building charm sturdiness, everytime walked bell men felt like coming home, secure, great single travelers, location fabulous, walk things pike market space needle.little grocery/drug store block away, today green, bravo, 1 double bed room room bed couch separated curtain, snoring mom slept curtain, great food nearby,"
            # text = "horrible customer service hotel stay february 3rd 4th 2007my friend picked hotel monaco appealing website online package included champagne late checkout 3 free valet gift spa weekend, friend checked room hours earlier came later, pulled valet young man just stood, asked valet open said, pull bags didn__Ç_é_ offer help, got garment bag suitcase came car key room number says not valet, car park car street pull, left key working asked valet park car gets, went room fine bottle champagne oil lotion gift spa, dressed went came got bed noticed blood drops pillows sheets pillows, disgusted just unbelievable, called desk sent somebody 20 minutes later, swapped sheets left apologizing, sunday morning called desk speak management sheets aggravated rude, apparently no manager kind supervisor weekend wait monday morning, young man spoke said cover food adding person changed sheets said fresh blood rude tone, checkout 3pm package booked, 12 1:30 staff maids tried walk room opening door apologizing closing, people called saying check 12 remind package, finally packed things went downstairs check, quickly signed paper took, way took closer look room, unfortunately covered food offered charged valet, called desk ask charges lady answered snapped saying aware problem experienced monday like told earlier, life treated like hotel, not sure hotel constantly problems lucky ones stay recommend anybody know,"
            # text = "Very good and Amazing."

        df = pd.DataFrame([text], columns=["Review"])

        text_preprocessor.fit_tokenizer(df)

        # Preprocess and encode the text
        preprocessed_text = text_preprocessor.preprocess_text(text)
        encoded_text = text_preprocessor.tokenizer.texts_to_sequences(
            [preprocessed_text]
        )
        padded_text = tf.keras.preprocessing.sequence.pad_sequences(
            encoded_text, maxlen=None
        )
        # Make predictions
        prediction = model.predict(padded_text)
        negative, neutral, positive = prediction[0]
        print(prediction)
        max = np.argmax(prediction[0], 0)
        print(max)
        if max > 1:
            sentiment = "Positive"
        elif max < 1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        return sentiment


# Initialize the SpeechToText class
speech_to_text = SpeechToText(model_path="./vosk-model-en-us-0.22")
# speech_to_text = SpeechToText(model_path="vosk-model-small-sv-rhasspy-0.15")

sentiment = speech_to_text.predict_sentiment()
print(sentiment)

quit()

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
