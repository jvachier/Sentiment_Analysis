import os
import sys
import vosk
import pyaudio
import threading
import json
import tensorflow as tf
import logging
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
