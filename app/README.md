# Speech-to-Text Translation and Sentiment Analysis App

This directory contains the Dash web application for real-time speech processing with translation and sentiment analysis capabilities.

## Overview

This interactive web application allows users to:

- **Record and convert spoken English to text** using Vosk speech recognition.
- **Translate the recognized English text to French** using a Transformer model.
- **Analyze the sentiment** of the recognized text (positive or negative).
- **Download the recognized text** as a text file.

## Features

- **Real-time Audio Processing**: Records audio directly from the user's microphone.
- **Speech Recognition**: Converts spoken words to text using Vosk.
- **Translation**: Translates English text to French using a Transformer-based model.
- **Sentiment Analysis**: Determines if the speech content is positive or negative.
- **Download Option**: Save recognized text for future reference.

## Prerequisites

Before running the application, ensure you have:

1. **Vosk Model**:
   - Download the Vosk model (`vosk-model-en-us-0.22`) from the [official Vosk repository](https://alphacephei.com/vosk/models).
   - Place the extracted folder in the project root directory.

2. **English-French Dataset**:
   - Ensure the dataset (`src/data/en-fr.parquet`) is available for training and preprocessing.

3. **Transformer Model**:
   - Train or download the Transformer model for translation.
   - Ensure the model is saved at the path defined in `ModelPaths.TRANSFORMER_MODEL.value`.

4. **Inference Model**:
   - Ensure the sentiment analysis inference model is available at the path defined in `ModelPaths.INFERENCE_MODEL.value`.

5. **Dependencies**:
   - Install all project dependencies using Poetry:
     ```bash
     poetry install
     ```

## How to Run

From the project root directory:

```bash
poetry run python app.py
```

The application will start and be accessible at: [http://127.0.0.1:8050](http://127.0.0.1:8050)

## Usage Instructions

### Start Recording:
- Click the **"Start Recording"** button.
- Speak clearly into your microphone in English.

### Stop Recording:
- Click the **"Stop Recording"** button when finished speaking.

### View Results:
- The recognized English text will appear.
- Below that, you'll see the French translation.
- The sentiment analysis (positive or negative) will be displayed.

### Download Results:
- Click **"Download Recognized Text"** to save the recognized text as a `.txt` file.

## Code Structure

The application consists of:

1. **Initialization**: Sets up the `SpeechToText` module, loads the translation model, and configures the Dash app.
2. **User Interface**: Defines the layout with buttons and display areas.
3. **Callback Function**: Handles the recording process, speech recognition, translation, and sentiment analysis.
4. **Main Entry Point**: Starts the Dash server.

## Troubleshooting

If you encounter issues:

- Verify the Vosk model is correctly installed at `vosk-model-en-us-0.22`.
- Check that the dataset exists at `en-fr.parquet`.
- Ensure the Transformer model is available at the path defined in `ModelPaths.TRANSFORMER_MODEL.value`.
- Ensure the sentiment analysis inference model is available at the path defined in `ModelPaths.INFERENCE_MODEL.value`.
- Look for error messages in the console logs.

## Technical Details

- **Framework**: Dash web application framework.
- **Speech Recognition**: Vosk speech recognition model.
- **Translation**: Custom Transformer model for English-to-French translation.
- **Sentiment Analysis**: Neural network-based sentiment classifier.
- **State Management**: Uses Dash callbacks and `dcc.Store` for managing application state.

## Development Notes

- The app runs in debug mode by default.
- For production deployment, set `debug=False` in the `app.run_server()` method.