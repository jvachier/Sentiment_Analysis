[![Linting: Ruff](https://img.shields.io/badge/linting-ruff-yellowgreen)](https://github.com/charliermarsh/ruff)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-red)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Sentiment Analysis

This repository contains a sentiment analysis application that uses TensorFlow and Keras to classify text data into positive or negative sentiments. The application includes a speech-to-text interface (`voice_to_text_app.py`) built with Dash, which allows users to record audio, transcribe it into text, and analyze its sentiment.


## Features

- **Speech-to-Text**: Converts spoken audio into text using the Vosk library.
- **Text Preprocessing**: Uses TensorFlow's `TextVectorization` layer to tokenize and vectorize text data.
- **Bidirectional LSTM Model**: Implements a deep learning model with embedding, bidirectional LSTM, and dense layers for sentiment classification.
- **Training and Evaluation**: Includes functionality to train the model on a dataset and evaluate its performance on validation and test sets.
- **Inference**: Provides an inference pipeline to predict sentiment for new text inputs.
- **Interactive Application**: A Dash-based web application for real-time speech-to-text and sentiment analysis.

## Requirements

- Python 3.8 or higher
- TensorFlow 2.12 or higher
- Poetry for dependency management
- NumPy
- Vosk for speech-to-text
- Dash for the web application
- PyAudio for audio input

### Install Dependencies

Install Poetry if you haven't already:

```bash
pip install poetry
```

Then, install the project dependencies:

```bash
poetry install
```

## Project Structure

```
Sentiment_Analysis/
├── app/                            # Application-specific files
│   ├── voice_to_text_app.py        # Main application script
│   ├── assets/                     # Static assets (CSS, JS, images)
│   └── callbacks/                  # Modularized Dash callbacks (optional)
│
├── src/                            # Source folder
│   ├── data/                       # Dataset folder
│   │   └── tripadvisor_hotel_reviews.csv
│   ├── models/                     # Saved models
│   │   ├── inference_model.keras
│   │   ├── sentiment_keras_binary.keras
│   │   └── optuna_model.json       # Hyperparameter optimization results
│   ├── modules/                    # Custom Python modules
│   │   ├── __init__.py             # Makes the folder a Python package
│   │   ├── load_data.py            # Data loading utilities
│   │   ├── model.py                # Model definition and training
│   │   ├── data_preprocess.py      # Data preprocessing utilities
│   │   ├── text_vecto.py           # Text vectorization utilities
│   │   └── speech_to_text.py       # Speech-to-text and sentiment analysis logic
│
├── tests/                          # Unit and integration tests
│   ├── __init__.py
│   ├── test_load_data.py           # Tests for load_data.py
│   ├── test_model.py               # Tests for model.py
│   ├── test_text_vecto.py          # Tests for text_vecto.py
│   └── test_speech_to_text.py      # Tests for speech_to_text.py
│
├── .github/                        # GitHub-specific files
│   ├── workflows/                  # GitHub Actions workflows
│   ├── AUTHORS.md                  # List of authors
│   ├── CODEOWNERS                  # Code owners for the repository
│   ├── CONTRIBUTORS.md             # List of contributors
│   └── pull_request_template.md    # Pull request template
│
├── .venv/                          # Virtual environment (ignored in .gitignore)
├── .gitignore                      # Git ignore file
├── LICENSE                         # License file
├── Makefile                        # Makefile for common tasks
├── pyproject.toml                  # Poetry configuration file
├── README.md                       # Project documentation
├── requirements.txt                # Optional: pip requirements file
├── ruff.toml                       # Ruff configuration file
└── sentiment_analysis.py           # Sentiment analysis pipeline script
```

## Usage

1. **Prepare the Dataset**
   
Place your dataset in the `src/data/` folder. The default dataset used is `tripadvisor_hotel_reviews.csv`.

2. **Train the Model**

Run the main script to train the model:

```bash
poetry run python sentiment_analysis.py
```

The script will preprocess the data, train the model, and save it in the `src/models/` folder.

3. **Inference**

The script includes a test example for inference. Modify the `raw_text_data` variable in `sentiment_analysis.py` to test with your own text input.

4. **Evaluate the Model**

The script evaluates the model on the test dataset and prints the accuracy.

Example Output:

```
Test Acc.: 95.00%
```

## Customization

- Modify hyperparameters like `embedding_dim`, `lstm_units`, and `dropout_rate` in `src/modules/model.py`.
- Replace the dataset in `src/data/` with your own CSV file.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.