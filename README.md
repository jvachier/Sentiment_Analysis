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
│   ├── __init__.py                 # Makes the folder a Python package
│   └── voice_to_text_app.py        # Main application script
│
├── src/                            # Source folder
│   ├── data/                       # Dataset folder
│   ├── models/                     # Saved models
│   │   ├── inference_model.keras
│   │   ├── sentiment_keras_binary.keras
│   │   └── optuna_model_binary.json # Hyperparameter optimization results
│   ├── configurations/             # Configuration files
│   │   ├── model_builder_config.json
│   │   ├── model_trainer_config.json
│   │   └── optuna_config.json
│   ├── modules/                    # Custom Python modules
│   │   ├── __init__.py             # Makes the folder a Python package
│   │   ├── load_data.py            # Data loading utilities
│   │   ├── model.py                # Model definition and training
│   │   ├── data_preprocess.py      # Data preprocessing utilities
│   │   ├── text_vectorizer.py      # Text vectorization utilities
│   │   ├── utils.py                # Enum classes
│   │   ├── sentiment_analysis_utils.py # Utils functions for sentiment_analysis
│   │   └── speech_to_text.py       # Speech-to-text and sentiment analysis logic
│   ├── sentiment_analysis_bert_other.py # Sentiment analysis using BERT
│   └── sentiment_analysis.py       # Sentiment analysis pipeline script
│
├── tests/                          # Unit and integration tests
│   └── test_model.py               # Tests for speech_to_text.py
│
├── .github/                        # GitHub-specific files
│   ├── workflows/                  # GitHub Actions workflows
│   ├── AUTHORS.md                  # List of authors
│   ├── CODEOWNERS                  # Code owners for the repository
│   ├── CONTRIBUTORS.md             # List of contributors
│   └── pull_request_template.md    # Pull request template
│
├── .gitignore                      # Git ignore file
├── LICENSE                         # License file
├── Makefile                        # Makefile for common tasks
├── pyproject.toml                  # Poetry configuration file
├── README.md                       # Project documentation
├── requirements.txt                # Optional: pip requirements file
└── ruff.toml                       # Ruff configuration file
```

## Usage

1. **Prepare the Dataset**
   
Place your dataset in the `src/data/` folder. The default dataset used is `tripadvisor_hotel_reviews.csv`.

2. **Train the Model**

Run the main script to train the model:

```bash
poetry run python src/sentiment_analysis.py
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