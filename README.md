[![Linting: Ruff](https://img.shields.io/badge/linting-ruff-yellowgreen)](https://github.com/charliermarsh/ruff)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-red)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Sentiment Analysis and Translation

This repository contains a sentiment analysis application and an English-to-French translation model. The sentiment analysis application uses TensorFlow and Keras to classify text data into positive or negative sentiments. The translation model implements a Transformer-based architecture for sequence-to-sequence learning.

## Features

### Sentiment Analysis
- **Speech-to-Text**: Converts spoken audio into text using the Vosk library.
- **Text Preprocessing**: Uses TensorFlow's `TextVectorization` layer to tokenize and vectorize text data.
- **Bidirectional LSTM Model**: Implements a deep learning model with embedding, bidirectional LSTM, and dense layers for sentiment classification.
- **Training and Evaluation**: Includes functionality to train the model on a dataset and evaluate its performance on validation and test sets.
- **Inference**: Provides an inference pipeline to predict sentiment for new text inputs.
- **Interactive Application**: A Dash-based web application for real-time speech-to-text and sentiment analysis.

### English-to-French Translation
- **Transformer Model**: Implements a sequence-to-sequence Transformer model for English-to-French translation.
- **BLEU Score Evaluation**: Evaluates the quality of translations using the BLEU metric.
- **Preprocessing**: Includes utilities for tokenizing and vectorizing English and French text.
- **Model Saving and Loading**: Supports saving and loading trained Transformer models for reuse.

---

## Installation

### Install Dependencies

Install Poetry if you haven't already:

```bash
pip install poetry
```

Then, install the project dependencies:

```bash
poetry install
```

---

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
│   │   ├── transformer_best_model.keras
│   │   └── optuna_model_binary.json # Hyperparameter optimization results
│   ├── configurations/             # Configuration files
│   │   ├── model_builder_config.json
│   │   ├── model_trainer_config.json
│   │   └── optuna_config.json
│   ├── modules/                        # Custom Python modules
│   │   ├── __init__.py                 # Makes the folder a Python package
│   │   ├── load_data.py                # Data loading utilities
│   │   ├── model.py                    # Model definition and training
│   │   ├── data_preprocess.py          # Data preprocessing utilities
│   │   ├── text_vectorizer.py          # Text vectorization utilities
│   │   ├── utils.py                    # Enum classes
│   │   ├── sentiment_analysis_utils.py # Utils functions for sentiment_analysis
│   │   ├── transformer_components.py   # Transformer model components
│   │   └── speech_to_text.py           # Speech-to-text and sentiment analysis logic
│   ├── scripts/                                # Scripts for dataset management and preprocessing
│   │   ├── __init__.py                         # Marks the directory as a Python package
│   │   ├── loading_kaggle_dataset_utils.py     # Utilities for downloading and optimizing Kaggle datasets
│   │   ├── loading_kaggle_dataset_script.py    # Script to process Kaggle datasets
│   │   └── README.md                           # Documentation for the scripts folder
│   ├── translation_french_english.py       # English-to-French translation pipeline
│   ├── sentiment_analysis_bert_other.py    # Sentiment analysis using BERT
│   └── sentiment_analysis.py               # Sentiment analysis pipeline script
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
└── ruff.toml                       # Ruff configuration file
```

---

## Usage

### Sentiment Analysis

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

   Output:

   ```
   Test Acc.: 95.00%
   ```

---

### English-to-French Translation

1. **Prepare the Dataset**

   Place your English-French dataset in the `src/data/` folder. The dataset should be in a format compatible with the `DatasetProcessor` class.

2. **Train or Load the Model**

   Run the translation script to train or load the Transformer model:

   ```bash
   poetry run python src/translation_french_english.py
   ```

   - If a saved model exists, it will be loaded.
   - Otherwise, a new model will be trained and saved in the `src/models/` folder.

3. **Evaluate the Model**

   The script evaluates the model on the test dataset and calculates the BLEU score.

   Output:

   ```
   Test loss: 1.97, Test accuracy: 67.26%
   BLEU score on the test dataset: 0.52
   ```

---

## Customization

- Modify hyperparameters like `embed_dim`, `dense_dim`, and `num_heads` in `src/translation_french_english.py` for the Transformer model.
- Replace the dataset in `src/data/` with your own English-French dataset.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.