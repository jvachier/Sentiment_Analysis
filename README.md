[![Linting: Ruff](https://img.shields.io/badge/linting-ruff-yellowgreen)](https://github.com/charliermarsh/ruff)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-red)](https://keras.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Sentiment Analysis and Translation

This repository provides a comprehensive solution for real-time **speech-to-text**, **sentiment analysis**, and **English-to-French translation** using state-of-the-art machine learning techniques. It includes an interactive web application and robust pipelines for text processing, sentiment classification, and language translation.

---

## Overview

![Application Workflow](docs/images/app_workflow.png)

*Figure: High-level workflow of the application, including speech-to-text, sentiment analysis, and translation.*


---

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
- **Integration with Speech-to-Text**: Translates recognized speech from English to French in real-time.

---

## Note on Models

The sentiment analysis and translation models included in this repository are **toy models** designed for demonstration purposes. They may not achieve production-level accuracy and are intended for educational and exploratory use.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Poetry for dependency management

### Install Dependencies
1. Install Poetry:
   ```bash
   pip install poetry
   ```
2. Install project dependencies:
   ```bash
   poetry install
   ```

### Download the Vosk Model
1. Download the `vosk-model-en-us-0.22` model from the [official Vosk repository](https://alphacephei.com/vosk/models).
2. Extract the `.zip` file into the project directory:
   ```bash
   unzip vosk-model-en-us-0.22.zip -d vosk-model-en-us-0.22
   ```
3. Ensure the extracted folder is located in the root directory:
   ```
   Sentiment_Analysis/
   ├── vosk-model-en-us-0.22/
   └── ...
   ```

---

## Required Datasets

### 1. Sentiment Analysis Dataset
- **Dataset**: [TripAdvisor Hotel Reviews Dataset (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews)
- **Description**: This dataset consists of 20,000 reviews crawled from TripAdvisor, allowing you to explore what makes a great hotel and potentially use this model in your travels.
- **Setup**:
   - Download the dataset from the link above.
   - Place the dataset in the `src/data/` directory.

### 2. English-to-French Translation Dataset
- **Dataset**: [English-French Dataset (Kaggle)](https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench)
- **Description**: This dataset contains English sentences paired with their French translations. It is used to train and evaluate the Transformer-based translation model.
- **Setup**:
  - Download the dataset from the link above.
  - Place the dataset in the `src/data/` directory as `en-fr.parquet`.

---

## Project Structure

```
Sentiment_Analysis/
├── app/                            # Application-specific files
│   ├── __init__.py                 # Makes the folder a Python package
│   └── voice_to_text_app.py        # Main application script
├── src/                            # Source folder
│   ├── data/                       # Dataset folder
│   ├── models/                     # Saved models
│   │   ├── inference_model.keras
│   │   ├── sentiment_keras_binary.keras
│   │   ├── transformer_best_model.keras
│   │   ├── optuna_model_binary.json
│   │   └── optuna_transformer_best_params.json 
│   ├── configurations/             # Configuration files
│   │   ├── model_builder_config.json
│   │   ├── model_trainer_config.json
│   │   └── optuna_config.json
│   ├── modules/                                   # Custom Python modules
│   │   ├── __init__.py                            # Makes the folder a Python package
│   │   ├── data_processor.py                      # Data loading and preprocessing utilities
│   │   ├── transformer_components.py              # Transformer model components
│   │   ├── speech_to_text.py                      # Speech-to-text and sentiment analysis logic
│   │   ├── text_vectorizer_sentiment_analysis.py  # Text vectorization for sentiment analysis
│   │   ├── load_data.py                           # Data loading utilities
│   │   ├── model_bert_other.py                    # BERT-based sentiment analysis model
│   │   ├── sentiment_analysis_utils.py            # Utilities for sentiment analysis
│   │   ├── optuna_transformer.py                  # Optuna-based hyperparameter optimization
│   │   ├── utils.py                               # Utility functions and enums
│   │   ├── mem_reduction.py                       # Memory optimization utilities
│   │   ├── data_preprocess_nltk.py                # NLTK-based text preprocessing
│   │   └── text_vectorizer.py                     # Text vectorization utilities
│   ├── scripts/                                # Scripts for dataset management and preprocessing
│   │   ├── __init__.py                         # Marks the directory as a Python package
│   │   ├── loading_kaggle_dataset_utils.py     # Utilities for downloading and optimizing Kaggle datasets
│   │   ├── loading_kaggle_dataset_script.py    # Script to process Kaggle datasets
│   │   └── README.md                           # Documentation for the scripts folder
│   ├── translation_french_english.py       # English-to-French translation pipeline
│   ├── sentiment_analysis_bert_other.py    # Sentiment analysis using BERT
│   └── sentiment_analysis.py               # Sentiment analysis pipeline script
├── tests/                          # Unit and integration tests
│   ├── test_data_processor.py      # Tests for data_processor.py
│   └── test_model.py               # Tests for speech_to_text.py
├── .github/                        # GitHub-specific files
│   ├── workflows/                  # GitHub Actions workflows
│   │   └── test.yaml               # Workflow for running tests
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

### Interactive Application
1. **Run the Application**:
   ```bash
   poetry run python app/voice_to_text_app.py
   ```
2. **Features**:
   - **Start Recording**: Begin recording your speech.
   - **Stop Recording**: Stop recording.
   - **Recognized Text**: Displays the text recognized from your speech.
   - **Translated Text**: Displays the English-to-French translation of the recognized text.
   - **Sentiment Analysis**: Displays the sentiment (positive or negative) of the recognized text.
   - **Download Recognized Text**: Provides a link to download the recognized text as a `.txt` file.

### Sentiment Analysis
1. **Train or Load the Model**:
   ```bash
   poetry run python src/sentiment_analysis.py
   ```
   - If a saved model exists, it will be loaded.
   - Otherwise, a new model will be trained and saved in the `src/models/` folder.
2. **Evaluate the Model**:
   The script evaluates the model on the test dataset:
   ```
   Test Accuracy: 95.00%
   ```

### English-to-French Translation
1. **Prepare the Dataset**:
   Place your English-French dataset in the `src/data/` folder.
2. **Train or Load the Model**:
   ```bash
   poetry run python src/translation_french_english.py
   ```
   - If a saved model exists, it will be loaded.
   - Otherwise, a new model will be trained and saved in the `src/models/` folder.
3. **Evaluate the Model**:
   The script evaluates the model on the test dataset and calculates the BLEU score:
   ```
   Test loss: 2.13, Test accuracy: 67.26%
   BLEU score on the test dataset: 0.52
   ```

---

## Customization

- Modify hyperparameters like `embed_dim`, `dense_dim`, and `num_heads` in `src/translation_french_english.py` for the Transformer model.
- Replace the dataset in `src/data/` with your own English-French dataset.

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---

## About

This repository is designed for researchers, developers, and enthusiasts interested in exploring advanced NLP techniques. It provides a practical implementation of speech-to-text, sentiment analysis, and translation pipelines, along with an interactive web application.

For questions or feedback, feel free to open an issue or contact the repository maintainers.