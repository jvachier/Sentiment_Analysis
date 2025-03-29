[![Linting: Ruff](https://img.shields.io/badge/linting-ruff-yellowgreen)](https://github.com/charliermarsh/ruff)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-red)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Sentiment Analysis

This repository contains a sentiment analysis project that uses TensorFlow and Keras to classify text data into positive or negative sentiments. The project includes data preprocessing, model training, evaluation, and inference. The model leverages a bidirectional LSTM layer for improved context understanding.

## Features

- **Text Preprocessing**: Uses TensorFlow's `TextVectorization` layer to tokenize and vectorize text data.
- **Bidirectional LSTM Model**: Implements a deep learning model with embedding, bidirectional LSTM, and dense layers for sentiment classification.
- **Training and Evaluation**: Includes functionality to train the model on a dataset and evaluate its performance on validation and test sets.
- **Inference**: Provides an inference pipeline to predict sentiment for new text inputs.
- **Dependency Management**: Uses Poetry for managing dependencies and virtual environments.

## Requirements

- Python 3.8 or higher
- TensorFlow 2.12 or higher
- Poetry for dependency management
- NumPy
- Transformers library (optional, for BERT integration)
- Optuna (optional, for hyperparameter tuning)

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
├── data/                     # Dataset folder
├── models/                   # Saved models
├── modules/                  # Custom modules
│   ├── load_data.py          # Data loading utilities
│   ├── model.py              # Model definition and training
│   ├── data_preprocess.py    # Data preprocessing utilities
├── sentiment_analysis.py     # Main script
├── README.md                 # Project documentation
├── pyproject.toml            # Poetry configuration file
```

## Usage

1. **Prepare the Dataset**
   
   Place your dataset in the `data/` folder. The default dataset used is `tripadvisor_hotel_reviews.csv`.

2. **Train the Model**

   Run the main script to train the model:


poetry run python sentiment_analysis.py


The script will preprocess the data, train the model, and save it in the `models/` folder.

3. **Inference**

The script includes a test example for inference. Modify the `raw_text_data` variable in `sentiment_analysis.py` to test with your own text input.

4. **Evaluate the Model**

The script evaluates the model on the test dataset and prints the accuracy.

Example Output:


Test Acc.: 95.00%


## Customization

- Modify hyperparameters like `embedding_dim`, `lstm_units`, and `dropout_rate` in `modules/model.py`.
- Replace the dataset in `data/` with your own CSV file.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.