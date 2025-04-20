# Scripts Directory

This directory contains scripts for managing datasets, preprocessing, and training models for the sentiment analysis project. Below is a description of the scripts and their purposes.

---

## **Files**

### 1. `__init__.py`
- Marks the directory as a Python package.
- Currently empty but can be used for package-level imports if needed.

---

### 2. `loading_kaggle_dataset_utils.py`
- Contains utility classes and functions for downloading, validating, and optimizing datasets from Kaggle.
- **Key Classes**:
  - **`KaggleDatasetConfig`**:
    - A Pydantic model to validate the configuration for Kaggle datasets.
    - Validates fields like `dataset_name`, `file_path`, and `output_path`.
  - **`KaggleDatasetLoader`**:
    - Handles the entire process of downloading, validating, optimizing, and saving Kaggle datasets.
    - Includes methods for:
      - Authenticating with Kaggle using the Kaggle API.
      - Validating required columns in the dataset.
      - Reducing memory usage for large datasets.
      - Saving the dataset as a Parquet file.

---

### 3. `loading_kaggle_dataset_script.py`
- A script to process a Kaggle dataset using the utilities in `loading_kaggle_dataset_utils.py`.
- **Workflow**:
  1. Defines the dataset configuration using `KaggleDatasetConfig`.
  2. Initializes the `KaggleDatasetLoader` with the configuration.
  3. Processes the dataset by:
     - Authenticating with Kaggle using the Kaggle API.
     - Downloading the dataset.
     - Validating required columns.
     - Optimizing memory usage.
     - Saving the dataset as a Parquet file.
- **Usage**:
  Run the script to process the dataset:
  ```bash
  python loading_kaggle_dataset_script.py
  ```