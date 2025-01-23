# Hybrid Elusive Case Repair Method

## Overview

This project provides a method to repair elusive cases in event logs by determining missing case IDs using a
Transformer model.

**Note:** This code is the instantiation source code of the journal paper titled "Case ID Revealed HERE: Hybrid Elusive
Case Repair Method for Transformer-Driven Business Process Event Log Enhancement" (_**LINK**_). Please refer to the
publication for detailed information.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
    - [Data Preparation (Optional)](#data-preparation-optional)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [Prototype Interaction](#prototype-interaction)
    - [Creating Executable](#creating-executable)
- [File Descriptions](#file-descriptions)
- [Dependencies](#dependencies)
- [License](#license)
- [References](#references)

## Introduction

This project involves training a Transformer model using `scripts/train.py` and evaluating the results using
`scripts/quality_metrics.py`. Follow the steps below to get started.

## Requirements

Install the required dependencies before running the code:

```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation (Optional)

1. Download the event log dataset in XES format for training and evaluation. You can use the following datasets for
   testing:
    - Running Example (requests for compensation within an airline):
      https://processintelligence.solutions/static/data/getting_started/running-example.xes
    - Review Example Large (reviewing process of a paper for a journal):
      https://doi.org/10.4121/uuid:da6aafef-5a86-4769-acf3-04e8ae5ab4fe
    - Hospital Billing - Event Log (billing of medical services):
      https://doi.org/10.4121/uuid:76c46b83-c930-4798-a1c9-4be94dfeb741
    - Renting Log Low (rental process):
      https://doi.org/10.5281/zenodo.8059488
    - You can also use your own dataset.

2. Execute the `scripts/log_preparation.py` script to prepare the event log for training and evaluation.
   ```bash
   python scripts/log_preparation.py
   ```
   **Hint**: Ensure your run configuration is set to the folder where the `scripts/log_preparation.py` file is located.

3. (Optional) Run the `scripts/log_statistics.py` script to calculate statistics from the event log.
   ```bash
   python scripts/log_statistics.py
   ```
   **Hint**: Ensure your run configuration is set to the folder where the `scripts/log_statistics.py` file is located.

### Training

1. Execute the `scripts/train.py` script to train the model.
   ```bash
   python scripts/train.py
   ```
   **Hint**: Ensure your run configuration is set to the folder where the `scripts/train.py` file is located.

2. Provide an event log in XES format as input for training. Ensure the event log is formatted correctly.

3. The training process may take some time depending on the dataset size and model complexity. Once completed, the
   trained model will be saved for further evaluation.

### Evaluation

1. Train benchmarks using the `scripts/lstm_repair.py` and `scripts/random_repair.py` scripts.
   ```bash
   python scripts/lstm_repair.py
   ```
    ```bash
   python scripts/random_repair.py
    ```
   **Hint**: Ensure your run configuration is set to the folder where the `scripts/lstm_repair.py` and
   `scripts/random_repair.py` file are located. These scripts are dummy implementations of the LSTM and Random repair
   methods, respectively, and need to be replaced with real implementations.

2. Execute the `scripts/quality_metrics.py` script to evaluate the performance of the trained model.
   ```bash
   python scripts/quality_metrics.py
   ```
   **Hint**: Ensure your run configuration is set to the folder where the `scripts/quality_metrics.py` file is located.
   Before running the script, make sure that relevant files are available in the corresponding directories. Also,
   ensure that the event logs are in CSV format and include a column named `Ground Truth Case ID` containing the
   correct case IDs for each event.

3. Analyse the evaluation results and make any necessary adjustments to improve the model's performance.

### Prototype Interaction

1. Ensure you have Jupyter Notebook installed. If not, install it using:
    ```bash
    pip install notebook
    ```

2. Navigate to the directory containing the `prototype_interaction.ipynb` notebook:
    ```bash
    cd path/to/your/notebook
    ```

3. Start the Jupyter Notebook server:
    ```bash
    jupyter notebook
    ```

4. Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:8888`).

5. In the Jupyter Notebook interface, open the `prototype_interaction.ipynb` notebook to interact with the system.

### Creating Executable

To create an executable (.exe) file for the `scripts/train.py` script, use pyinstaller:

1. Install `pyinstaller` if you haven't already:
   ```bash
    pip install pyinstaller
   ```

2. Navigate to the project directory containing `scripts/train.py`.

3. Run the following command to create a standalone executable:
   ```bash
    pyinstaller --onefile -n HERE --distpath . scripts/train.py
   ```
   This will generate an executable named `HERE.exe` in the current directory.

## File Descriptions

- `notebooks/`: Contains Jupyter Notebooks for interaction and evaluation.
    - `prototype_interaction.ipynb`: Dummy interaction with the system.
- `scripts/`: Contains Python scripts for data preparation, statistics, and training.
    - `log_preparation.py`: Script for preparing the event log dataset.
    - `log_statistics.py`: Calculates statistics from XES event logs.
    - `lstm_repair.py`: Dummy implementation of the LSTM repair method.
    - `quality_metrics.py`: Implementation of quality metrics for evaluation.
    - `random_repair.py`: Dummy implementation of the random repair methods.
    - `train.py`: Script for training the Transformer model.
- `src/`: Contains the source code files for the project.
    - `config.py`: Configuration file for the model.
    - `dataset.py`: Implementation of the dataset loader.
    - `hybrid_elusive_case_repair.py`: Dummy implementation of the interaction with the system.
    - `model.py`: Contains the implementation of the Transformer model.
- `LICENSE.md`: License file (MIT).
- `README.md`: This file. Contains an overview of the project.
- `requirements.txt`: Lists the required Python packages.

## Dependencies

- Python 3.10 or higher
- Windows operating system

## License

See the [LICENSE](LICENSE.md) file for license rights and limitations (MIT).

## References

The training algorithm used in `scripts/train.py` is based on https://youtu.be/ISNdQcPhsts.
