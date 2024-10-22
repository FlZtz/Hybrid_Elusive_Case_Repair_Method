# Hybrid Elusive Case Repair Method

## Overview

This code can be used to repair the elusive case in event logs, i.e. (missing) case IDs of an event log are determined 
using a transformer model.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Data Preparation (Optional)](#data-preparation-optional)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Creating Executable](#creating-executable)
- [File Descriptions](#file-descriptions)
- [Results](#results)
- [Dependencies](#dependencies)
- [License](#license)
- [References](#references)

## Introduction

This project involves training a Transformer model using `train.py` and then evaluating the results using 
`quality_metrics.ipynb`. Follow the steps below to get started.

## Requirements

Make sure to install the required dependencies before running the code:

```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation (Optional)

1. Download the event log dataset in XES format that you want to use for training and evaluation. You can use the 
   following datasets for testing:
   - Running Example (requests for compensation within an airline):
     https://pm4py.fit.fraunhofer.de/static/assets/examples/running-example.xes
   - Review Example Large (reviewing process of a paper for a journal):
     https://doi.org/10.4121/uuid:da6aafef-5a86-4769-acf3-04e8ae5ab4fe
   - Hospital Billing - Event Log (billing of medical services):
     https://doi.org/10.4121/uuid:76c46b83-c930-4798-a1c9-4be94dfeb741
   - Renting Log Low (rental process):
     https://doi.org/10.5281/zenodo.8059488
   - You can also use your own dataset.

2. Execute the `log_preparation.py` script to prepare the event log for training and evaluation.
   ```bash
   python log_preparation.py
   ```
   
3. (Optional) Run the `log_statistics.py` script to calculate statistics from the event log.
   ```bash
   python log_statistics.py
   ```

### Training

1. Execute the `train.py` script to train the model.
   ```bash
   python train.py
   ```

2. The script will prompt you to provide an event log in XES format as input for training. Please ensure that your 
   event log is formatted correctly and follows the specified structure.

3. The training process may take some time depending on the size of the dataset and the complexity of the model. Once 
   completed, the trained model will be saved for further evaluation.

### Evaluation

1. Open `quality_metrics.ipynb` using Jupyter Notebook or any compatible environment.

2. Run the cells in the notebook to evaluate the performance of the trained model on a dataset.

3. Analyse the evaluation results and make any necessary adjustments to improve the model's performance.

### Creating Executable

To create an executable (.exe) file for the `train.py` script, you can use pyinstaller. Follow these steps:

1. Install `pyinstaller` if you haven't already:
   ```bash
    pip install pyinstaller
   ```
   
2. Navigate to the project directory containing `train.py`.

3. Run the following command to create a standalone executable:
   ```bash
    pyinstaller --onefile -n HERE --distpath . train.py
   ```
   This will generate an executable named `HERE.exe` in the current directory.

## File Descriptions

- `config.py`: Configuration file for the model.
- `dataset.py`: Implementation of the dataset loader.
- `elusive_case_transformer.py`: Dummy implementation of the interaction with the system.
- `LICENSE.md`: License file (MIT).
- `log_preparation.py`: Script for preparing the event log dataset.
- `log_statistics.py`: Calculates statistics from XES event logs.
- `model.py`: Contains the implementation of the Transformer model.
- `prototype_interaction.ipynb`: Jupyter Notebook for dummy interaction with the system.
- `quality_metrics.ipynb`: Jupyter Notebook for evaluating the trained model.
- `README.md`: This file. Contains an overview of the project.
- `requirements.txt`: Lists the required Python packages.
- `train.py`: Script for training the Transformer model.

## Results

The results can be seen in the `quality_metrics.ipynb` notebook.

## Dependencies

Python 3.10 or higher

## License

See the [LICENSE](LICENSE.md) file for license rights and limitations (MIT).

## References

The training algorithm used in `train.py` is based on https://youtu.be/ISNdQcPhsts.
