# Elusive Case Transformer

## Overview

This code can be used to repair the elusive case in event logs, i.e. (missing) case IDs of an event log are determined 
using a transformer model.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
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
`evaluation.ipynb`. Follow the steps below to get started.

## Requirements

Make sure to install the required dependencies before running the code:

```bash
pip install -r requirements.txt
```

## Usage

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

1. Open `evaluation.ipynb` using Jupyter Notebook or any compatible environment.

2. Run the cells in the notebook to evaluate the performance of the trained model on a test dataset.

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
    pyinstaller --onefile -n ElusiveRepairer --distpath . train.py
   ```
   This will generate an executable named `ElusiveRepairer.exe` in the current directory.

## File Descriptions

- `model.py`: Contains the implementation of the Transformer model. 
- `requirements.txt`: Lists the required Python packages.
- `evaluation.ipynb`: Jupyter Notebook for evaluating the trained model. 
- `config.py`: Configuration file for the model. 
- `train.py`: Script for training the Transformer model. 
- `dataset.py`: Implementation of the dataset loader.

## Results

The results can be seen in the `evaluation.ipynb` notebook.

## Dependencies

- Python 3.10 or higher

## License

See the [LICENSE](LICENSE.md) file for license rights and limitations (MIT).

## References

- The training algorithm used in `train.py` is based on https://youtu.be/ISNdQcPhsts.
- The datasets used for training are publicly available and can be found at the following links:
  - Running Example: https://pm4py.fit.fraunhofer.de/static/assets/examples/running-example.xes
  - Review Example Large: https://doi.org/10.4121/uuid:da6aafef-5a86-4769-acf3-04e8ae5ab4fe