# Elusive Case Transformer

## Overview

This project involves training a Transformer model using `train.py` and then evaluating the results using `evaluation.ipynb`. Follow the steps below to get started.

## Instructions

### Step 1: Training

1. Execute the `train.py` script to train the model.
   ```bash
   python train.py
   ```

2. The script will prompt you to provide an event log in either XES or CSV format as input for training. Please ensure that your event log is formatted correctly and follows the specified structure.

3. The training process may take some time depending on the size of the dataset and the complexity of the model. Once completed, the trained model will be saved for further evaluation.

### Step 2: Evaluation

1. Open `evaluation.ipynb` using Jupyter Notebook or any compatible environment.

2. Run the cells in the notebook to evaluate the performance of the trained model on a test dataset.

3. Analyze the evaluation results and make any necessary adjustments to improve the model's performance.

## File Descriptions

- `train.py`: Script for training the model.
- `evaluation.ipynb`: Jupyter Notebook for evaluating the trained model.

## Dependencies

- Python 3.10 or higher

## License

This project is not licensed.

## Acknowledgements

- The training algorithm used in `train.py` is based on https://youtu.be/ISNdQcPhsts.
- The dataset used for training is sourced from https://pm4py.fit.fraunhofer.de/static/assets/examples/running-example.xes.