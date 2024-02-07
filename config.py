# config.py - Configuration parameters and utility functions for model training.
import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from typing import Union

# Global variables to store log_path and extracted log_name
log_path: Union[str, None] = None
log_name: Union[str, None] = None

# Global variable to cache DataFrame copy
cached_df_copy: Union[pd.DataFrame, None] = None


def get_config() -> dict:
    """
    Get configuration parameters for the model training.

    :return: Dictionary containing various configuration parameters.
    """
    global log_path, log_name

    # If log_path is not set, get it using the get_file_path function
    if log_path is None:
        log_path = get_file_path()

    # If log_name is not set, extract it from log_path
    if log_name is None:
        log_name = extract_log_name(log_path)

    num_tokens = 2

    attribute_dictionary = {
        'Case ID': 'case:concept:name',
        'Activity': 'concept:name',
        'Timestamp': 'time:timestamp',
        'Resource': 'org:resource'
    }

    # Return a dictionary with configuration parameters
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10 ** -4,
        "seq_len": 10 + num_tokens,  # Adjusted seq_len accounting for SOS and EOS tokens
        "d_model": 512,
        "log_path": log_path,
        "log_name": log_name,
        "tf_input": "Activity",
        "tf_output": "Case ID",
        "model_folder": f"{log_name}_weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "{0}_tokenizer_{1}.json",
        "experiment_name": f"runs/{log_name}_tmodel",
        "result_file_name": f"determined_{log_name}.csv",
        "attribute_dictionary": attribute_dictionary
    }


def get_weights_file_path(config: dict, epoch: str) -> str:
    """
    Get the file path for the weights of the model at a specific epoch.

    :param config: Configuration parameters.
    :param epoch: Epoch number.
    :return: File path for the weights of the model at the specified epoch.
    """
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


def latest_weights_file_path(config: dict) -> str or None:
    """
    Get the file path for the latest weights of the model.

    :param config: Configuration parameters.
    :return: File path for the latest weights of the model, or None if no weights are found.
    """
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))

    # Check if any weights files are found
    if len(weights_files) == 0:
        return None

    # Sort the list of weights files and return the path of the latest one
    weights_files.sort()
    return str(weights_files[-1])


def reset_log() -> None:
    """
    Reset the global variables log_path, log_name, and cached_df_copy to None.
    """
    global log_path, log_name, cached_df_copy
    log_path = None
    log_name = None
    cached_df_copy = None


def get_file_path() -> str:
    """
    Get the file path for the event log, either through GUI or manual input.

    :return: File path for the event log.
    """
    if "DISPLAY" in os.environ:
        # GUI components can be used
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        file_path = filedialog.askopenfilename(title="Select the file that contains the event log")

        # Check if the user selected a file or canceled the dialog
        if file_path:
            return file_path
        else:
            raise ValueError("Error: No file selected.")
    else:
        # No display available, use alternative method (e.g., manual input)
        file_path = input("Enter the path to the file that contains the event log: ")

        # Check if the user entered a file path
        if file_path:
            file_path = file_path.strip('"')
            return file_path
        else:
            raise ValueError("Error: No file selected.")


def extract_log_name(log_path: str) -> str:
    """
    Extract the log_name from the given log_path.

    :param log_path: File path to the event log.
    :return: Extracted log_name.
    """
    # Extract log_name from log_path
    base_name = os.path.basename(log_path)
    log_name, _ = os.path.splitext(base_name)
    return log_name


def get_cached_df_copy() -> pd.DataFrame or None:
    """
    Get the cached DataFrame copy.

    :return: Cached DataFrame copy, or None if it does not exist.
    """
    global cached_df_copy
    return cached_df_copy


def set_cached_df_copy(df: pd.DataFrame) -> None:
    """
    Set the cached DataFrame copy.

    :param df: DataFrame to cache.
    """
    global cached_df_copy
    cached_df_copy = df.copy()
