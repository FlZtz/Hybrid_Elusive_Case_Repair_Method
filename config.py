# config.py - Configuration parameters and utility functions for model training.
import os
from pathlib import Path
from typing import List, Union

import pandas as pd
import tkinter as tk
from tkinter import filedialog


# Global variables to store log_path and extracted log_name
log_path: Union[str, None] = None
log_name: Union[str, None] = None
tf_input: Union[List[str], None] = None

# Global variable to cache DataFrame copy
cached_df_copy: Union[pd.DataFrame, None] = None

# Global variable to store attribute configuration
attribute_config = {
    'Case ID': {'mapping': 'case:concept:name', 'property': 'discrete'},
    'Activity': {'mapping': 'concept:name', 'property': 'discrete'},
    'Timestamp': {'mapping': 'time:timestamp', 'property': 'continuous'},
    'Concept Definition': {'mapping': 'concept:definition', 'property': 'discrete'},
    'Instance': {'mapping': 'concept:instance', 'property': 'discrete'},
    'Lifecycle Transition': {'mapping': 'lifecycle:transition', 'property': 'discrete'},
    'Cost': {'mapping': 'org:cost', 'property': 'continuous'},
    'Cost Currency': {'mapping': 'org:cost:currency', 'property': 'discrete'},
    'Group': {'mapping': 'org:group', 'property': 'discrete'},
    'Resource': {'mapping': 'org:resource', 'property': 'discrete'},
    'Role': {'mapping': 'org:role', 'property': 'discrete'},
    'Role Label': {'mapping': 'org:role:label', 'property': 'discrete'},
    'Value': {'mapping': 'org:value', 'property': 'continuous'},
    'Value Currency': {'mapping': 'org:value:currency', 'property': 'discrete'}
}


def get_config() -> dict:
    """
    Get configuration parameters for the model training.

    :return: Dictionary containing various configuration parameters.
    """
    global log_path, log_name, tf_input, attribute_config

    # If log_path is not set, get it using the get_file_path function
    if log_path is None:
        log_path = get_file_path()

    # If log_name is not set, extract it from log_path
    if log_name is None:
        log_name = extract_log_name(log_path)

    num_tokens = 2

    # Creating a dictionary to map attributes to their corresponding data mappings
    attribute_dictionary = {key: value['mapping'] for key, value in attribute_config.items()}

    if tf_input is None:
        tf_input = ["Activity"]

    # Initialize lists for discrete and continuous input attributes
    discrete_input_attributes = []
    continuous_input_attributes = []

    # Categorize input attributes as discrete or continuous
    for attr in tf_input:
        if attribute_config[attr]['property'] == "discrete":
            discrete_input_attributes.append(attr)
        else:
            continuous_input_attributes.append(attr)

    # Assign model_name based on the number of discrete and continuous attributes
    if len(discrete_input_attributes) == 1 and len(continuous_input_attributes) == 0:
        model_name = "baseline"
    elif len(discrete_input_attributes) > 1 and len(continuous_input_attributes) == 0:
        model_name = "multiple_discrete"
    elif len(discrete_input_attributes) > 0 and len(continuous_input_attributes) > 0:
        model_name = "discrete_continuous"
    else:
        model_name = "unknown"

    # Return a dictionary with configuration parameters
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10 ** -4,
        "seq_len": 10 + num_tokens,  # Adjusted seq_len accounting for SOS and EOS tokens
        "d_model": 512,
        "log_path": log_path,
        "log_name": log_name,
        "tf_input": tf_input,
        "tf_output": "Case ID",
        "discrete_input_attributes": discrete_input_attributes,
        "continuous_input_attributes": continuous_input_attributes,
        "model_folder": f"weights/{model_name}/{log_name}",
        "model_name": model_name,
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_folder": f"tokenizers/{model_name}/{log_name}",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": f"runs/{model_name}/{log_name}",
        "result_folder": f"repaired_logs/{model_name}/{log_name}",
        "result_file": f"determined_{log_name}.csv",
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
    Reset the global variables log_path, log_name, tf_input, and cached_df_copy to None.
    """
    global log_path, log_name, tf_input, cached_df_copy
    log_path = None
    log_name = None
    tf_input = None
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


def set_tf_input(*attributes: str) -> None:
    """
    Sets the Transformer input attributes based on the provided arguments.

    :param attributes: Variable number of string arguments representing the attributes to be set.
    """
    global tf_input, attribute_config

    # Unpack list if there is only one argument and it's a list
    if len(attributes) == 1 and isinstance(attributes[0], list):
        attributes = attributes[0]

    # Validate input attributes
    validated_attributes = []
    for attr in attributes:
        found = False
        for key in attribute_config:
            if attr.lower() == key.lower():
                validated_attributes.append(key)  # Use the spelling from attribute_config
                found = True
                break
        if not found:
            raise ValueError(f"Attribute '{attr}' is not a valid attribute.")

    # If all attributes are valid, update tf_input
    tf_input = validated_attributes


def attribute_specification() -> None:
    """
    Allows the user to specify Transformer input attributes based on provided attributes.
    """
    # Prompt user to enter attributes separated by commas
    attributes = input("Please enter the input attribute(s) for the transformer (separated by commas): ")

    # Check if user provided any attributes
    if not attributes:
        raise ValueError("No attributes provided. Please try again.")

    # Split input string into a list of attributes and strip extra whitespace
    attributes_list = [activity.strip() for activity in attributes.split(',')]

    # Set Transformer input attributes based on the provided attributes
    # Note: *attributes_list is used to unpack the list into separate arguments
    set_tf_input(*attributes_list)
