# config.py - Configuration parameters and utility functions for model training.
import os
from pathlib import Path
from typing import List, Optional, Union, Tuple

import pandas as pd
import tkinter as tk
from tkinter import filedialog

log_path: Optional[str] = None
log_name: Optional[str] = None
expert_input_columns: List[str] = []
tf_input: List[str] = []

# Global variable to cache DataFrame copy
cached_df_copy: Optional[pd.DataFrame] = None

# Global variable to store attribute configuration
attribute_config: dict = {
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

# Global variable to store expert attributes
expert_attributes: dict = {
    'Start Activity': {'type': 'unary', 'attribute': 'Activity'},
    'End Activity': {'type': 'unary', 'attribute': 'Activity'},
    'Directly Following': {'type': 'binary', 'attribute': 'Activity'}
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

    if not tf_input:
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

    expert_input_attributes = get_expert_attributes()

    # Determine the model name based on attribute lengths
    model_name = get_model_name(len(discrete_input_attributes), len(continuous_input_attributes),
                                len(expert_input_attributes))

    expert_input_values = {}

    # Check if there are any expert input attributes
    if expert_input_attributes:
        # Query expert values for the expert input attributes
        expert_input_values = expert_value_query(expert_input_attributes)

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
        "expert_input_attributes": expert_input_attributes,
        "expert_input_values": expert_input_values,
        "model_folder": f"weights/{model_name}/{log_name}",
        "model_name": model_name,
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_folder": f"tokenizers/{model_name}/{log_name}",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": f"runs/{model_name}/{log_name}",
        "result_folder": f"repaired_logs/{model_name}/{log_name}",
        "result_csv_file": f"determined_{log_name}.csv",
        "result_xes_file": f"determined_{log_name}.xes",
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
    Reset the global variables log_path, log_name, expert_input_columns, tf_input, and cached_df_copy to None.
    """
    global log_path, log_name, expert_input_columns, tf_input, cached_df_copy
    log_path = None
    log_name = None
    expert_input_columns = []
    tf_input = []
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


def get_expert_input_columns() -> List[str]:
    """
    Get the expert input columns.

    :return: Expert input columns.
    """
    global expert_input_columns
    return expert_input_columns


def set_cached_df_copy(df: pd.DataFrame) -> None:
    """
    Set the cached DataFrame copy.

    :param df: DataFrame to cache.
    """
    global cached_df_copy
    cached_df_copy = df.copy()


def set_expert_input_columns(columns: List[str]) -> None:
    """
    Set the expert input columns.

    :param columns: List of expert input columns.
    """
    global expert_input_columns
    expert_input_columns = columns


def set_tf_input(*attributes: str) -> None:
    """
    Sets the Transformer input attributes based on the provided arguments.

    :param attributes: Variable number of string arguments representing the attributes to be set.
    """
    global tf_input, attribute_config

    # Unpack list if there is only one argument and it's a list
    if len(attributes) == 1 and isinstance(attributes[0], list):
        attributes = attributes[0]

    # If all attributes are valid, update tf_input
    tf_input = input_validation(attributes, attribute_config)


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


def get_expert_attributes() -> List[str]:
    """
    Asks the user whether to add expert attributes and retrieves them if desired.

    :return: A list of expert attributes entered by the user.
    """
    global expert_attributes

    expert = input("Do you want to add one or more expert attributes? (yes/no): ").strip().lower()

    if expert == "yes":
        attribute_input = input(f"Following expert attribute is available: {list(expert_attributes.keys())[0]}.\n"
                                if len(expert_attributes) == 1 else
                                f"Following expert attributes are available: "
                                f"{', '.join(list(expert_attributes.keys())[:-1])} and "
                                f"{list(expert_attributes.keys())[-1]}.\n"
                                f"Please enter the attribute(s) for which you have expert knowledge "
                                f"(separated by commas): ")
        if not attribute_input:
            raise ValueError("No attributes provided. Please try again.")
        attribute_list = [attribute.strip() for attribute in attribute_input.split(',')]
        attributes = input_validation(attribute_list, expert_attributes)
        return attributes

    elif not expert or expert == "no":
        return []

    else:
        raise ValueError("Invalid input. Please enter 'yes' or 'no'.")


def input_validation(attributes: Union[List[str], Tuple[str, ...]], dictionary: dict) -> List[str]:
    """
    Validates a list of attributes against a dictionary.

    :param attributes: List or tuple of attribute names to validate.
    :param dictionary: Dictionary to validate against.
    :return: List of validated attribute names.
    """
    # Initialize an empty list to store validated attributes
    validated_attributes = []

    # Iterate through each attribute in the provided list
    for attr in attributes:
        # Flag to track if the attribute is found in the dictionary
        found = False

        # Iterate through each key in the dictionary
        for key in dictionary:
            # Check if the attribute matches the key (case-insensitive comparison)
            if attr.lower() == key.lower():
                # If a match is found, add the key to the validated attributes list
                validated_attributes.append(key)
                found = True
                break

        # If the attribute is not found in the dictionary, raise a ValueError
        if not found:
            raise ValueError(f"Attribute '{attr}' is not a valid attribute.")

    # Return the list of validated attributes
    return validated_attributes


def get_model_name(discrete_len: int, continuous_len: int, expert_len: int) -> str:
    """
    Determine the model name based on the lengths of discrete, continuous, and expert attributes.

    :param discrete_len: The length of the list of discrete input attributes.
    :param continuous_len: The length of the list of continuous input attributes.
    :param expert_len: The length of the list of expert input attributes.
    :return: The model name corresponding to the given attribute lengths.
    """
    if discrete_len == 1 and continuous_len == 0 and expert_len == 0:
        return "baseline"
    elif discrete_len > 1 and continuous_len == 0 and expert_len == 0:
        return "multiple_discrete"
    elif discrete_len == 1 and continuous_len == 1 and expert_len == 0:
        return "continuous"
    elif discrete_len > 1 and continuous_len == 1 and expert_len == 0:
        return "multiple_discrete_single_continuous"
    elif discrete_len == 1 and continuous_len > 1 and expert_len == 0:
        return "multiple_continuous"
    elif discrete_len > 1 and continuous_len > 1 and expert_len == 0:
        return "multiple_discrete_continuous"
    elif discrete_len == 1 and continuous_len == 0 and expert_len == 1:
        return "expert"
    elif discrete_len > 1 and continuous_len == 0 and expert_len == 1:
        return "multiple_discrete_single_expert"
    elif discrete_len == 1 and continuous_len == 1 and expert_len == 1:
        return "continuous_expert"
    elif discrete_len > 1 and continuous_len == 1 and expert_len == 1:
        return "multiple_discrete_single_continuous_expert"
    elif discrete_len == 1 and continuous_len > 1 and expert_len == 1:
        return "multiple_continuous_single_expert"
    elif discrete_len > 1 and continuous_len > 1 and expert_len == 1:
        return "multiple_discrete_continuous_single_expert"
    elif discrete_len == 1 and continuous_len == 0 and expert_len > 1:
        return "multiple_expert"
    elif discrete_len > 1 and continuous_len == 0 and expert_len > 1:
        return "multiple_discrete_expert"
    elif discrete_len == 1 and continuous_len == 1 and expert_len > 1:
        return "continuous_multiple_expert"
    elif discrete_len > 1 and continuous_len == 1 and expert_len > 1:
        return "multiple_discrete_expert_single_continuous"
    elif discrete_len == 1 and continuous_len > 1 and expert_len > 1:
        return "multiple_continuous_expert"
    else:
        return "complete"


def expert_value_query(attributes: list) -> dict:
    """
    Queries expert values for given attributes.

    :param attributes: A list of attribute names to query.
    :return: A dictionary containing attribute names as keys and dictionaries as values. Each value dictionary contains
    the expert values, the corresponding attribute name, and the occurrences of each value or combination.
    """
    global expert_attributes
    expert_values = {}

    for attribute in attributes:
        # Check if the attribute is in the expert_attributes dictionary
        if attribute in expert_attributes:
            attribute_type = expert_attributes[attribute]['type']
            attribute_name = expert_attributes[attribute]['attribute']

            # Initialize a dictionary to store expert values, attribute name, and occurrences
            value_dict = {'values': None, 'attribute': attribute_name, 'occurrences': []}

            if attribute_type == 'unary':
                # Prompt the user for unary attribute values
                expert_value = input(f"Please enter the value(s) that represent(s) the attribute '{attribute}' "
                                     f"(separated by commas): ")
                if not expert_value:
                    # Raise an error if no value is provided
                    raise ValueError("No value provided. Please try again.")
                # Split the input values and strip any leading/trailing whitespace
                value_list = [value.strip() for value in expert_value.split(',')]

                # Prompt the user for occurrences for each value in value_list
                for value in value_list:
                    occurrence = input(f"Does '{value}' always or sometimes represent the attribute '{attribute}'? "
                                       f"Enter 'always' or 'sometimes': ").strip().lower()
                    if occurrence not in ['always', 'sometimes']:
                        raise ValueError("Invalid occurrence value. Please enter 'always' or 'sometimes'.")
                    value_dict['occurrences'].append(occurrence)

                value_dict['values'] = value_list
            elif attribute_type == 'binary':
                # Prompt the user for binary attribute values
                expert_value = input(f"Please enter the combination(s) for the attribute '{attribute}' \n"
                                     f"(values of a combination in the correct order connected with a plus sign and "
                                     f"combinations separated by commas): ")
                if not expert_value:
                    # Raise an error if no value is provided
                    raise ValueError("No value provided. Please try again.")
                # Split the input values, map each combination, and strip any leading/trailing whitespace
                value_list = [list(map(str.strip, value.split('+'))) for value in expert_value.split(',')]

                # Prompt the user for occurrences for each combination in value_list
                for combination in value_list:
                    occurrence = input(f"Does the combination '{' + '.join(combination)}' always or sometimes represent"
                                       f" the attribute '{attribute}'? Enter 'always' or 'sometimes': ").strip().lower()
                    if occurrence not in ['always', 'sometimes']:
                        raise ValueError("Invalid occurrence value. Please enter 'always' or 'sometimes'.")
                    value_dict['occurrences'].append(occurrence)

                value_dict['values'] = value_list
            else:
                # Raise an error if the attribute type is neither unary nor binary
                raise ValueError(f"Invalid attribute type '{attribute_type}' for attribute '{attribute}'.")

            # Store the value dictionary in the expert_values dictionary
            expert_values[attribute] = value_dict

        else:
            # Raise an error if the attribute is not found in expert_attributes
            raise ValueError(f"Attribute '{attribute}' not found in expert attributes.")

    return expert_values
