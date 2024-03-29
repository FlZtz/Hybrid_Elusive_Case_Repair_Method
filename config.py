# config.py - Configuration parameters and utility functions for model training.
import os
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

from dateutil import parser
import pandas as pd
import pm4py
from pm4py.algo.discovery.dfg import algorithm as dfg_algorithm
from pm4py.objects.conversion.log import converter as log_converter
import tkinter as tk
from tkinter import filedialog

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
cached_df_copy: Optional[pd.DataFrame] = None
expert_attributes: dict = {
    'Start Activity': {'type': 'unary', 'attribute': 'Activity'},
    'End Activity': {'type': 'unary', 'attribute': 'Activity'},
    'Directly Following': {'type': 'binary', 'attribute': 'Activity'}
}
expert_input_attributes: List[str] = []
expert_input_columns: List[str] = []
expert_input_values: dict = {}
input_config: Optional[Iterator[str]] = None
log: Optional[pd.DataFrame] = None
log_name: Optional[str] = None
log_path: Optional[str] = None
missing_placeholder: str = "[NONE]"
missing_placeholder_xes: str = ""
probability_threshold: Optional[float] = None
tf_input: List[str] = []


def attribute_specification() -> None:
    """
    Allows the user to specify Transformer input attributes based on provided attributes.
    """
    global input_config

    attributes = None
    if input_config is not None:
        attributes = next(input_config, None)

    if attributes is not None:
        print(f"Please enter the input attribute(s) for the transformer (separated by commas): {attributes}")
    else:
        attributes = input("Please enter the input attribute(s) for the transformer (separated by commas): ")

    # Check if user provided any attributes
    if not attributes:
        raise ValueError("No attributes provided. Please try again.")

    # Split input string into a list of attributes and strip extra whitespace
    attributes_list = [activity.strip() for activity in attributes.split(',')]

    # Set Transformer input attributes based on the provided attributes
    # Note: *attributes_list is used to unpack the list into separate arguments
    set_tf_input(*attributes_list)


def expert_value_query(attributes: list) -> dict:
    """
    Queries expert values for given attributes.

    :param attributes: A list of attribute names to query.
    :return: A dictionary containing attribute names as keys and dictionaries as values. Each value dictionary contains
     the expert values, the corresponding attribute name, and the occurrences of each value or combination.
    """
    global expert_attributes, input_config, log
    expert_values = {}

    for attribute in attributes:
        # Check if the attribute is in the expert_attributes dictionary
        if attribute in expert_attributes:
            attribute_type = expert_attributes[attribute]['type']
            attribute_name = expert_attributes[attribute]['attribute']

            # Initialize a dictionary to store expert values, attribute name, and occurrences
            value_dict = {'values': None, 'attribute': attribute_name, 'occurrences': []}

            if attribute_type == 'unary':
                suggestions = []

                if all(col in log.columns for col in ['concept:name', 'time:timestamp', 'case:concept:name']):
                    if not pd.api.types.is_datetime64_any_dtype(log['time:timestamp']):
                        log['time:timestamp'] = log[
                            'time:timestamp'
                        ].apply(lambda x: parser.isoparse(x) if isinstance(x, str) else x)
                        log['time:timestamp'] = pd.to_datetime(log['time:timestamp'], utc=True)

                    if attribute == 'Start Activity':
                        start_activities = pm4py.get_start_activities(log)
                        total_count = sum(start_activities.values())
                        suggestions = [
                            (activity, count / total_count)
                            for activity, count in sorted(start_activities.items(), key=lambda x: x[1], reverse=True)
                            [:min(3, len(start_activities))]
                        ]
                    elif attribute == 'End Activity':
                        end_activities = pm4py.get_end_activities(log)
                        total_count = sum(end_activities.values())
                        suggestions = [
                            (activity, count / total_count)
                            for activity, count in sorted(end_activities.items(), key=lambda x: x[1], reverse=True)
                            [:min(3, len(end_activities))]
                        ]

                suffix = 's' if len(suggestions) > 1 else ''
                suggestion_str = ', '.join(
                    [f"{activity} ({frequency * 100:.2f}%)" for activity, frequency in suggestions]
                )
                prompt_end = ' –\nSuggestion' + suffix + ': ' + suggestion_str + ':\n' if suggestions else ': '

                expert_value = None
                if input_config is not None:
                    expert_value = next(input_config, None)

                if expert_value is not None:
                    print(f"Please enter the value(s) that represent(s) the attribute '{attribute}' "
                          f"(separated by commas){prompt_end}{expert_value}")
                else:
                    expert_value = input(
                        f"Please enter the value(s) that represent(s) the attribute '{attribute}' "
                        f"(separated by commas){prompt_end}"
                    )

                if not expert_value:
                    # Raise an error if no value is provided
                    raise ValueError("No value provided. Please try again.")
                # Split the input values and strip any leading/trailing whitespace
                value_list = [value.strip() for value in expert_value.split(',')]

                # Prompt the user for occurrences for each value in value_list
                for value in value_list:
                    occurrence = None
                    if input_config is not None:
                        occurrence = next(input_config, None)
                        occurrence = occurrence.strip().lower() if occurrence is not None else None

                    if occurrence is not None:
                        print(f"Does '{value}' always or sometimes represent the attribute '{attribute}'?\n"
                              f"Enter 'always' or 'sometimes': {occurrence}")
                    else:
                        occurrence = input(f"Does '{value}' always or sometimes represent the attribute '{attribute}'?"
                                           f"\nEnter 'always' or 'sometimes': ").strip().lower()

                    if occurrence not in ['always', 'sometimes']:
                        raise ValueError("Invalid occurrence value. Please enter 'always' or 'sometimes'.")
                    value_dict['occurrences'].append(occurrence)

                value_dict['values'] = value_list
            elif attribute_type == 'binary':
                suggestions = []

                if attribute == 'Directly Following':
                    dfg = dfg_algorithm.apply(log, variant=dfg_algorithm.Variants.FREQUENCY)
                    total_count = sum(dfg.values())
                    suggestions = [
                        (activity_pair, count / total_count)
                        for activity_pair, count in sorted(dfg.items(), key=lambda x: x[1], reverse=True)
                        [:min(3, len(dfg))]
                    ]

                suffix = 's' if len(suggestions) > 1 else ''
                suggestions_str = ', '.join(
                    [f'{activity_pair[0]} + {activity_pair[1]} ({frequency * 100:.2f}%)'
                     for activity_pair, frequency in suggestions]
                )
                prompt_end = ' –\nSuggestion' + suffix + ': ' + suggestions_str + ':' if suggestions else ':'

                expert_value = None
                if input_config is not None:
                    expert_value = next(input_config, None)

                if expert_value is not None:
                    print(f"Please enter the combination(s) for the attribute '{attribute}' \n"
                          f"(values of a combination in the correct order connected with a plus sign and "
                          f"combinations separated by commas){prompt_end}\n{expert_value}")
                else:
                    expert_value = input(
                        f"Please enter the combination(s) for the attribute '{attribute}' \n"
                        f"(values of a combination in the correct order connected with a plus sign and "
                        f"combinations separated by commas){prompt_end}\n"
                    )

                if not expert_value:
                    # Raise an error if no value is provided
                    raise ValueError("No values provided. Please try again.")
                # Split the input values, map each combination, and strip any leading/trailing whitespace
                value_list = [list(map(str.strip, value.split('+'))) for value in expert_value.split(',')]

                # Prompt the user for occurrences for each combination in value_list
                for combination in value_list:
                    occurrence = None
                    if input_config is not None:
                        occurrence = next(input_config, None)
                        occurrence = occurrence.strip().lower() if occurrence is not None else None

                    if occurrence is not None:
                        print(f"Does the combination '{' + '.join(combination)}' always or sometimes "
                              f"represent the attribute '{attribute}'?\n"
                              f"Enter 'always' or 'sometimes': {occurrence}")
                    else:
                        occurrence = input(f"Does the combination '{' + '.join(combination)}' always or sometimes "
                                           f"represent the attribute '{attribute}'?\n"
                                           f"Enter 'always' or 'sometimes': ").strip().lower()

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


def extract_log_name(path: str) -> str:
    """
    Extract the log name from the given log path.

    :param path: File path to the event log.
    :return: Extracted log name.
    """
    # Extract log_name from log_path
    base_name = os.path.basename(path)
    name, _ = os.path.splitext(base_name)
    if name.startswith("determined_"):
        name = name.replace("determined_", "", 1)
    return name


def get_cached_df_copy() -> pd.DataFrame or None:
    """
    Get the cached DataFrame copy.

    :return: Cached DataFrame copy, or None if it does not exist.
    """
    global cached_df_copy
    return cached_df_copy


def get_config() -> dict:
    """
    Get configuration parameters for the model training.

    :return: Dictionary containing various configuration parameters.
    """
    global attribute_config, expert_input_attributes, expert_input_values, log_name, log_path, missing_placeholder
    global missing_placeholder_xes, tf_input

    # If log_path is not set, get it using the get_file_path function
    if log_path is None:
        log_path = get_file_path()

    # If log_name is not set, extract it from log_path
    if log_name is None:
        log_name = extract_log_name(log_path)

    if log is None:
        read_file(log_path)

    num_tokens = 2

    # Creating a dictionary to map attributes to their corresponding data mappings
    attribute_dictionary = {key: value['mapping'] for key, value in attribute_config.items()}

    if not tf_input:
        attribute_specification()

    if "Activity" not in tf_input:
        tf_input.append("Activity")

    # Initialize lists for discrete and continuous input attributes
    discrete_input_attributes = []
    continuous_input_attributes = []

    # Categorize input attributes as discrete or continuous
    for attr in tf_input:
        if attribute_config[attr]['property'] == "discrete":
            discrete_input_attributes.append(attr)
        else:
            continuous_input_attributes.append(attr)

    if not expert_input_attributes:
        expert_input_attributes = get_expert_attributes()
        expert_input_values = {}
        if expert_input_attributes:
            expert_input_values = expert_value_query(expert_input_attributes)

    # Determine the model name based on attribute lengths
    model_name = get_model_name(len(discrete_input_attributes), len(continuous_input_attributes),
                                len(expert_input_attributes))

    # Return a dictionary with configuration parameters
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10 ** -4,
        "eps": 1e-9,
        "seq_len": 10 + num_tokens,  # Adjusted seq_len accounting for SOS and EOS tokens
        "d_model": 512,
        "num_layers": 6,
        "num_heads": 8,
        "dropout": 0.1,
        "dff": 2048,
        "log_head": 10,
        "missing_placeholder": missing_placeholder,
        "missing_placeholder_xes": missing_placeholder_xes,
        "log": log,
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
        "config_file": f"model_configurations/{model_name}/{log_name}/config.pkl",
        "attribute_dictionary": attribute_dictionary
    }


def get_expert_attributes() -> List[str]:
    """
    Asks the user whether to add expert attributes and retrieves them if desired.

    :return: A list of expert attributes entered by the user.
    """
    global expert_attributes, input_config

    expert = None
    if input_config is not None:
        expert = next(input_config, None)
        expert = expert.strip().lower() if expert is not None else None

    if expert is not None and expert in ["yes", "no"]:
        print(f"Do you want to add one or more expert attributes? (yes/no): {expert}")
    else:
        expert = input("Do you want to add one or more expert attributes? (yes/no): ").strip().lower()

    if expert == "yes":
        if len(expert_attributes) == 1:
            suffix = ' is'
            attributes = list(expert_attributes.keys())[0]
        else:
            suffix = 's are'
            attributes = ', '.join(list(expert_attributes.keys())[:-1]) + f" and {list(expert_attributes.keys())[-1]}"

        attribute_input = None
        if input_config is not None:
            attribute_input = next(input_config, None)

        if attribute_input is not None:
            print(f"Following expert attribute{suffix} available: {attributes}.\n"
                  f"Please enter the attribute(s) for which you have expert knowledge "
                  f"(separated by commas): {attribute_input}")
        else:
            attribute_input = input(f"Following expert attribute{suffix} available: {attributes}.\n"
                                    f"Please enter the attribute(s) for which you have expert knowledge "
                                    f"(separated by commas): ")

        if not attribute_input:
            raise ValueError("No attribute(s) provided. Please try again.")
        attribute_list = [attribute.strip() for attribute in attribute_input.split(',')]
        attributes = input_validation(attribute_list, expert_attributes)
        return attributes

    elif not expert or expert == "no":
        return []

    else:
        raise ValueError("Invalid input. Please enter 'yes' or 'no'.")


def get_expert_input_columns() -> List[str]:
    """
    Get the expert input columns.

    :return: Expert input columns.
    """
    global expert_input_columns
    return expert_input_columns


def get_file_path(file_type: str = "event log") -> str:
    """
    Get the file path, either through GUI or manual input.

    :param file_type: Type of file to select. Defaults to "event log".
    :return: File path.
    """
    if "DISPLAY" in os.environ:
        # GUI components can be used
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        file_path = filedialog.askopenfilename(title=f"Select the file that contains the {file_type}")

        # Check if the user selected a file or canceled the dialog
        if file_path:
            return file_path
        else:
            raise ValueError("Error: No file selected.")
    else:
        # No display available, use alternative method (e.g., manual input)
        file_path = input(f"Enter the path to the file that contains the {file_type}: ")

        # Check if the user entered a file path
        if file_path:
            file_path = file_path.strip('"')
            return file_path
        else:
            raise ValueError("Error: No file selected.")


def get_input_config() -> Tuple[Optional[Iterator[str]], Optional[str]]:
    """
    Get the input configuration and the next value from that configuration.

    :return: A tuple containing the current input configuration iterator and the next value from the input iterator.
    """
    global input_config

    input_con = input_config
    next_value = None
    if input_con is not None:
        next_value = next(input_con, None)

    return input_con, next_value


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


def get_prob_threshold(config: dict) -> float:
    """
    Get the probability threshold for determining config['tf_output'].

    :param config: Configuration parameters.
    :return: Probability threshold for determining config['tf_output'].
    """
    global probability_threshold

    if probability_threshold is None:
        set_prob_threshold(config)

    return probability_threshold


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


def latest_weights_file_path(config: dict, use_second_latest: bool = False) -> str or None:
    """
    Get the file path for the latest or second-latest weights of the model.

    :param config: Configuration parameters.
    :param use_second_latest: Boolean flag to indicate whether to return the second-latest file. Defaults to False.
    :return: File path for the latest or second-latest weights of the model, or None if no weights are found.
    """
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))

    # Check if any weights files are found
    if len(weights_files) == 0:
        return None

    # Sort the list of weights files
    weights_files.sort()

    # Return the path of the latest or second-latest file based on the flag
    if use_second_latest:
        return str(weights_files[-2]) if len(weights_files) > 1 else None
    else:
        return str(weights_files[-1])


def read_file(path: str) -> None:
    """
    Read file into a DataFrame.

    :param path: File path.
    """
    global input_config, log, missing_placeholder, missing_placeholder_xes

    if path.endswith('.xes'):
        file = pm4py.read_xes(path)
        log = log_converter.apply(file, variant=log_converter.Variants.TO_DATA_FRAME)
        log.replace(missing_placeholder_xes, missing_placeholder, inplace=True)
        print("XES file successfully read.")
    elif path.endswith('.txt'):
        with open(path, 'r') as file:
            lines = file.readlines()
        input_values = [line.strip() for line in lines]
        input_config = (value for value in input_values)
        print("TXT file successfully read.")
    else:
        raise ValueError('Unknown file type. Supported types are .xes and .txt.')


def reset_log(new_process: bool = True) -> None:
    """
    Reset the global variables cached_df_copy, log, log_name, and log_path to None. Optionally reset
    expert_input_attributes, expert_input_columns, expert_input_values, input_config, probability_threshold and
    tf_input to empty lists or dictionaries if new_process is True.

    :param new_process: Specifies whether to reset expert_input_attributes, expert_input_columns, expert_input_values,
     input_config, probability_threshold and tf_input. Defaults to True.
    """
    global cached_df_copy, expert_input_attributes, expert_input_columns, expert_input_values, input_config, log
    global log_name, log_path, probability_threshold, tf_input

    cached_df_copy = None
    log = None
    log_name = None
    log_path = None
    if new_process:
        expert_input_attributes = []
        expert_input_columns = []
        expert_input_values = {}
        input_config = None
        probability_threshold = None
        tf_input = []


def reset_prob_threshold() -> None:
    """
    Reset the probability threshold to None.
    """
    global probability_threshold
    probability_threshold = None


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


def set_log_path(path: str) -> None:
    """
    Set the log path.

    :param path: File path to the event log.
    """
    global log_path
    log_path = path


def set_prob_threshold(config: dict) -> None:
    """
    Set the probability threshold for determining config['tf_output'].

    :param config: Configuration parameters.
    """
    global input_config, probability_threshold

    thresh = None
    if input_config is not None:
        thresh = next(input_config, None)
        thresh = thresh.strip() if thresh is not None else None

    if thresh is not None:
        print(f"Please enter the minimum probability (in %) with which the {config['tf_output']} must be "
              f"determined in order for it to be accepted: {thresh}")
    else:
        thresh = input(f"Please enter the minimum probability (in %) with which the {config['tf_output']} must be "
                       f"determined in order for it to be accepted: ").strip()

    if not thresh:
        raise ValueError("No threshold provided. Please try again.")

    if not thresh.lstrip('-').replace('.', '').isdigit() or thresh.count('.') > 1:
        raise ValueError("Invalid threshold. Please enter a number.")

    thresh = float(thresh)
    if thresh < 0 or thresh > 100:
        raise ValueError("Invalid threshold. Please enter a number between 0 and 100.")

    probability_threshold = thresh / 100


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
