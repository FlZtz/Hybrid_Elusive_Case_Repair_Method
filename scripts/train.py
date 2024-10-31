# train.py - Script for training a Transformer model and creating a log with determined config['tf_output']
import os  # For filesystem operations
from pathlib import Path  # For working with file paths
import pickle  # For object serialization
import shutil  # For filesystem operations
import sys
from typing import Callable, List, Optional, Tuple  # For type hints
import warnings  # For managing warnings

# Third-party library imports
from datasets import Dataset as HuggingfaceDataset  # For Huggingface datasets
from dateutil import parser  # For parsing date strings
from deepdiff import DeepDiff  # For deep comparison of dictionaries
import matplotlib.pyplot as plt  # For plotting graphs
import pandas as pd  # For working with dataframes
import pm4py  # For process mining operations
from sklearn.model_selection import train_test_split  # For splitting dataset into train and
from sklearn.preprocessing import MinMaxScaler  # For normalizing continuous data
from tokenizers import models, pre_tokenizers, Tokenizer, trainers  # For tokenization
import torch  # PyTorch library
import torch.nn as nn  # PyTorch neural network module
import torchmetrics  # For evaluation metrics
from torch.utils.data import DataLoader, Dataset, random_split  # For working with PyTorch datasets
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard visualization

# Local application/library specific imports
from src.config import (add_response, extend_expert_input, get_cached_df_copy, get_config,
                        get_determination_probability, get_expert_input_columns, get_file_path, get_input_config,
                        get_original_values, get_prob_threshold, get_weights_file_path, latest_weights_file_path,
                        read_file, reset_log, reset_prob_threshold, save_responses, set_cached_df_copy,
                        set_determination_probability, set_expert_input_columns, set_log_path, set_original_values)
from src.dataset import causal_mask, InOutDataset  # Importing custom dataset and mask functions
from src.model import build_transformer, Transformer  # Importing Transformer model builder
from tqdm import tqdm  # For progress bars

sorted_index: Optional[pd.DataFrame] = None


def check_boundary_activity_rule(config: dict, log: pd.DataFrame, ex_post: bool, activity_type: str) -> pd.DataFrame:
    """
    Check the log for boundary activity rules specified in the configuration.

    :param config: Configuration parameters.
    :param log: Event log represented as a pandas DataFrame.
    :param ex_post: Flag for ex-post checking.
    :param activity_type: Type of activity to check for. Should be 'start' or 'end'.
    :return: Event log after rule checking as a pandas DataFrame.
    """
    attribute, values, occurrences, activity_column = validate_boundary_activity_input_data(config, log, activity_type)

    always_boundary_activities = []
    sometimes_boundary_activities = []

    for val, occurrence in zip(values, occurrences):
        if occurrence.lower() == 'always':
            always_boundary_activities.append(val)
        elif occurrence.lower() == 'sometimes':
            sometimes_boundary_activities.append(val)
        else:
            raise ValueError("Invalid occurrence value. Please enter 'always' or 'sometimes'.")

    log['original_index'] = log.index

    if ex_post:
        possible_boundary_activities = always_boundary_activities + sometimes_boundary_activities
        log = handle_boundary_activity_ex_post_processing(
            log,
            config,
            possible_boundary_activities,
            activity_type,
            activity_column,
            attribute
        )
    else:
        log = handle_boundary_activity_ex_ante_processing(
            log,
            config,
            always_boundary_activities,
            activity_type,
            activity_column,
            attribute
        )

    log.drop(columns='original_index', inplace=True)

    return log


def check_directly_following_rule(config: dict, log: pd.DataFrame, ex_post: bool) -> pd.DataFrame:
    """
    Check the log for directly following rules specified in the configuration.

    :param config: Configuration parameters.
    :param log: Event log represented as a pandas DataFrame.
    :param ex_post: Flag for ex-post checking.
    :return: Event log after rule checking as a pandas DataFrame.
    """
    validation = validate_directly_following_input_data(config, log)
    attribute = validation[0]
    values = validation[1]
    occurrences = validation[2]
    activity_column = validation[3]

    always_directly_following = [val for val, occurrence in zip(values, occurrences) if occurrence.lower() == 'always']

    log['original_index'] = log.index
    log[f"Determined {config['tf_output']}_str"] = log[f"Determined {config['tf_output']}"].astype(str)

    if ex_post:
        log = handle_directly_following_ex_post_processing(
            log,
            config,
            always_directly_following,
            activity_column,
            attribute
        )
    else:
        log = handle_directly_following_ex_ante_processing(
            log,
            config,
            always_directly_following,
            activity_column,
            attribute
        )

    log.drop(columns=['original_index', f"Determined {config['tf_output']}_str"], inplace=True)

    return log


def create_log(config: dict, chunk_size: int = None, repetition: bool = False, iteration: int = 1) -> pd.DataFrame:
    """
    Creates a log with determined config['tf_output'] based on the given configuration and chunk size.

    :param config: Dictionary containing configuration parameters for log creation. It includes keys like `tf_input`,
     `tf_output`, and `seq_len`.
    :param chunk_size: Number of rows to be processed as a single chunk. Default is set to `config['seq_len'] - 2`.
    :param repetition: Flag to indicate if the log creation is a repetition. Default is set to False.
    :param iteration: Iteration number. Default is set to 1.
    :return: DataFrame representing the log with determined config['tf_output'].
    """
    data_complete, ds_dataloader, vocab_src, vocab_tgt, device, consider_ids = prepare_log_creation_data(
        config,
        repetition,
        chunk_size
    )
    determined_log = process_log_creation_data(
        config,
        vocab_src,
        vocab_tgt,
        device,
        ds_dataloader,
        data_complete,
        consider_ids,
        repetition
    )
    save_created_logs(config, determined_log, repetition, iteration)
    return determined_log


def declarative_rule_checking(config: dict, log: pd.DataFrame, ex_post: bool = False) -> pd.DataFrame:
    """
    Check the log for declarative rules specified in the configuration.

    :param config: Configuration parameters including 'expert_input_attributes'.
    :param log: Event log represented as a pandas DataFrame.
    :param ex_post: Flag for ex-post checking. Defaults to False.
    :return: Event log after rule checking. If no rule checking is chosen, the original log is returned unchanged.

    """
    log['Modification'] = False

    if config['complete_expert_attributes']:
        examination = "ex post" if ex_post else "ex ante"
        rule_integration = get_user_choice(f"Please note that incorporating declarative rule checking may result in "
                                           f"assumption-based modifications.\nDo you want to proceed with "
                                           f"{examination} rule checking in this iteration? (yes/no): ")
        if rule_integration == 'no':
            return log

        if f"Determined {config['tf_output']}" in log.columns:
            if 'Directly Following' in config['complete_expert_attributes']:
                log = check_directly_following_rule(config, log, ex_post)
                print(f"Directly Following rule checking completed for {examination} analysis.")
            if 'Start Activity' in config['complete_expert_attributes']:
                log = check_boundary_activity_rule(config, log, ex_post, 'start')
                print(f"Start Activity rule checking completed for {examination} analysis.")
            if 'End Activity' in config['complete_expert_attributes']:
                log = check_boundary_activity_rule(config, log, ex_post, 'end')
                print(f"End Activity rule checking completed for {examination} analysis.")

    return log


def ex_ante_rule_checking(config: dict, log: pd.DataFrame, first_iteration: bool,
                          elusive_log: bool = True) -> pd.DataFrame:
    """
    Check the log ex-ante for declarative rules specified in the configuration.

    :param config: Configuration parameters.
    :param log: Event log represented as a pandas DataFrame.
    :param first_iteration: Flag to indicate if this is the first iteration.
    :param elusive_log: Flag to indicate if the log is elusive. Defaults to True.
    :return: Event log after rule checking as a pandas DataFrame.
    """
    log.rename(columns={f"{config['tf_output']}": f"Determined {config['tf_output']}"}, inplace=True)
    log[f"Determined {config['tf_output']}"].replace(config['missing_placeholder'], pd.NA, inplace=True)
    log[['Probability', 'Follow-up Probability']] = pd.NA
    if first_iteration:
        set_original_values(log[[f"Determined {config['tf_output']}"]]
                            .rename(columns={f"Determined {config['tf_output']}": f"Original {config['tf_output']}"}))
    if elusive_log or not first_iteration:
        log = declarative_rule_checking(config, log)
        set_determination_probability(log[['Probability', 'Follow-up Probability', 'Modification']])
        log.drop(columns='Modification', inplace=True)
    log.drop(columns=['Probability', 'Follow-up Probability'], inplace=True)
    log[f"Determined {config['tf_output']}"].replace(pd.NA, config['missing_placeholder'], inplace=True)
    log.rename(columns={f"Determined {config['tf_output']}": f"{config['tf_output']}"}, inplace=True)
    return log


def get_all_sequences(ds: Dataset, data: str) -> str:
    """
    Generator function to yield all sequences in a dataset.

    :param ds: Dataset to extract sequences from.
    :param data: Name of the data field.
    :return: Sequences from the dataset.
    """
    for item in ds:
        yield item[data]


def get_attribute_values(config: dict, dictionary: str, attribute: str) -> Tuple[List[str], str, List[str]]:
    """
    Get values, attribute column, and occurrences for a given attribute from the configuration.

    :param config: Configuration parameters.
    :param dictionary: Dictionary to extract values from.
    :param attribute: Attribute name.
    :return: Tuple containing values, attribute column, and occurrences.
    """
    values = config[dictionary][attribute]['values']
    attribute_column = config[dictionary][attribute]['attribute']
    occurrences = config[dictionary][attribute]['occurrences']

    return values, attribute_column, occurrences


def get_ds(config: dict, different_configuration: bool = False) -> Tuple[DataLoader, DataLoader, Tokenizer, Tokenizer]:
    """
    Gets training and validation DataLoaders along with tokenizers for the dataset.

    :param config: Configuration options.
    :param different_configuration: Flag to indicate if the configuration has changed. Defaults to False.
    :return: Training DataLoader; Validation DataLoader; Source Tokenizer; Target Tokenizer.
    """
    df = read_log(config)
    train, test = train_test_split(df, test_size=0.1, shuffle=True)
    ds_raw = HuggingfaceDataset.from_pandas(train)

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, 'Discrete Attributes', different_configuration)
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['tf_output'], different_configuration)

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    expert_input_columns = get_expert_input_columns()

    train_ds = InOutDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, 'Discrete Attributes', 'Continuous Attributes',
                            config['tf_output'], config['seq_len'],
                            len(config['discrete_input_attributes']) + len(expert_input_columns),
                            len(config['continuous_input_attributes']))
    val_ds = InOutDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, 'Discrete Attributes', 'Continuous Attributes',
                          config['tf_output'], config['seq_len'],
                          len(config['discrete_input_attributes']) + len(expert_input_columns),
                          len(config['continuous_input_attributes']))

    # Find the maximum length of each sequence in the source and target sequence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['Discrete Attributes']).ids
        tgt_ids = tokenizer_tgt.encode(item[config['tf_output']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sequence: {max_len_src}')
    print(f'Max length of target sequence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config: dict, vocab_src_len: int, vocab_tgt_len: int) -> Transformer:
    """
    Builds and returns the Transformer model.

    :param config: Configuration options.
    :param vocab_src_len: Length of source vocabulary.
    :param vocab_tgt_len: Length of target vocabulary.
    :return: Transformer model.
    """
    expert_input_columns = get_expert_input_columns()

    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        (config['seq_len'] - 2) * (len(config['discrete_input_attributes']) + len(expert_input_columns)) + 2,
        config['seq_len'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        h=config['num_heads'],
        dropout=config['dropout'],
        d_ff=config['dff']
    )

    return model


def get_or_build_tokenizer(config: dict, ds: Dataset, data: str, diff_config: bool = False) -> Tokenizer:
    """
    Get or build a tokenizer for a specific dataset and data field.

    :param config: Configuration options.
    :param ds: Dataset to build tokenizer for.
    :param data: Name of the data field.
    :param diff_config: Flag to indicate if the configuration has changed. Defaults to False.
    :return: Tokenizer for the dataset.
    """
    # Check if tokenizer file exists
    os.makedirs(config['tokenizer_folder'], exist_ok=True)
    tokenizer_path = Path(os.path.join(config['tokenizer_folder'], config['tokenizer_file'].format(data)))
    if not tokenizer_path.exists() or diff_config:
        # Train a new tokenizer
        tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
        # Customize pre-tokenization and training
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.WordLevelTrainer(special_tokens=[
            "[UNK]", "[PAD]", "[SOS]", "[EOS]", config['missing_placeholder']
        ])
        tokenizer.train_from_iterator(get_all_sequences(ds, data), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        # Load existing tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_response_configuration() -> bool:
    """
    Get response configuration file for model training if the user chooses to use it.

    :return: True if the user chooses to use a response configuration file, False otherwise.
    """
    response_configuration = get_user_choice("Do you want to use a specific response configuration file for model "
                                             "training? (yes/no): ")

    if response_configuration == 'yes':
        config_path = get_file_path("configuration")
        read_file(config_path, "configuration")
        return True

    return False


def get_user_choice(prompt: str) -> str:
    """
    Get user choice for the given prompt.

    :param prompt: Prompt message for the user.
    :return: User choice.
    """
    input_config, value = get_input_config()
    user_choice = None
    if input_config is not None:
        user_choice = value.strip().lower() if value is not None else None

    if user_choice is not None:
        print(f"{prompt}{user_choice}")
    else:
        user_choice = input(prompt).strip().lower()

    if not user_choice or user_choice not in ['yes', 'no']:
        raise ValueError("Invalid input! Please enter 'yes' or 'no'.")

    return user_choice


def greedy_decode(model: Transformer, source: torch.Tensor, source_mask: torch.Tensor, cont_input: torch.Tensor,
                  cont_mask: torch.Tensor, tokenizer_tgt: Tokenizer, max_len: int, device: torch.device,
                  config: dict, prob_threshold: float = 0) -> Tuple[torch.Tensor, List[float], List[float]]:
    """
    Greedy decoding algorithm for generating target sequences based on a trained Transformer model.

    :param model: The trained Transformer model.
    :param source: The source input tensor.
    :param source_mask: The mask for source sequence.
    :param cont_input: The continuous input tensor.
    :param cont_mask: The mask for the input sequence enriched with continuous data.
    :param tokenizer_tgt: Tokenizer for output.
    :param max_len: Maximum length of the output sequence.
    :param device: Device to perform computations on.
    :param config: Configuration options.
    :param prob_threshold: Probability threshold for selecting the next token. Defaults to 0.
    :return: The decoded target sequence tensor and the corresponding probabilities and follow-up probabilities.
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    none_idx = tokenizer_tgt.token_to_id(config['missing_placeholder'])

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)

    if config['continuous_input_attributes']:
        encoder_output = model.cont_enrichment(encoder_output, cont_input)
        source_mask = cont_mask

    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    probabilities = []
    follow_up_probabilities = []

    while True:
        if decoder_input.size(1) == max_len - 1:
            break

        # Build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        prob = torch.softmax(prob, dim=1)
        probs, indices = torch.topk(prob, k=3, dim=1)
        next_index = indices[:, 0]
        if next_index == eos_idx:
            if probs[:, 1].item() >= prob_threshold:
                next_output = indices[:, 1].item()
                probabilities.append(probs[:, 1].item())
                follow_up_probabilities.append(probs[:, 2].item())
            else:
                next_output = none_idx
                probabilities.append(0)
                follow_up_probabilities.append(0)
        else:
            if probs[:, 0].item() >= prob_threshold:
                next_output = next_index.item()
                probabilities.append(probs[:, 0].item())
                follow_up_probabilities.append(probs[:, 1].item())
            else:
                next_output = none_idx
                probabilities.append(0)
                follow_up_probabilities.append(0)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_output).to(device)], dim=1
        )

        if next_output == eos_idx:
            break

    return decoder_input.squeeze(0), probabilities, follow_up_probabilities


def handle_boundary_activity_ex_ante_processing(log: pd.DataFrame, config: dict, always_boundary_activities: List[str],
                                                activity_type: str, activity_column: str,
                                                attribute: str) -> pd.DataFrame:
    """
    Handle the ex-ante processing logic for boundary activity rule checking.

    For every always boundary activity, if there isn't an 'always' boundary activity present, we determine the
    index of the current boundary activity for each associated case. Subsequently, it is assessed which case aligns
    most closely with the boundary activity, considering the relevant role. If the time gap between the ongoing
    boundary activity and the particular boundary activity is shorter than the case's current lead time, the activity
    is allocated to that case.

    :param log: Event log represented as a pandas DataFrame.
    :param config: Configuration parameters.
    :param always_boundary_activities: List of always boundary activities.
    :param activity_type: Type of activity to check for ('start' or 'end').
    :param activity_column: Activity column name.
    :param attribute: Attribute name.
    :return: Modified log DataFrame.
    """
    set_count = 0

    current_ids = log[f"Determined {config['tf_output']}"].unique()
    current_ids = [ids for ids in current_ids if pd.notna(ids)]

    for activity in always_boundary_activities:
        elusive_indices = log.loc[
            (log[f"Determined {config['tf_output']}"].isna()) & (log[activity_column] == activity),
            'original_index'
        ].tolist()

        if elusive_indices:
            ids_with_boundary = log[log[activity_column] == activity][f"Determined {config['tf_output']}"].unique()
            ids_with_boundary = [ids for ids in ids_with_boundary if pd.notna(ids)]
            possible_ids = [ids for ids in current_ids if ids not in ids_with_boundary]

            if len(possible_ids) > 0:
                if activity_type == 'end':
                    current_boundary_activity_index = log[
                        log[f"Determined {config['tf_output']}"].isin(possible_ids)
                    ].groupby(f"Determined {config['tf_output']}")['original_index'].max().to_dict()
                else:
                    current_boundary_activity_index = log[
                        log[f"Determined {config['tf_output']}"].isin(possible_ids)
                    ].groupby(f"Determined {config['tf_output']}")['original_index'].min().to_dict()

                current_boundary_activity_index = dict(
                    sorted(current_boundary_activity_index.items(),
                           key=lambda item: item[1],
                           reverse=activity_type == 'end')
                )

                for idx in elusive_indices:
                    closest_index = None
                    corresponding_case = None
                    for case, bd_idx in current_boundary_activity_index.items():
                        if (activity_type == 'end' and idx < bd_idx) or (activity_type == 'start' and idx > bd_idx):
                            continue
                        closest_index = bd_idx
                        corresponding_case = case
                        break

                    if closest_index is not None:
                        filtered_log = log[log[f"Determined {config['tf_output']}"] == corresponding_case]
                        duration = filtered_log.iloc[-1]['Timestamp'] - filtered_log.iloc[0]['Timestamp']

                        if idx in log.index and closest_index in log.index:
                            if abs(log.loc[idx, 'Timestamp'] - log.loc[closest_index, 'Timestamp']) < duration:
                                log.loc[idx, [f"Determined {config['tf_output']}", "Probability",
                                              "Follow-up Probability", "Modification"]] = [
                                    corresponding_case, "Rule-based", pd.NA, True
                                ]
                                current_boundary_activity_index.pop(corresponding_case)
                                set_count += 1

    if set_count > 0:
        suffix = 's' if set_count > 1 else ''
        print(f"Set {set_count} determined {config['tf_output']} value{suffix} based on the input for '{attribute}'.")

    return log


def handle_boundary_activity_ex_post_processing(log: pd.DataFrame, config: dict,
                                                possible_boundary_activities: List[str], activity_type: str,
                                                activity_column: str, attribute: str) -> pd.DataFrame:
    """
    Handle the ex-post processing logic for boundary activity rule checking.

    If a potential boundary activity occurs within a case, then it is designated as the boundary of that case, and all
    preceding and subsequent determinations are cleared. If no potential boundary activity is identified within a case,
    all determinations for that case are reset, provided there are no remaining undefined case IDs in the log.

    :param log: Event log represented as a pandas DataFrame.
    :param config: Configuration parameters.
    :param possible_boundary_activities: List of possible boundary activities.
    :param activity_type: Type of activity to check for ('start' or 'end').
    :param activity_column: Activity column name.
    :param attribute: Attribute name.
    :return: Modified log DataFrame.
    """
    reset_count = 0
    elusive_log = log[f"Original {config['tf_output']}"].isna().any()

    log[f"Determined {config['tf_output']}_str"] = log[f"Determined {config['tf_output']}"].astype(str)
    modified_log = []

    grouped_log = log.groupby(f"Determined {config['tf_output']}_str")

    for case_id, group in grouped_log:
        if case_id == '<NA>':
            modified_log.append(group)
            continue

        group.reset_index(drop=True, inplace=True)

        boundary_activity_index = float('-inf') if activity_type == 'end' else float('inf')

        for activity in possible_boundary_activities:
            positions = group[group[activity_column] == activity].index.tolist()

            if positions:
                occurrence = positions[-1] if activity_type == 'end' else positions[0]

                if ((activity_type == 'end' and occurrence > boundary_activity_index) or
                        (activity_type == 'start' and occurrence < boundary_activity_index)):
                    boundary_activity_index = occurrence

        if boundary_activity_index != float('-inf') and boundary_activity_index != float('inf'):
            if ((activity_type == 'end' and boundary_activity_index != group.index[-1]) or
                    (activity_type == 'start' and boundary_activity_index != 0)):
                condition = group.loc[
                            boundary_activity_index + 1:
                            ] if activity_type == 'end' else group.loc[:boundary_activity_index - 1]
            else:
                condition = pd.DataFrame()
        else:
            condition = group if '<NA>' not in grouped_log.groups.keys() else pd.DataFrame()

        if elusive_log:
            column_name = f"Original {config['tf_output']}"
            condition = condition[condition[column_name].isna()] if column_name in condition.columns else condition

        group.loc[condition.index, [f"Determined {config['tf_output']}", "Probability", "Follow-up Probability",
                                    "Modification"]] = [pd.NA, pd.NA, pd.NA, True]
        reset_count += len(condition)
        modified_log.append(group)

    log = pd.concat(modified_log, ignore_index=True).sort_values(by='original_index').reset_index(drop=True)

    if reset_count > 0:
        suffix = 's' if reset_count > 1 else ''
        print(f"Reset {reset_count} determined {config['tf_output']} value{suffix} based on the input for "
              f"'{attribute}'.")

    log.drop(columns=f"Determined {config['tf_output']}_str", inplace=True)

    return log


def handle_directly_following_ex_ante_processing(log: pd.DataFrame, config: dict, always_directly_following: List[str],
                                                 activity_column: str, attribute: str) -> pd.DataFrame:
    """
    Handle the ex-ante processing logic for directly following rule checking.

    When analyzing each pair, both the preceding and succeeding elements are checked to see if either lacks a
    case ID. Cases missing the corresponding activity are then reviewed. The closest matching counterpart is
    identified, and its case ID is assigned to the activity if the time interval between the counterparts is shorter
    than the current lead time of the corresponding case.

    :param log: Event log represented as a pandas DataFrame.
    :param config: Configuration parameters.
    :param always_directly_following: List of always directly following activities.
    :param activity_column: Activity column name.
    :param attribute: Attribute name.
    :return: Modified log DataFrame.
    """
    set_count = 0

    predecessors = [pair[0] for pair in always_directly_following]
    successors = [pair[1] for pair in always_directly_following]

    for predecessor, successor in zip(predecessors, successors):

        elusive_indices_predecessor = log.loc[
            (log[f"Determined {config['tf_output']}"].isna()) & (log[activity_column] == predecessor),
            'original_index'
        ].tolist()

        elusive_indices_successor = log.loc[
            (log[f"Determined {config['tf_output']}"].isna()) & (log[activity_column] == successor),
            'original_index'
        ].tolist()

        for elusive_indices, label, other_label in [(elusive_indices_predecessor, predecessor, successor),
                                                    (elusive_indices_successor, successor, predecessor)]:

            if not elusive_indices:
                continue

            possible_ids = {}
            corresponding_indices = []
            grouped_log = log.groupby(f"Determined {config['tf_output']}_str")

            for case_id, group in grouped_log:
                if case_id == '<NA>':
                    continue
                positions_current = group[group[activity_column] == label].index.tolist()
                positions_other = group[group[activity_column] == other_label]['original_index'].tolist()

                if len(positions_current) < len(positions_other):

                    if case_id not in possible_ids:
                        difference = abs(len(positions_other) - len(positions_current))
                        time_diff = group.iloc[-1]['Timestamp'] - group.iloc[0]['Timestamp']
                        possible_ids[case_id] = {'difference': difference, 'indices': positions_other,
                                                 'time': time_diff}
                        corresponding_indices.extend(positions_other)

            if not possible_ids:
                continue

            corresponding_indices = sorted(corresponding_indices, reverse=(label == successor))

            for idx in elusive_indices:
                closest_index = next((corr_idx for corr_idx in corresponding_indices
                                      if (label == predecessor and idx < corr_idx) or
                                      (label == successor and idx > corr_idx)), None)

                if closest_index is not None:
                    corresponding_case = log.loc[closest_index, f"Determined {config['tf_output']}"]
                    if idx in log.index and closest_index in log.index:
                        if (abs(log.loc[closest_index, 'Timestamp'] - log.loc[idx, 'Timestamp']) <
                                possible_ids[corresponding_case]['time']):
                            log.loc[idx, [f"Determined {config['tf_output']}", "Probability", "Follow-up Probability",
                                          "Modification"]] = [corresponding_case, "Rule-based", pd.NA, True]
                            set_count += 1
                            corresponding_indices.remove(closest_index)

                            possible_ids[corresponding_case]['difference'] -= 1
                            if possible_ids[corresponding_case]['difference'] == 0:
                                corresponding_indices = [
                                    idx for idx in corresponding_indices
                                    if idx not in possible_ids[corresponding_case]['indices']
                                ]
                                corresponding_indices = sorted(corresponding_indices, reverse=(label == successor))

    if set_count > 0:
        suffix = 's' if set_count > 1 else ''
        print(f"Set {set_count} determined {config['tf_output']} value{suffix} based on the input for '{attribute}'.")

    return log


def handle_directly_following_ex_post_processing(log: pd.DataFrame, config: dict, always_directly_following: List[str],
                                                 activity_column: str, attribute: str) -> pd.DataFrame:
    """
    Handle the ex-post processing logic for directly following rule checking.

    For each 'always' relationship specified, it is checked whether the related activities exist in a given
    case. If neither activity is found and there are no undefined case IDs left in the log, all determinations related
    to that case are reset. Otherwise, to ensure an equal number of predecessors and successors, any excess activities
    are reset. If a case has an equal number of both, it is checked whether a 'follows' relationship is feasible by
    resetting potential successors located before the first predecessor or potential predecessors positioned after the
    last successor.

    :param log: Event log represented as a pandas DataFrame.
    :param config: Configuration parameters.
    :param always_directly_following: List of always directly following activities.
    :param activity_column: Activity column name.
    :param attribute: Attribute name.
    :return: Modified log DataFrame.
    """
    elusive_log = log[f"Original {config['tf_output']}"].isna().any()
    modified_log = log.copy()
    reset_count = 0

    for sublist in always_directly_following:
        grouped_log = modified_log.groupby(f"Determined {config['tf_output']}_str")
        temp_modified_log = []

        for case_id, group in grouped_log:
            if case_id == '<NA>':
                temp_modified_log.append(group)
                continue

            group.reset_index(drop=True, inplace=True)

            positions_predecessor = group[group[activity_column] == sublist[0]].index.tolist()
            positions_successor = group[group[activity_column] == sublist[1]].index.tolist()

            if not positions_predecessor and not positions_successor:
                if '<NA>' not in grouped_log.groups.keys():
                    condition = group
                    if elusive_log:
                        condition = group[group[f"Original {config['tf_output']}"].isna()]
                    group.loc[condition.index, [f"Determined {config['tf_output']}", "Probability",
                                                "Follow-up Probability", "Modification"]] = [pd.NA, pd.NA, pd.NA, True]
                    reset_count += len(condition)
                temp_modified_log.append(group)
                continue

            inadmissible_activities = abs(len(positions_predecessor) - len(positions_successor))

            if inadmissible_activities:
                if len(positions_predecessor) > len(positions_successor):
                    inadmissible_indices = positions_predecessor
                    if elusive_log:
                        inadmissible_indices = [idx for idx in positions_predecessor if
                                                pd.isna(group.loc[idx, f"Original {config['tf_output']}"])]
                        inadmissible_activities = min(inadmissible_activities, len(inadmissible_indices))
                    group.loc[inadmissible_indices[-inadmissible_activities:], [
                        f"Determined {config['tf_output']}", "Probability", "Follow-up Probability",
                        "Modification"]] = [pd.NA, pd.NA, pd.NA, True]
                    reset_count += inadmissible_activities
                    positions_predecessor = [idx for idx in positions_predecessor
                                             if idx not in inadmissible_indices[-inadmissible_activities:]]
                else:
                    inadmissible_indices = positions_successor
                    if elusive_log:
                        inadmissible_indices = [idx for idx in positions_successor if
                                                pd.isna(group.loc[idx, f"Original {config['tf_output']}"])]
                        inadmissible_activities = min(inadmissible_activities, len(inadmissible_indices))
                    group.loc[inadmissible_indices[:inadmissible_activities], [
                        f"Determined {config['tf_output']}", "Probability", "Follow-up Probability",
                        "Modification"]] = [pd.NA, pd.NA, pd.NA, True]
                    reset_count += inadmissible_activities
                    positions_successor = [idx for idx in positions_successor
                                           if idx not in inadmissible_indices[:inadmissible_activities]]

                inadmissible_activities = abs(len(positions_predecessor) - len(positions_successor))

            if not inadmissible_activities:
                first_predecessor, last_predecessor = positions_predecessor[0], positions_predecessor[-1]
                first_successor, last_successor = positions_successor[0], positions_successor[-1]

                if last_predecessor < first_successor:
                    temp_modified_log.append(group)
                    continue

                if first_predecessor > first_successor:
                    condition = group.loc[group.index.isin(positions_successor) & (group.index < first_predecessor)]
                    if elusive_log:
                        condition = condition.loc[condition[f"Original {config['tf_output']}"].isna()]
                    group.loc[condition.index, [f"Determined {config['tf_output']}", "Probability",
                                                "Follow-up Probability", "Modification"]] = [pd.NA, pd.NA, pd.NA, True]
                    reset_count += len(condition)

                if last_predecessor > last_successor:
                    condition = group.loc[group.index.isin(positions_predecessor) & (group.index > last_successor)]
                    if elusive_log:
                        condition = condition.loc[condition[f"Original {config['tf_output']}"].isna()]
                    group.loc[condition.index, [f"Determined {config['tf_output']}", "Probability",
                                                "Follow-up Probability", "Modification"]] = [pd.NA, pd.NA, pd.NA, True]
                    reset_count += len(condition)

            temp_modified_log.append(group)

        modified_log = pd.concat(temp_modified_log, ignore_index=True)

    log = modified_log.sort_values(by='original_index').reset_index(drop=True)

    if reset_count > 0:
        suffix = 's' if reset_count > 1 else ''
        print(f"Reset {reset_count} determined {config['tf_output']} value{suffix} based on the input for "
              f"'{attribute}'.")

    return log


def initialize_device() -> torch.device:
    """
    Initialize the device to use for training.

    :return: Device to use for training.
    """
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.has_mps or torch.backends.mps.is_available()
    else "cpu")
    print("Using device:", device)
    if device == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"Device memory: "
              f"{torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024 ** 3} GB")
    elif device == 'mps':
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print(
            "      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print(
            "      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url "
            "https://download.pytorch.org/whl/nightly/cpu")
    return torch.device(device)


def load_model_from_file(filename: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         device: torch.device) -> Tuple[Optional[int], Optional[int]]:
    """
    Load model and optimizer state from a file.

    :param filename: Path to the model file.
    :param model: Neural network model to load the state_dict into.
    :param optimizer: Optimizer to load the state_dict into.
    :param device: Device where the model will be loaded.
    :return: A tuple containing the loaded epoch and global_step. Returns (None, None) if an error occurs.
    """
    try:
        state = torch.load(filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        epoch = state['epoch'] + 1
        global_step = state['global_step']
        print(f'Successfully preloaded model from {filename}')
        return epoch, global_step
    except Exception as e:
        print(f'Error loading model from {filename}: {e}')
        return None, None


def load_or_initialize_model(config: dict, device: torch.device) -> Tuple[
    Transformer, int, int, DataLoader, DataLoader, Tokenizer, Tokenizer, torch.optim.Adam
]:
    """
    Load or initialize a Transformer model based on the provided configuration.

    :param config: Configuration parameters including model folder, configuration file, learning rate, etc.
    :param device: The device to which the model should be loaded (e.g., 'cpu' or 'cuda').
    :return: A tuple containing the Transformer model, initial epoch, global step, training and validation dataloaders,
     source and target tokenizers, and the Adam optimizer.
    """
    Path(f"{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    different_config = False
    model_config_path = os.path.dirname(config['config_file'])
    filtered_config = {k: v for k, v in config.items() if k not in ['log', 'log_path', 'attribute_dictionary']}

    if os.path.exists(model_config_path):
        with open(config['config_file'], 'rb') as file:
            loaded_config = pickle.load(file)

        loaded_config = {key: value.to_dict() if isinstance(value, pd.DataFrame) else value for key, value in
                         loaded_config.items()}
        curr_config = {key: value.to_dict() if isinstance(value, pd.DataFrame) else value for key, value in
                       filtered_config.items()}

        diff = DeepDiff(loaded_config, curr_config)

        if diff:
            different_config = True
            choice = input("\nAttention: The model's configuration has been modified since the last training.\nIf you "
                           "proceed, previously calculated weights will be deleted. Do you want to continue? "
                           "(yes/no): ").strip().lower()
            print("")

            if not choice or choice not in ['yes', 'no']:
                raise ValueError("Invalid input! Please enter 'yes' or 'no'.")

            if choice == 'no':
                sys.exit()

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config, different_config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=config['eps'])

    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = (latest_weights_file_path(config) if preload == 'latest'
                      else get_weights_file_path(config, preload) if preload
    else None)

    if different_config:
        model_filename = None

        # Delete all saved weights
        weights_folder = Path(f"{config['model_folder']}")
        for file in weights_folder.glob(f"{config['model_basename']}*.pt"):
            os.remove(file)

    os.makedirs(model_config_path, exist_ok=True)
    with open(config['config_file'], 'wb') as file:
        pickle.dump(filtered_config, file)

    if model_filename:
        print(f'Attempting to preload model from {model_filename}')
        epoch, g_step = load_model_from_file(model_filename, model, optimizer, device)
        if epoch is None:
            model_filename = latest_weights_file_path(config, use_second_latest=True)
            if model_filename:
                print(f'Attempting to preload model from {model_filename}')
                epoch, g_step = load_model_from_file(model_filename, model, optimizer, device)
        if epoch is None:
            print('No valid model to preload, starting from scratch')
        else:
            initial_epoch = epoch
            global_step = g_step
    else:
        print('No model to preload, starting from scratch')

    return model, initial_epoch, global_step, train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, optimizer


def normalize_continuous_attributes(df: pd.DataFrame, continuous_columns: List[str]) -> pd.DataFrame:
    """
    Normalize continuous attributes in the DataFrame using MinMaxScaler.

    :param df: DataFrame containing the log data.
    :param continuous_columns: List of column names containing continuous attributes.
    :return: DataFrame with normalized continuous attributes.
    """
    # Extract continuous columns from the dataframe
    continuous_df = df[continuous_columns]

    # Iterate over each column to handle potential string representations of numeric values
    for column in continuous_df.columns:
        # Convert string representations of numeric values to float
        continuous_df[column] = pd.to_numeric(continuous_df[column], errors='coerce')

    # Normalize numeric columns using MinMaxScaler
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(continuous_df)

    # Create new column names for the normalized columns
    normalized_column_names = [col + '_normalized' if '_normalized' not in col else col for col in continuous_columns]

    # Update the original dataframe with normalized columns
    df[normalized_column_names] = normalized_data

    return df


def prepare_dataframe_for_sequence_processing(df: pd.DataFrame, config: dict, chunk_size: int = None) -> pd.DataFrame:
    """
    Prepare the DataFrame for sequence processing.

    :param df: DataFrame containing the log data.
    :param config: Configuration object containing necessary parameters.
    :param chunk_size: Number of rows to be processed as a single chunk. Defaults to None.
    :return: Prepared DataFrame for sequence processing.
    """
    expert_input_columns = get_expert_input_columns()

    # Create a list with normalized column names for continuous attributes and original column names for others,
    # then extend with expert input columns
    all_input_columns = [
                            col + '_normalized' if col in config['continuous_input_attributes'] else col
                            for col in config['tf_input']
                        ] + expert_input_columns

    # Create a copy of DataFrame with required attributes
    df = df[[*all_input_columns, config['tf_output']]].copy()

    # Convert DataFrame to string type and replace spaces and hyphens with underscores
    df = df.astype(str).applymap(lambda x: x.replace(' ', '_').replace('-', '_'))

    # Prepare the DataFrame for sequence processing
    if chunk_size is None:
        length = config['seq_len'] - 2
        if len(df) >= length:
            for i in range(len(df) - length + 1):
                # Concatenate sequences of input and output attributes
                for attr in all_input_columns:
                    df.at[i, attr] = ' '.join(df[attr].iloc[i:i + length])
                df.at[i, config['tf_output']] = ' '.join(df[config['tf_output']].iloc[i:i + length])

            # Drop rows with insufficient sequence length
            df = df.drop(df.index[len(df) - length + 1:])
        else:
            raise ValueError(f"Length of the dataframe ({len(df)}) is less than {length}.")
    else:
        # Loop through the data in chunks and concatenate values for input and output features
        for i in range(len(df) // chunk_size):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size

            # Concatenate values for input feature(s)
            for attr in all_input_columns:
                df.at[start_idx, attr] = ' '.join(df[attr].iloc[start_idx:end_idx])
            # Concatenate values for output feature
            df.at[start_idx, config['tf_output']] = ' '.join(df[config['tf_output']].iloc[start_idx:end_idx])

        # Handle remaining rows if any
        remaining_rows = len(df) % chunk_size
        if remaining_rows > 0:
            start_idx = len(df) - remaining_rows
            # Concatenate values for input and output features for remaining rows
            for attr in all_input_columns:
                df.loc[start_idx:, attr] = ' '.join(df[attr].iloc[start_idx:])
            df.loc[start_idx:, config['tf_output']] = ' '.join(df[config['tf_output']].iloc[start_idx:])

        # Drop unnecessary rows to keep only one row per chunk
        df = df.iloc[::chunk_size].reset_index(drop=True)

    # Concatenate values from discrete input attributes and expert input columns into a single column
    df['Discrete Attributes'] = df[
        config['discrete_input_attributes'] + expert_input_columns
        ].apply(lambda row: ' '.join(row), axis=1)

    # Drop the original columns
    df.drop(config['discrete_input_attributes'] + expert_input_columns, axis=1, inplace=True)

    if config['continuous_input_attributes']:
        # Concatenate continuous attributes
        df['Continuous Attributes'] = df[
            [attr + '_normalized' for attr in config['continuous_input_attributes']]
        ].apply(lambda row: ' '.join(row), axis=1)
    else:
        df['Continuous Attributes'] = ''

    # Drop the original continuous attribute columns
    df.drop([attr + '_normalized' for attr in config['continuous_input_attributes']], axis=1, inplace=True)

    return df


def prepare_log_creation_data(config: dict, repetition: bool, chunk_size: int) -> Tuple[
    pd.DataFrame, DataLoader, Tokenizer, Tokenizer, torch.device, bool
]:
    """
    Prepare data for log creation based on the provided configuration.

    :param config: A dictionary containing configuration parameters.
    :param repetition: A boolean indicating whether the process is a repetition.
    :param chunk_size: An integer specifying the chunk size for sequence processing.
    :return: A tuple containing the complete data, DataLoader, source tokenizer, target tokenizer, torch device, and a
     boolean indicating whether IDs should be considered.
    """
    consider_ids = True

    if not repetition:
        prediction_preference = get_user_choice(f"Do you want to predict the {config['tf_output']} values using the "
                                                f"event log that was used for training? (yes/no): ")
        add_response(prediction_preference)
        if prediction_preference == 'yes':
            consider_ids = False

    if consider_ids:
        repaired_log_path = ""
        if repetition:
            repaired_log_path = os.path.join(config['result_folder'], config['result_xes_file'])
            if not os.path.exists(repaired_log_path):
                raise FileNotFoundError("The repaired log file does not exist. Please create it first.")

        reset_log(False)

        if repaired_log_path:
            set_log_path(repaired_log_path)
        else:
            print("Please ensure the new event log and its name match the process used during training.")

        config = get_config()

    data_complete = read_log(config, complete=True) if not repetition else (
        read_log(config, complete=True, first_iteration=False))

    if chunk_size is None:
        chunk_size = config['seq_len'] - 2

    df = prepare_dataframe_for_sequence_processing(data_complete, config, chunk_size)
    ds_raw = HuggingfaceDataset.from_pandas(df)

    vocab_src = get_or_build_tokenizer(config, ds_raw, 'Discrete Attributes')
    vocab_tgt = get_or_build_tokenizer(config, ds_raw, config['tf_output'])
    expert_input_columns = get_expert_input_columns()

    ds = InOutDataset(ds_raw, vocab_src, vocab_tgt, 'Discrete Attributes', 'Continuous Attributes',
                      config['tf_output'], config['seq_len'],
                      len(config['discrete_input_attributes']) + len(expert_input_columns),
                      len(config['continuous_input_attributes']))

    ds_dataloader = DataLoader(ds, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return data_complete, ds_dataloader, vocab_src, vocab_tgt, device, consider_ids


def process_expert_input_attributes(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Process expert input attributes and append them to the DataFrame.

    :param df: DataFrame containing the log data.
    :param config: Configuration object containing necessary parameters.
    :return: DataFrame with expert input attributes appended.
    """
    if not config['expert_input_attributes']:
        return df

    validate_config_and_columns(config, df)

    expert_input_columns = []
    for attribute in config['expert_input_attributes']:
        values, attribute_column, occurrences = get_attribute_values(config, 'expert_input_values', attribute)

        if isinstance(values, list) and all(isinstance(val, str) for val in values):
            df, columns = process_one_dimensional_values(df, attribute, attribute_column, values, occurrences)
        elif isinstance(values, list) and all(isinstance(val, str) for sublist in values for val in sublist):
            df, columns = process_two_dimensional_values(df, attribute, attribute_column, values, occurrences)
        else:
            raise ValueError(
                "Unsupported format for values. Please provide either a one-dimensional or two-dimensional list.")

        expert_input_columns.extend(columns)

    set_expert_input_columns(expert_input_columns)

    return df


def process_log_creation_data(config: dict, vocab_src: Tokenizer, vocab_tgt: Tokenizer, device: torch.device,
                              ds_dataloader: DataLoader, data_complete: pd.DataFrame, consider_ids: bool,
                              repetition: bool) -> pd.DataFrame:
    """
    Process the log creation data based on the provided configurations and models.

    :param config: Configuration dictionary containing model and data parameters.
    :param vocab_src: Tokenizer for the source vocabulary.
    :param vocab_tgt: Tokenizer for the target vocabulary.
    :param device: Device where the model will be loaded.
    :param ds_dataloader: DataLoader containing the dataset for determination.
    :param data_complete: Complete DataFrame containing the data attributes.
    :param consider_ids: Boolean flag to consider previous IDs for determination.
    :param repetition: Boolean flag to indicate if the process is a repetition.
    :return: DataFrame containing the processed log creation data.
    """
    global sorted_index

    model = get_model(config, vocab_src.get_vocab_size(), vocab_tgt.get_vocab_size()).to(device)
    model_filename = get_weights_file_path(config, f"{config['num_epochs'] - 1}")
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])

    determined_log = []
    prob_threshold = get_prob_threshold(config)

    for batch in tqdm(ds_dataloader, desc=f"Determining {config['tf_output']} Values"):
        encoder_input = batch["encoder_input"].to(device)
        encoder_mask = batch["encoder_mask"].to(device)
        continuous_input = batch["continuous_input"].to(device)
        continuous_mask = batch["continuous_mask"].to(device)
        target_text = batch["tgt_text"][0].split()
        batch_size = len(target_text)

        try:
            assert encoder_input.size(0) == 1, "Batch size must be 1 for evaluation"
            model_out, probabilities, follow_up_probabilities = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                continuous_input,
                continuous_mask,
                vocab_tgt,
                batch_size + 2,
                device,
                config,
                prob_threshold
            )

            model_out_values = vocab_tgt.decode(model_out.detach().cpu().numpy(), False).split()
            model_out_values = model_out_values[1:]

            determined_log.extend((model_out_value, f"{prob * 100:.2f}%", f"{follow_prob * 100:.2f}%", target_value)
                                  for target_value, model_out_value, prob, follow_prob
                                  in zip(target_text, model_out_values, probabilities, follow_up_probabilities))
        except Exception as e:
            print(f"Error occurred during decoding: {e}")
            continue

    determined_log = pd.DataFrame(
        determined_log,
        columns=[
            f"Determined {config['tf_output']}",
            "Probability",
            "Follow-up Probability",
            f"Previous {config['tf_output']}"
        ]
    )

    determined_log.replace(config['missing_placeholder'], pd.NA, inplace=True)

    columns_to_mask = ['Probability', 'Follow-up Probability']
    for col in columns_to_mask:
        determined_log[col] = determined_log[col].mask(
            determined_log[f"Determined {config['tf_output']}"].isna(), pd.NA
        )

    if consider_ids:
        for idx, row in determined_log.iterrows():
            if pd.notna(row[f"Previous {config['tf_output']}"]):
                determined_log.at[idx, f"Determined {config['tf_output']}"] = row[f"Previous {config['tf_output']}"]
                determined_log.loc[idx, ['Probability', 'Follow-up Probability']] = pd.NA

    input_attributes = config['tf_input'].copy()

    if 'Timestamp' not in input_attributes:
        input_attributes.append('Timestamp')

    restored_columns = pd.DataFrame()
    for attr in input_attributes:
        attr_column = pd.DataFrame(data_complete[attr])
        restored_columns = pd.concat([restored_columns, attr_column], axis=1)

    original_values = get_original_values()
    determined_log = pd.concat([determined_log, original_values, restored_columns], axis=1)

    if not repetition:
        determined_log.drop(columns=[f"Previous {config['tf_output']}"], inplace=True)

    determined_log = declarative_rule_checking(config, determined_log, True)

    set_determination_probability(determined_log[['Probability', 'Follow-up Probability', 'Modification']])
    determined_log.drop(columns='Modification', inplace=True)

    determination_probability = get_determination_probability()
    determination_probability.rename(
        columns={'Probability': 'Determination Probability',
                 'Follow-up Probability': 'Determination Follow-up Probability'},
        inplace=True
    )

    original_col_index = determined_log.columns.get_loc(f"Original {config['tf_output']}")
    left_cols = determined_log.columns[:original_col_index]
    right_cols = determined_log.columns[original_col_index:]

    determined_log = pd.concat([determined_log[left_cols], determination_probability, determined_log[right_cols]],
                               axis=1)

    determined_log.rename(
        columns={'Probability': 'Iteration Probability', 'Follow-up Probability': 'Iteration Follow-up Probability'},
        inplace=True
    )

    determined_log = remove_redundant_columns(determined_log)

    if 'Sorted Index' not in determined_log.columns:
        determined_log = pd.concat([determined_log, sorted_index], axis=1)

    return determined_log


def process_one_dimensional_values(df: pd.DataFrame, attribute: str, attribute_column: str, values: List[str],
                                   occurrences: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Process one-dimensional values and append them to the DataFrame.

    :param df: DataFrame containing the log data.
    :param attribute: Attribute to process.
    :param attribute_column: Column in the DataFrame corresponding to the attribute.
    :param values: Values associated with the attribute.
    :param occurrences: Occurrences associated with the values.
    :return: DataFrame with processed one-dimensional values appended, expert_input_columns
    """
    attribute_name = attribute.lower().replace(' ', '_')
    transformed_attribute_column = df.columns[df.columns.str.lower() == attribute_column.lower()][0]

    for val in values:
        if not df[transformed_attribute_column].str.lower().str.contains(val.lower()).any():
            raise ValueError(f"The value '{val}' specified for attribute '{attribute}' does not "
                             f"exist in the DataFrame column '{transformed_attribute_column}'.")

    df[attribute] = f'expert_non_{attribute_name}'
    for val, occurrence in zip(values, occurrences):
        if occurrence.lower() == 'always':
            df.loc[df[transformed_attribute_column].str.lower().isin(
                [val.lower()]), attribute] = f'expert_always_{attribute_name}'
        elif occurrence.lower() == 'sometimes':
            df.loc[df[transformed_attribute_column].str.lower().isin(
                [val.lower()]), attribute] = f'expert_sometimes_{attribute_name}'
        else:
            raise ValueError("Invalid occurrence value. Please enter 'always' or 'sometimes'.")

    return df, [attribute]


def process_two_dimensional_values(df: pd.DataFrame, attribute: str, attribute_column: str, values: List[str],
                                   occurrences: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Process two-dimensional values and append them to the DataFrame.

    :param df: DataFrame containing the log data.
    :param attribute: Attribute to process.
    :param attribute_column: Column in the DataFrame corresponding to the attribute.
    :param values: Values associated with the attribute.
    :param occurrences: Occurrences associated with the values.
    :return: DataFrame with processed two-dimensional values appended, expert_input_columns
    """
    attribute_name = attribute.lower().replace(' ', '_')
    transformed_attribute_column = df.columns[df.columns.str.lower() == attribute_column.lower()][0]

    for sublist in values:
        for val in sublist:
            if not df[transformed_attribute_column].str.lower().str.contains(val.lower()).any():
                raise ValueError(f"The value '{val}' specified for attribute '{attribute}' "
                                 f"does not exist in the DataFrame column '{transformed_attribute_column}'.")

    first_values_list = list(set(sublist[0].lower() for sublist in values))
    expert_input_columns = []

    for value in first_values_list:
        filtered_values = [val for val in values if val[0].lower() == value]
        filtered_occurrences = [occurrence for val, occurrence in zip(values, occurrences) if val[0].lower() == value]

        transformed_value = (df[df[transformed_attribute_column].str.lower() == value.lower()]
                             [transformed_attribute_column].iloc[0].replace(' ', '_'))

        df[f'{attribute}_{transformed_value}'] = f'expert_non_{attribute_name}_{transformed_value}'
        expert_input_columns.extend([f'{attribute}_{transformed_value}'])

        for val, occurrence in zip(filtered_values, filtered_occurrences):
            if occurrence.lower() == 'always':
                df.loc[df[transformed_attribute_column].str.lower().isin([val[1].lower()]),
                f'{attribute}_{transformed_value}'] = f'expert_always_{attribute_name}_{transformed_value}'
            elif occurrence.lower() == 'sometimes':
                df.loc[df[transformed_attribute_column].str.lower().isin([val[1].lower()]),
                f'{attribute}_{transformed_value}'] = f'expert_sometimes_{attribute_name}_{transformed_value}'
            else:
                raise ValueError("Invalid occurrence value. Please enter 'always' or 'sometimes'.")

    return df, expert_input_columns


def read_log(config: dict, complete: bool = False, first_iteration: bool = True) -> pd.DataFrame:
    """
    Read log file and preprocess data according to configuration.

    :param config: Configuration object containing necessary parameters.
    :param complete: Flag to indicate if the complete DataFrame should be returned. Defaults to False.
    :param first_iteration: Flag to indicate if it's the first iteration. Defaults to True.
    :return: Processed DataFrame according to the specified configuration.
    """
    global sorted_index

    cached_df_copy = get_cached_df_copy()
    if cached_df_copy is not None and complete:
        cached_df_copy = ex_ante_rule_checking(config, cached_df_copy, first_iteration, False)
        return cached_df_copy

    df = config['log']

    # Define the required attributes including 'Timestamp' only if it's not already in config['tf_input']
    # or config['tf_output']
    required_attributes = list({*config['tf_input'], config['tf_output'], 'Timestamp'})

    # Create a mapping of attribute aliases to column names
    column_mapping = {key: value for key, value in config['attribute_dictionary'].items()
                      if key in required_attributes}

    df = select_columns(df, column_mapping, first_iteration)

    print("Input processing. Please wait ...")

    # Convert Timestamp column to datetime
    df['Timestamp'] = df['Timestamp'].apply(lambda x: parser.isoparse(x) if isinstance(x, str) else x)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
    df = df.sort_values(['Timestamp', 'Sorted Index']).reset_index(drop=True)
    df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)
    sorted_index = df['Sorted Index'].copy()
    df.drop(columns='Sorted Index', inplace=True)

    if config['continuous_input_attributes']:
        continuous_columns = config['continuous_input_attributes'][:]

        # Check if 'Timestamp' is in continuous columns
        if 'Timestamp' in continuous_columns:
            # Calculate the time differences in seconds relative to the first timestamp
            df['Timestamp_normalized'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds()

            continuous_columns.append('Timestamp_normalized')
            continuous_columns.remove('Timestamp')

        df = normalize_continuous_attributes(df, continuous_columns)

    df = process_expert_input_attributes(df, config)

    # Cache a copy of the DataFrame for potential future use
    set_cached_df_copy(df)

    # If complete flag is set, return the DataFrame
    if complete:
        df = ex_ante_rule_checking(config, df, first_iteration)
        return df

    os.makedirs(config['log_folder'], exist_ok=True)
    df.to_csv(os.path.join(config['log_folder'], config['log_file']), index=False)

    df = prepare_dataframe_for_sequence_processing(df, config)

    # Return the prepared DataFrame
    return df


def remove_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove redundant columns from a DataFrame based on specific criteria.

    :param df: Input DataFrame.
    :return: DataFrame with redundant columns removed.
    """
    it_prob_column = df[df['Iteration Probability'].notna()]['Iteration Probability']
    det_prob_column = df[df['Determination Probability'].notna()]['Determination Probability']

    if (len(it_prob_column) == len(det_prob_column) and (it_prob_column == det_prob_column).all()
            and (it_prob_column.index == det_prob_column.index).all()):
        modified_log = df.drop(columns=['Iteration Probability'])
        df = modified_log

    it_follow_up_column = df[df['Iteration Follow-up Probability'].notna()]['Iteration Follow-up Probability']
    det_follow_up_column = df[df['Determination Follow-up Probability'].notna()]['Determination Follow-up Probability']

    if (len(it_follow_up_column) == len(det_follow_up_column) and (it_follow_up_column == det_follow_up_column).all()
            and (it_follow_up_column.index == det_follow_up_column.index).all()):
        modified_log = df.drop(columns=['Iteration Follow-up Probability'])
        df = modified_log

    return df


def repair_loop(log: pd.DataFrame, config: dict) -> None:
    """
    Repair loop for the determined log.

    :param log: Log with determined config['tf_output'] values.
    :param config: Configuration object containing necessary parameters.
    """
    percentage_na = (
                            log[f"Original {config['tf_output']}"]
                            .isna().
                            sum()
                            / len(log[f"Original {config['tf_output']}"])
                    ) * 100
    print(f"\nFor {percentage_na:.2f}% of the events, the {config['tf_output']} has originally not been recorded.")

    iteration = 1

    while True:
        print(f"\nRepair Iteration {iteration}:\n")

        num_rows = config['log_head']
        print(log.head(num_rows))

        remaining_rows = len(log) - num_rows
        if remaining_rows > 0:
            print(f"\n... (+ {remaining_rows} more row{'s' if remaining_rows > 1 else ''})")

        percentage_na = (
                                log[f"Determined {config['tf_output']}"]
                                .isna().
                                sum()
                                / len(log[f"Determined {config['tf_output']}"])
                        ) * 100

        if percentage_na == 0:
            print(f"\nFor all events, the {config['tf_output']} has been determined.")
            break
        else:
            print(f"\nFor {percentage_na:.2f}% of the events, the {config['tf_output']} has not yet been "
                  f"determined.")

        print('-' * 80)

        repetition_input = get_user_choice("Do you want to use the repaired log as the baseline for an additional "
                                           "repair? (yes/no): ")

        if repetition_input == 'no':
            break

        iteration += 1

        threshold_input = get_user_choice("Do you want to keep the probability threshold for the next repair? "
                                          "(yes/no): ")

        if threshold_input == 'no':
            reset_prob_threshold()

        config['complete_expert_attributes'], config['complete_expert_values'] = extend_expert_input()

        log = create_log(config, repetition=True, iteration=iteration)

    path_until_last_slash = config['config_file'].rsplit('/', 1)[0] + '/'
    model_config_path = os.path.join(path_until_last_slash, "repair")
    os.makedirs(model_config_path, exist_ok=True)
    filtered_config = {k: v for k, v in config.items() if k not in ['log', 'log_path', 'attribute_dictionary']}
    with open(os.path.join(model_config_path, "config.pkl"), 'wb') as file:
        pickle.dump(filtered_config, file)


def run_validation(model: Transformer, validation_ds: DataLoader, tokenizer_tgt: Tokenizer, max_len: int,
                   device: torch.device, print_msg: Callable[[str], None], global_step: int, writer: SummaryWriter,
                   config: dict, num_examples: int = 2) -> dict:
    """
    Run validation on the trained model and log evaluation metrics.

    :param model: The trained Transformer model.
    :param validation_ds: DataLoader for the validation dataset.
    :param tokenizer_tgt: Tokenizer for output.
    :param max_len: Maximum length of the output sequence.
    :param device: Device to perform computations on.
    :param print_msg: Function to print messages.
    :param global_step: Current global step in training.
    :param writer: TensorBoard SummaryWriter.
    :param config: Configuration options.
    :param num_examples: Number of examples to show during validation. Defaults to 2.
    :return: Dictionary containing the validation metrics.
    """
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    validation_metrics = {'wer': 0.0}

    try:
        # get the console window width
        console_width = shutil.get_terminal_size().columns
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, 1, seq_len)
            continuous_input = batch["continuous_input"].to(device)
            continuous_mask = batch["continuous_mask"].to(device)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out, _, _ = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                continuous_input,
                continuous_mask,
                tokenizer_tgt,
                max_len,
                device,
                config
            )

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            if config['continuous_input_attributes']:
                source_text = source_text + " " + batch["continuous_data"][0]

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print the source, target and model output
            print_msg('-' * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-' * console_width)
                break

    if writer:
        # Evaluate the character error rate
        # Compute the char error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()
        validation_metrics['wer'] = wer

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

    return validation_metrics


def save_created_logs(config: dict, determined_log: pd.DataFrame, repetition: bool, iteration: int) -> None:
    """
    Save determined logs to CSV and XES formats.

    :param config: Configuration parameters.
    :param determined_log: DataFrame containing the determined log.
    :param repetition: Flag to indicate if the process is a repetition.
    :param iteration: Iteration number.
    """
    os.makedirs(config['result_folder'], exist_ok=True)
    determined_log.to_csv(os.path.join(config['result_folder'], config['result_csv_file']), index=False)

    folder_path = os.path.join(config['result_folder'], 'iterations')
    os.makedirs(folder_path, exist_ok=True)

    if not repetition:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    iteration_file = os.path.join(folder_path, config['result_csv_iteration_file'].format(iteration))
    determined_log.to_csv(iteration_file, index=False)

    common_columns = [f"Original {config['tf_output']}", "Determination Probability",
                      "Determination Follow-up Probability"]
    prob_column = ["Iteration Probability"]
    follow_up_column = ["Iteration Follow-up Probability"]

    columns_to_drop = common_columns

    if 'Iteration Probability' in determined_log.columns:
        columns_to_drop += prob_column

    if 'Iteration Follow-up Probability' in determined_log.columns:
        columns_to_drop += follow_up_column

    extended_log = determined_log.drop(columns=columns_to_drop)

    if repetition:
        extended_log.drop(columns=[f"Previous {config['tf_output']}"], inplace=True)

    extended_log.fillna(config['missing_placeholder_xes'], inplace=True)
    extended_log.rename(columns={f"Determined {config['tf_output']}": f"{config['tf_output']}"}, inplace=True)
    extended_log.rename(columns=config['attribute_dictionary'], inplace=True)

    pm4py.write_xes(extended_log, os.path.join(config['result_folder'], config['result_xes_file']))


def select_columns(df: pd.DataFrame, column_mapping: dict, first_iteration: bool = True) -> pd.DataFrame:
    """
    Select columns from DataFrame based on selected columns.

    :param df: DataFrame containing the log data.
    :param column_mapping: Mapping of attribute aliases to column names.
    :param first_iteration: Flag to indicate whether it's the first iteration. Defaults to True.
    :return: DataFrame with selected columns.
    """
    selected_columns = {}
    automatically_selected_columns = {}
    selected_col_alias = []

    for col_alias, col_name in column_mapping.items():
        # Check if the column exists in the DataFrame
        if col_name.lower() in map(str.lower, df.columns):
            # If found, automatically select the column
            automatically_selected_columns[col_alias] = df.columns[
                df.columns.str.lower() == col_name.lower()].values[0]
            selected_col_alias.append(col_alias)
        elif col_alias.lower() in map(str.lower, df.columns):
            automatically_selected_columns[col_alias] = df.columns[
                df.columns.str.lower() == col_alias.lower()].values[0]
            selected_col_alias.append(col_alias)
        else:
            # If not found, prompt the user to select the column
            print(f"Column '{col_name}' for '{col_alias}' not found in the dataframe.")
            selected_column = select_matching_column(df, col_alias, selected_columns)
            selected_columns[col_alias] = selected_column

    # If some columns are selected automatically
    if len(selected_col_alias) != 0:
        if first_iteration:
            print(f"Following column{'s' if len(selected_col_alias) != 1 else ''} "
                  f"w{'ere' if len(selected_col_alias) != 1 else 'as'} automatically matched:")
            for index, col_alias in enumerate(selected_col_alias):
                print(f"'{automatically_selected_columns[col_alias]}' for '{col_alias}'"
                      f"{';' if index != len(selected_col_alias) - 1 else '.'}")

            # Ask for user confirmation on the automatically selected columns
            user_confirmation = get_user_choice(f"Is this {'completely ' if len(selected_col_alias) != 1 else ''}"
                                                f"correct? (yes/no): ")
            add_response(user_confirmation)

            if user_confirmation == 'no':
                if len(selected_col_alias) != 1:
                    input_config, value = get_input_config()
                    user_input = None
                    if input_config is not None:
                        user_input = value

                    if user_input is not None:
                        print(f"Please enter the incorrect attribute(s) separated by a comma: {user_input}")
                    else:
                        user_input = input(f"Please enter the incorrect attribute(s) separated by a comma: ")

                    incorrect_aliases = [alias.strip() for alias in user_input.split(',')]
                    if not all(alias in selected_col_alias for alias in incorrect_aliases):
                        raise ValueError("One or more provided incorrect attributes are not valid.")
                else:
                    incorrect_aliases = [selected_col_alias[0]]
                for alias in incorrect_aliases:
                    del automatically_selected_columns[alias]
                    selected_column = select_matching_column(df, alias, selected_columns)
                    selected_columns[alias] = selected_column

        # Check for duplicate selections
        for col_alias, col_name in automatically_selected_columns.items():
            if col_name in selected_columns.values():
                raise ValueError(f"Column '{col_name}' was inadmissibly selected twice.")

        # Update the selected columns with the automatically matched columns
        selected_columns.update(automatically_selected_columns)

    if 'Sorted Index' not in df.columns:
        df['Sorted Index'] = df.index

    original_column_names = list(selected_columns.values()) + ['Sorted Index']
    new_column_names = list(selected_columns.keys()) + ['Sorted Index']

    # Select columns from DataFrame based on selected columns
    df = df[original_column_names]
    df.columns = new_column_names

    return df


def select_matching_column(df: pd.DataFrame, alias: str, selected_columns: dict) -> str:
    """
    Prompt the user to select a column corresponding to the provided alias from the DataFrame.

    :param df: DataFrame containing the columns to choose from.
    :param alias: Alias corresponding to the column to be selected.
    :param selected_columns: Dictionary containing already selected columns.
    :return: Name of the selected column.
    """
    # List available columns for user selection
    available_columns = [col for col in df.columns.tolist() if col not in selected_columns.values()]
    num_available_columns = len(available_columns)
    if num_available_columns == 0:
        raise ValueError("No available columns left. Cannot proceed.")
    else:
        print(f"{num_available_columns} available column{'s' if num_available_columns > 1 else ''}: "
              f"{', '.join(available_columns)}")

    input_config, value = get_input_config()
    user_input = None
    if input_config is not None:
        user_input = value

    if user_input is not None:
        print(f"Please select the name of the column corresponding to '{alias}': {user_input}")
    else:
        user_input = input(f"Please select the name of the column corresponding to '{alias}': ")

    matching_column = next((col for col in df.columns if col.lower() == user_input.lower()), None)
    if matching_column:
        print(f"Column '{matching_column}' selected for '{alias}'.\n")
        return matching_column
    else:
        raise ValueError("No matching column found. Please try again.")


def train_epoch(config: dict, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, device: torch.device,
                initial_epoch: int, model: Transformer, train_dataloader: DataLoader, val_dataloader: DataLoader,
                global_step: int, optimizer: torch.optim.Adam) -> None:
    """
    Train multiple epochs of the Transformer model.

    :param config: A dictionary containing the configuration settings for the training.
    :param tokenizer_src: The tokenizer for the input data.
    :param tokenizer_tgt: The tokenizer for the output data.
    :param device: The device on which the model and data are processed.
    :param initial_epoch: The starting epoch number.
    :param model: The Transformer model to be trained.
    :param train_dataloader: The DataLoader for the training dataset.
    :param val_dataloader: The DataLoader for the validation dataset.
    :param global_step: The global step counter for logging and saving checkpoints.
    :param optimizer: The optimizer used for training the model.
    """
    writer = SummaryWriter(config['experiment_name'])
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    train_losses = []
    val_wer_values = []

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

        total_loss = 0.0

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            continuous_input = batch['continuous_input'].to(device)

            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            if config['continuous_input_attributes']:
                encoder_output = model.cont_enrichment(encoder_output, continuous_input)
                encoder_mask = batch['continuous_mask'].to(device)

            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            total_loss += loss.item()

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        average_loss = total_loss / len(train_dataloader)
        train_losses.append(average_loss)

        validation_metrics = run_validation(model, val_dataloader, tokenizer_tgt, config['seq_len'], device,
                                            lambda msg: batch_iterator.write(msg), global_step, writer, config)
        val_wer_values.append(validation_metrics['wer'])

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

    if initial_epoch != config['num_epochs']:
        plt.plot(range(initial_epoch, config['num_epochs']), train_losses, label='Training Loss')
        plt.plot(range(initial_epoch, config['num_epochs']), val_wer_values, label='Validation WER')

        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training Loss and Validation WER over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

    writer.close()


def train_model(config: dict) -> None:
    """
    Train the model based on the provided configuration.

    :param config: Configuration parameters.
    """
    device = initialize_device()

    model_configuration = load_or_initialize_model(config, device)
    model = model_configuration[0]
    initial_epoch = model_configuration[1]
    global_step = model_configuration[2]
    train_dataloader = model_configuration[3]
    val_dataloader = model_configuration[4]
    tokenizer_src = model_configuration[5]
    tokenizer_tgt = model_configuration[6]
    optimizer = model_configuration[7]

    train_epoch(config, tokenizer_src, tokenizer_tgt, device, initial_epoch, model, train_dataloader, val_dataloader,
                global_step, optimizer)


def validate_boundary_activity_input_data(config: dict, log: pd.DataFrame,
                                          activity_type: str) -> Tuple[str, List[str], List[str], str]:
    """
    Validate the input data for boundary activity rule checking.

    :param config: Configuration parameters.
    :param log: Event log represented as a pandas DataFrame.
    :param activity_type: Type of activity to check for ('start' or 'end').
    :return: Tuple containing attribute, values, occurrences, and activity_column.
    """
    if activity_type.lower() not in ['start', 'end']:
        raise ValueError("Invalid activity type. Please enter 'start' or 'end'.")

    attribute = f"{activity_type.capitalize()} Activity"
    values, attribute_column, occurrences = get_attribute_values(config, 'complete_expert_values', attribute)

    if not any(col.lower() == attribute_column.lower() for col in log.columns):
        raise ValueError(f"The column '{attribute_column}' specified for attribute '{attribute}' does not exist in "
                         f"the DataFrame.")

    if not isinstance(values, list) and all(isinstance(val, str) for val in values):
        raise ValueError("Unsupported format for values. Please provide a one-dimensional list.")

    activity_column = log.columns[log.columns.str.lower() == attribute_column.lower()][0]

    for val in values:
        if not log[activity_column].str.lower().str.contains(val.lower()).any():
            raise ValueError(f"The value '{val}' specified for attribute '{attribute}' does not exist in the DataFrame "
                             f"column '{activity_column}'.")

    return attribute, values, occurrences, activity_column


def validate_config_and_columns(config: dict, df: pd.DataFrame) -> None:
    """
    Validate the configuration and DataFrame columns.

    :param config: Configuration object containing necessary parameters.
    :param df: DataFrame containing the log data.
    """
    for attribute in config['expert_input_attributes']:
        if attribute.lower() not in [key.lower() for key in config['expert_input_values']]:
            raise ValueError(f"The attribute '{attribute}' is not found in config['expert_input_values'].")

        attribute_column = config['expert_input_values'][attribute]['attribute']
        if attribute_column.lower() not in [col.lower() for col in df.columns]:
            raise ValueError(f"The column '{attribute_column}' specified for attribute '{attribute}' "
                             f"does not exist in the DataFrame.")


def validate_directly_following_input_data(config: dict,
                                           log: pd.DataFrame) -> Tuple[str, List[str], List[str], str]:
    """
    Validate the input data based for directly following rule checking.

    :param config: Configuration parameters.
    :param log: Event log represented as a pandas DataFrame.
    :return: Tuple containing attribute, values, occurrences, and activity column.
    """
    attribute = 'Directly Following'

    values, attribute_column, occurrences = get_attribute_values(config, 'complete_expert_values', attribute)

    if not any(col.lower() == attribute_column.lower() for col in log.columns):
        raise ValueError(f"The column '{attribute_column}' specified for attribute '{attribute}' does not exist in "
                         f"the DataFrame.")

    if not isinstance(values, list) and all(isinstance(val, str) for sublist in values for val in sublist):
        raise ValueError("Unsupported format for values. Please provide a two-dimensional list.")

    activity_column = log.columns[log.columns.str.lower() == attribute_column.lower()][0]

    for sublist in values:
        for val in sublist:
            if not log[activity_column].str.lower().str.contains(val.lower()).any():
                raise ValueError(f"The value '{val}' specified for attribute '{attribute}' does not exist in the "
                                 f"DataFrame column '{activity_column}'.")

    if any(occurrence.lower() not in ['always', 'sometimes'] for occurrence in occurrences):
        raise ValueError("Invalid occurrence value. Please enter 'always' or 'sometimes'.")

    return attribute, values, occurrences, activity_column


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    use_config = get_response_configuration()

    print("\nConfiguration of model training")
    if use_config:
        print("Please ensure that the process matches the one in the response configuration file.")

    config = get_config()

    train_model(config)

    print("\nConfiguration of log repair")

    repaired_log = create_log(config)

    save_responses(config)

    repair_loop(repaired_log, config)
