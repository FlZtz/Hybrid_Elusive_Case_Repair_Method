# train.py - Script for training a Transformer model and creating a log with determined config['tf_output']
import os  # For filesystem operations
import shutil  # For filesystem operations
import warnings  # For managing warnings
from pathlib import Path  # For working with file paths
from typing import Callable, List, Tuple  # For type hints

# Third-party library imports
import pandas as pd  # For working with dataframes
from sklearn.model_selection import train_test_split  # For splitting dataset into train and
from sklearn.preprocessing import MinMaxScaler  # For normalizing continuous data

import torch  # PyTorch library
import torch.nn as nn  # PyTorch neural network module
from torch.utils.data import DataLoader, Dataset, random_split  # For working with PyTorch datasets
import torchmetrics  # For evaluation metrics
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard visualization

import pm4py  # For process mining operations

from dateutil import parser  # For parsing date strings

from datasets import Dataset as HuggingfaceDataset  # For Huggingface datasets
from tokenizers import models, pre_tokenizers, Tokenizer, trainers  # For tokenization

# Local application/library specific imports
from model import build_transformer, Transformer  # Importing Transformer model builder
from dataset import causal_mask, InOutDataset  # Importing custom dataset and mask functions
from config import (get_cached_df_copy, get_config, get_expert_input_columns, get_prob_threshold, get_weights_file_path,
                    latest_weights_file_path, reset_log, set_cached_df_copy, set_expert_input_columns)

from tqdm import tqdm  # For progress bars


def create_log(config: dict, chunk_size: int = None) -> pd.DataFrame:
    """
    Creates a log with determined config['tf_output'] based on the given configuration and chunk size.

    :param config: Dictionary containing configuration parameters for log creation. It includes keys like `tf_input`,
     `tf_output`, and `seq_len`.
    :param chunk_size: Number of rows to be processed as a single chunk. Default is set to `config['seq_len'] - 2`.
    :return: DataFrame representing the log with determined config['tf_output'].
    """
    prediction_preference = input(f"Do you want to predict the {config['tf_output']} values using the event log that "
                                  f"was used for training? (yes/no): ").strip().lower()

    if not prediction_preference or prediction_preference not in ['yes', 'no']:
        raise ValueError("Invalid input! Please enter 'yes' or 'no'.")

    if prediction_preference == 'yes':
        consider_ids = False
    else:
        reset_log(False)
        print("Please ensure the new event log and its name match the process used during training.")
        config = get_config()
        consider_ids = True

    data_complete = read_log(config, True)

    if chunk_size is None:
        chunk_size = config['seq_len'] - 2

    df = prepare_dataframe_for_sequence_processing(data_complete, config, chunk_size)

    # Create a raw dataset from the DataFrame
    ds_raw = HuggingfaceDataset.from_pandas(df)

    # Get or build tokenizers for source and target features
    vocab_src = get_or_build_tokenizer(config, ds_raw, 'Discrete Attributes')
    vocab_tgt = get_or_build_tokenizer(config, ds_raw, config['tf_output'])

    expert_input_columns = get_expert_input_columns()

    # Create a InOutDataset using source and target tokenizers
    ds = InOutDataset(ds_raw, vocab_src, vocab_tgt, 'Discrete Attributes', 'Continuous Attributes',
                      config['tf_output'], config['seq_len'],
                      len(config['discrete_input_attributes']) + len(expert_input_columns),
                      len(config['continuous_input_attributes']))

    # Create a DataLoader for the InOutDataset
    ds_dataloader = DataLoader(ds, batch_size=1, shuffle=False)

    # Determine the device to use for training (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the model based on the configuration and move it to the selected device
    model = get_model(config, vocab_src.get_vocab_size(), vocab_tgt.get_vocab_size()).to(device)

    # Load the pretrained weights for the model
    model_filename = get_weights_file_path(config, f"19")
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])

    # Initialize an empty list to store tuples representing rows
    determined_log = []

    prob_threshold = get_prob_threshold(config)

    # Iterate over batches in the DataLoader
    for batch in ds_dataloader:
        # Get input and mask tensors
        encoder_input = batch["encoder_input"].to(device)
        encoder_mask = batch["encoder_mask"].to(device)
        continuous_input = batch["continuous_input"].to(device)
        continuous_mask = batch["continuous_mask"].to(device)

        # Split target text into a list of values
        target_text = batch["tgt_text"][0].split()
        batch_size = len(target_text)

        try:
            # Ensure batch size is 1 for evaluation
            assert encoder_input.size(0) == 1, "Batch size must be 1 for evaluation"

            # Perform greedy decoding using the pre-trained model
            model_out, probabilities = greedy_decode(
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

            # Extend the determined_log list with tuples representing rows
            determined_log.extend((model_out_value, f"{prob * 100:.2f}%", target_value)
                                  for target_value, model_out_value, prob
                                  in zip(target_text, model_out_values, probabilities))
        except Exception as e:
            # If an error occurs during decoding, log the error and skip this batch
            print(f"Error occurred during decoding: {e}")
            continue

    # Convert the list of tuples to a DataFrame
    determined_log = pd.DataFrame(
        determined_log,
        columns=[
            f"Determined {config['tf_output']}",
            'Probability',
            f"Actual {config['tf_output']}"
        ]
    )

    determined_log.replace('[NONE]', pd.NA, inplace=True)
    determined_log['Probability'] = determined_log[
        'Probability'
    ].mask(determined_log[f"Determined {config['tf_output']}"].isna(), pd.NA)

    # Update Determined config['tf_output'] and Probability columns based on Actual config['tf_output'] if consider_ids
    if consider_ids:
        for idx, row in determined_log.iterrows():
            if pd.notna(row[f"Actual {config['tf_output']}"]):
                determined_log.at[idx, f"Determined {config['tf_output']}"] = row[f"Actual {config['tf_output']}"]
                determined_log.at[idx, 'Probability'] = "-"

    # Initialize an empty DataFrame to store the restored columns
    restored_columns = pd.DataFrame()

    for attr in config['tf_input']:
        attr_column = pd.DataFrame(data_complete[attr])
        restored_columns = pd.concat([restored_columns, attr_column], axis=1)

    # Concatenate determined_log with restored_columns and assign it back to determined_log
    determined_log = pd.concat([determined_log, restored_columns], axis=1)

    # Create directories if they don't exist
    os.makedirs(config['result_folder'], exist_ok=True)

    # Save the determined log as a CSV file
    determined_log.to_csv(os.path.join(config['result_folder'], config['result_csv_file']), index=False)

    # Prepare extended log
    extended_log = determined_log.drop(columns=[f"Actual {config['tf_output']}", 'Probability'])

    extended_log.fillna('', inplace=True)

    if 'Timestamp' not in config['tf_input']:
        timestamp_column = pd.DataFrame(data_complete['Timestamp'])
        extended_log = pd.concat([extended_log, timestamp_column], axis=1)

    # Rename 'Determined config['tf_output']' to 'config['tf_output']' first
    extended_log.rename(columns={f"Determined {config['tf_output']}": f"{config['tf_output']}"}, inplace=True)

    # Then rename the rest of the columns specified in config['attribute_dictionary']
    extended_log.rename(columns=config['attribute_dictionary'], inplace=True)

    # Convert to event log
    log = pm4py.convert_to_event_log(extended_log)

    # Write XES file
    pm4py.write_xes(log, os.path.join(config['result_folder'], config['result_xes_file']))

    # Return the determined log DataFrame
    return determined_log


def get_all_sequences(ds: Dataset, data: str) -> str:
    """
    Generator function to yield all sequences in a dataset.

    :param ds: Dataset to extract sequences from.
    :param data: Name of the data field.
    :return: Sequences from the dataset.
    """
    for item in ds:
        yield item[data]


def get_ds(config: dict) -> Tuple[DataLoader, DataLoader, Tokenizer, Tokenizer]:
    """
    Gets training and validation DataLoaders along with tokenizers for the dataset.

    :param config: Configuration options.
    :return: Training DataLoader; Validation DataLoader; Source Tokenizer; Target Tokenizer.
    """
    df = read_log(config)
    train, test = train_test_split(df, test_size=0.1, shuffle=True)
    ds_raw = HuggingfaceDataset.from_pandas(train)

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, 'Discrete Attributes')
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['tf_output'])

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
        d_model=config['d_model']
    )

    return model


def get_or_build_tokenizer(config: dict, ds: Dataset, data: str) -> Tokenizer:
    """
    Get or build a tokenizer for a specific dataset and data field.

    :param config: Configuration options.
    :param ds: Dataset to build tokenizer for.
    :param data: Name of the data field.
    :return: Tokenizer for the dataset.
    """
    # Check if tokenizer file exists
    os.makedirs(config['tokenizer_folder'], exist_ok=True)
    tokenizer_path = Path(os.path.join(config['tokenizer_folder'], config['tokenizer_file'].format(data)))
    if not tokenizer_path.exists():
        # Train a new tokenizer
        tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
        # Customize pre-tokenization and training
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]", "[NONE]"])
        tokenizer.train_from_iterator(get_all_sequences(ds, data), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        # Load existing tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def greedy_decode(model: Transformer, source: torch.Tensor, source_mask: torch.Tensor, cont_input: torch.Tensor,
                  cont_mask: torch.Tensor, tokenizer_tgt: Tokenizer, max_len: int, device: torch.device,
                  config: dict, prob_threshold: float = 0) -> Tuple[torch.Tensor, List[float]]:
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
    :return: The decoded target sequence tensor and the corresponding probabilities.
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    none_idx = tokenizer_tgt.token_to_id('[NONE]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)

    if config['continuous_input_attributes']:
        encoder_output = model.cont_enrichment(encoder_output, cont_input)
        source_mask = cont_mask

    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    probabilities = []

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
        probs, indices = torch.topk(prob, k=2, dim=1)
        next_index = indices[:, 0]
        if next_index == eos_idx:
            if probs[:, 1].item() >= prob_threshold:
                next_output = indices[:, 1].item()
                probabilities.append(probs[:, 1].item())
            else:
                next_output = none_idx
                probabilities.append(0)
        else:
            if probs[:, 0].item() >= prob_threshold:
                next_output = next_index.item()
                probabilities.append(probs[:, 0].item())
            else:
                next_output = none_idx
                probabilities.append(0)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_output).to(device)], dim=1
        )

        if next_output == eos_idx:
            break

    return decoder_input.squeeze(0), probabilities


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


def process_expert_input_attributes(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Process expert input attributes and append them to the DataFrame.

    :param df: DataFrame containing the log data.
    :param config: Configuration object containing necessary parameters.
    :return: DataFrame with expert input attributes appended.
    """
    # Initialize a list to store column headers describing expert input
    expert_input_columns = []

    # Check if there are values in config['expert_input_attributes']
    if config['expert_input_attributes']:
        # Create columns based on expert input attributes and values and append them to the DataFrame
        for attribute in config['expert_input_attributes']:
            if any(attribute.lower() == key.lower() for key in config['expert_input_values']):
                values = config['expert_input_values'][attribute]['values']
                attribute_column = config['expert_input_values'][attribute]['attribute']
                occurrences = config['expert_input_values'][attribute]['occurrences']

                # Check if attribute_column exists in df.columns without case sensitivity
                if any(col.lower() == attribute_column.lower() for col in df.columns):
                    transformed_attribute_column = df.columns[df.columns.str.lower() == attribute_column.lower()][0]

                    # One-dimensional list case
                    if isinstance(values, list) and all(isinstance(val, str) for val in values):
                        for val in values:
                            if not df[transformed_attribute_column].str.lower().str.contains(val.lower()).any():
                                raise ValueError(f"The value '{val}' specified for attribute '{attribute}' does not "
                                                 f"exist in the DataFrame column '{transformed_attribute_column}'.")

                        attribute_name = attribute.lower().replace(' ', '_')
                        # Create column for attribute
                        df[attribute] = f'expert_non_{attribute_name}'

                        # Append the column header for the attribute to the list
                        expert_input_columns.append(attribute)

                        # Iterate over values and occurrences after creating the column
                        for val, occurrence in zip(values, occurrences):
                            # Set values based on config['expert_input_values'] and occurrences
                            if occurrence.lower() == 'always':
                                df.loc[
                                    df[transformed_attribute_column]
                                    .str.lower()
                                    .isin([val.lower()]),
                                    attribute
                                ] = f'expert_always_{attribute_name}'
                            elif occurrence.lower() == 'sometimes':
                                df.loc[
                                    df[transformed_attribute_column]
                                    .str.lower()
                                    .isin([val.lower()]),
                                    attribute
                                ] = f'expert_sometimes_{attribute_name}'
                            else:
                                raise ValueError("Invalid occurrence value. Please enter 'always' or 'sometimes'.")

                    # Two-dimensional list case
                    elif isinstance(values, list) and all(
                            isinstance(val, str) for sublist in values for val in sublist):
                        for sublist in values:
                            for val in sublist:
                                # Check if each value in the sublist exists in the DataFrame column
                                if not df[transformed_attribute_column].str.lower().str.contains(val.lower()).any():
                                    raise ValueError(f"The value '{val}' specified for attribute '{attribute}' "
                                                     f"does not exist in the DataFrame column "
                                                     f"'{transformed_attribute_column}'.")

                        attribute_name = attribute.lower().replace(' ', '_')

                        # Extract unique first values from the lists within the list of values
                        first_values_list = list(set(sublist[0].lower() for sublist in values))

                        for value in first_values_list:
                            # Filter values and occurrences based on the condition
                            filtered_values = [val for val in values if val[0].lower() == value]
                            filtered_occurrences = [occurrence for val, occurrence in zip(values, occurrences) if
                                                    val[0].lower() == value]

                            # Transform value to match the case of the values in the DataFrame column
                            transformed_value = (
                                df[df[transformed_attribute_column].str.lower() == value.lower()]
                                [transformed_attribute_column]
                                .iloc[0]
                                .replace(' ', '_')
                            )

                            # Create new column in DataFrame
                            df[f'{attribute}_{transformed_value}'] = f'expert_non_{attribute_name}_{transformed_value}'

                            # Append the column headers containing the attribute to the list
                            expert_input_columns.extend([f'{attribute}_{transformed_value}'])

                            # Assign values based on occurrences
                            for val, occurrence in zip(filtered_values, filtered_occurrences):
                                if occurrence.lower() == 'always':
                                    df.loc[
                                        df[transformed_attribute_column]
                                        .str.lower()
                                        .isin([val[1].lower()]),
                                        f'{attribute}_{transformed_value}'
                                    ] = f'expert_always_{attribute_name}_{transformed_value}'
                                elif occurrence.lower() == 'sometimes':
                                    df.loc[
                                        df[transformed_attribute_column]
                                        .str.lower()
                                        .isin([val[1].lower()]),
                                        f'{attribute}_{transformed_value}'
                                    ] = f'expert_sometimes_{attribute_name}_{transformed_value}'
                                else:
                                    raise ValueError("Invalid occurrence value. Please enter 'always' or 'sometimes'.")

                    else:
                        # Raise a ValueError for unsupported value format
                        raise ValueError("Unsupported format for values. "
                                         "Please provide either a one-dimensional or two-dimensional list.")
                else:
                    raise ValueError(f"The column '{attribute_column}' specified for attribute "
                                     f"'{attribute}' does not exist in the DataFrame.")

            else:
                raise ValueError(f"The attribute '{attribute}' is not found in config['expert_input_values'].")

    # Set the expert input columns
    set_expert_input_columns(expert_input_columns)

    return df


def read_log(config: dict, complete: bool = False) -> pd.DataFrame:
    """
    Read log file and preprocess data according to configuration.

    :param config: Configuration object containing necessary parameters.
    :param complete: Flag to indicate if the complete DataFrame should be returned. Defaults to False.
    :return: Processed DataFrame according to the specified configuration.
    """
    cached_df_copy = get_cached_df_copy()
    if cached_df_copy is not None and complete:
        return cached_df_copy

    df = config['log']

    # Define the required attributes including 'Timestamp' only if it's not already in config['tf_input']
    # or config['tf_output']
    required_attributes = list({*config['tf_input'], config['tf_output'], 'Timestamp'})

    # Create a mapping of attribute aliases to column names
    column_mapping = {key: value for key, value in config['attribute_dictionary'].items()
                      if key in required_attributes}

    df = select_columns(df, column_mapping)

    # Convert Timestamp column to datetime
    df['Timestamp'] = df['Timestamp'].apply(lambda x: parser.isoparse(x) if isinstance(x, str) else x)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
    df = df.sort_values(['Timestamp']).reset_index(drop=True)
    df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)

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
        return df

    df = prepare_dataframe_for_sequence_processing(df, config)

    # Return the prepared DataFrame
    return df


def run_validation(model: Transformer, validation_ds: DataLoader, tokenizer_tgt: Tokenizer, max_len: int,
                   device: torch.device, print_msg: Callable[[str], None], global_step: int, writer: SummaryWriter,
                   config: dict, num_examples: int = 2) -> None:
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
    """
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

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

            model_out, _ = greedy_decode(
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

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()


def select_columns(df: pd.DataFrame, column_mapping: dict) -> pd.DataFrame:
    """
    Select columns from DataFrame based on selected columns.

    :param df: DataFrame containing the log data.
    :param column_mapping: Mapping of attribute aliases to column names.
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
        print(f"Following column{'s' if len(selected_col_alias) != 1 else ''} "
              f"w{'ere' if len(selected_col_alias) != 1 else 'as'} automatically matched:")
        for index, col_alias in enumerate(selected_col_alias):
            print(f"'{automatically_selected_columns[col_alias]}' for '{col_alias}'"
                  f"{';' if index != len(selected_col_alias) - 1 else '.'}")

        # Ask for user confirmation on the automatically selected columns
        user_confirmation = input(f"Is this {'completely ' if len(selected_col_alias) != 1 else ''}"
                                  f"correct? (yes/no): ").lower().strip()

        if not user_confirmation or user_confirmation not in ('yes', 'no'):
            raise ValueError("Please enter either 'yes' or 'no'.")

        if user_confirmation == 'no':
            if len(selected_col_alias) != 1:
                incorrect_aliases = [alias.strip() for alias in input(f"Please enter the incorrect attribute(s) "
                                                                      f"separated by a comma: ").split(',')]
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

    # Select columns from DataFrame based on selected columns
    df = df[list(selected_columns.values())]
    df.columns = list(selected_columns.keys())

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
    # Prompt the user to select the column
    user_input = input(f"Please select the name of the column corresponding to '{alias}': ")
    matching_column = next((col for col in df.columns if col.lower() == user_input.lower()), None)
    if matching_column:
        print(f"Column '{matching_column}' selected for '{alias}'.\n")
        return matching_column
    else:
        raise ValueError("No matching column found. Please try again.")


def train_model(config: dict) -> None:
    """
    Trains the Transformer model.

    :param config: Configuration options.
    """
    # Define the device
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.has_mps or torch.backends.mps.is_available()
              else "cpu")
    print("Using device:", device)
    if device == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif device == 'mps':
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print(
            "      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print(
            "      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url "
            "https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = (latest_weights_file_path(config)
                      if preload == 'latest'
                      else get_weights_file_path(config, preload) if preload
                      else None)
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)  # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)
            continuous_input = batch['continuous_input'].to(device)

            encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)

            if config['continuous_input_attributes']:
                encoder_output = model.cont_enrichment(encoder_output, continuous_input)
                encoder_mask = batch['continuous_mask'].to(device)

            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input,
                                          decoder_mask)  # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)  # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device)  # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_tgt, config['seq_len'], device,
                       lambda msg: batch_iterator.write(msg), global_step, writer, config)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    print("Configuration of model training")
    config = get_config()
    train_model(config)
    create_log(config)
