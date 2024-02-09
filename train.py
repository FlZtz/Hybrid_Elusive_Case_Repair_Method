# train.py - Script for training a Transformer model and creating a log with determined case IDs
import shutil  # For filesystem operations
import pandas as pd  # For working with dataframes
import pm4py  # For process mining operations
from pm4py.objects.conversion.log import converter as log_converter  # For log conversion
from model import build_transformer, Transformer  # Importing Transformer model builder
from dataset import BilingualDataset, causal_mask  # Importing custom dataset and mask functions
from config import get_config, get_weights_file_path, latest_weights_file_path, get_cached_df_copy, set_cached_df_copy
import torchtext.datasets as datasets  # For accessing torchtext datasets
import torch  # PyTorch library
import torch.nn as nn  # PyTorch neural network module
from torch.utils.data import Dataset, DataLoader, random_split  # For working with PyTorch datasets
from torch.optim.lr_scheduler import LambdaLR  # For learning rate scheduling
import warnings  # For managing warnings
from tqdm import tqdm  # For progress bars
import os  # For operating system related operations
from pathlib import Path  # For working with file paths
from dateutil import parser  # For parsing date strings
from sklearn.model_selection import train_test_split  # For splitting dataset into train and test
from datasets import load_dataset, Dataset  # For Huggingface datasets
from tokenizers import Tokenizer, models, trainers, pre_tokenizers  # For tokenization
import torchmetrics  # For evaluation metrics
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard visualization
from typing import Callable, Tuple, Union  # For type hints
from argparse import Namespace  # For argument parsing


def greedy_decode(model: Transformer, source: torch.Tensor, source_mask: torch.Tensor, tokenizer_src: Tokenizer,
                  tokenizer_tgt: Tokenizer, max_len: int, device: torch.device) -> torch.Tensor:
    """
    Greedy decoding algorithm for generating target sequences based on a trained Transformer model.

    :param model: The trained Transformer model.
    :param source: The source input tensor.
    :param source_mask: The mask for source sequence.
    :param tokenizer_src: Tokenizer for source language.
    :param tokenizer_tgt: Tokenizer for target language.
    :param max_len: Maximum length of the output sequence.
    :param device: Device to perform computations on.
    :return: The decoded target sequence tensor.
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len - 1:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, indices = torch.topk(prob, k=2, dim=1)
        next_index = indices[:, 0]
        if next_index == eos_idx:
            next_output = indices[:, 1]
        else:
            next_output = next_index
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_output.item()).to(device)], dim=1
        )

        if next_output == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model: Transformer, validation_ds: DataLoader, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer,
                   max_len: int, device: torch.device, print_msg: Callable[[str], None], global_step: int,
                   writer: SummaryWriter, num_examples: int = 2) -> None:
    """
    Run validation on the trained model and log evaluation metrics.

    :param model: The trained Transformer model.
    :param validation_ds: DataLoader for the validation dataset.
    :param tokenizer_src: Tokenizer for source language.
    :param tokenizer_tgt: Tokenizer for target language.
    :param max_len: Maximum length of the output sequence.
    :param device: Device to perform computations on.
    :param print_msg: Function to print messages.
    :param global_step: Current global step in training.
    :param writer: TensorBoard SummaryWriter.
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

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

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


def get_all_sentences(ds: Dataset, data: str) -> str:
    """
    Generator function to yield all sentences in a dataset.

    :param ds: Dataset to extract sentences from.
    :param data: Name of the data field.
    :return: Sentences from the dataset.
    """
    for item in ds:
        yield item[data]


def get_or_build_tokenizer(config: Namespace, ds: Dataset, data: str) -> Tokenizer:
    """
    Get or build a tokenizer for a specific dataset and data field.

    :param config: Configuration options.
    :param ds: Dataset to build tokenizer for.
    :param data: Name of the data field.
    :return: Tokenizer for the dataset.
    """
    # Check if tokenizer file exists
    tokenizer_path = Path(config['tokenizer_file'].format(config['log_name'], data))
    if not Path.exists(tokenizer_path):
        # Train a new tokenizer
        tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
        # Customize pre-tokenization and training
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"])
        tokenizer.train_from_iterator(get_all_sentences(ds, data), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        # Load existing tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def read_log(config: Namespace, complete: bool = False) -> pd.DataFrame:
    """
    Read log file and preprocess data according to configuration.

    :param config: Configuration object containing necessary parameters.
    :param complete: Flag to indicate if the complete DataFrame should be returned. Defaults to False.
    :return: Processed DataFrame according to the specified configuration.
    """
    cached_df_copy = get_cached_df_copy()
    # Check if the cached DataFrame exists and if complete is True
    if cached_df_copy is not None and complete:
        return cached_df_copy

    # Check the file extension to determine the type of log file
    if config['log_path'].endswith('.csv'):
        # Read CSV file into a DataFrame
        df = pd.read_csv(config['log_path'])
    elif config['log_path'].endswith('.xes'):
        # Read XES file into a DataFrame using PM4Py library
        file = pm4py.read_xes(config['log_path'])
        df = log_converter.apply(file, variant=log_converter.Variants.TO_DATA_FRAME)
    else:
        # Raise an error for unsupported file types
        raise ValueError('Unknown file type. Supported types are .csv and .xes.')

    # Define the required attributes including 'Timestamp' only if it's not already in config['tf_input']
    # or config['tf_output']
    required_attributes = list({config['tf_input'], config['tf_output'], 'Timestamp'})

    # Create a mapping of attribute aliases to column names
    column_mapping = {key: value for key, value in config['attribute_dictionary'].items()
                      if key in required_attributes}

    # Initialize dictionaries and lists for selected columns
    selected_columns = {}
    automatically_selected_columns = {}
    selected_col_alias = []

    # Iterate through the attribute mapping
    for col_alias, col_name in column_mapping.items():
        # Check if the column exists in the DataFrame
        if col_name.lower() in map(str.lower, df.columns):
            # If found, automatically select the column
            automatically_selected_columns[col_alias] = col_name
            selected_col_alias.append(col_alias)
        else:
            # If not found, prompt the user to select the column
            print(f"Column '{col_name}' for '{col_alias}' not found in the dataframe.")
            # List available columns for user selection
            available_columns = [col for col in df.columns.tolist() if col not in selected_columns.values()]
            num_available_columns = len(available_columns)
            if num_available_columns == 0:
                raise ValueError("No available columns left. Cannot proceed.")
            else:
                print(f"{num_available_columns} available column{'s' if num_available_columns > 1 else ''}: "
                      f"{', '.join(available_columns)}")
            # Prompt the user to select the column
            user_input = input(f"Please select the name of the column corresponding to '{col_alias}': ")
            matching_column = next((col for col in df.columns if col.lower() == user_input.lower()), None)
            if matching_column:
                selected_columns[col_alias] = matching_column
                print(f"Column '{matching_column}' selected for '{col_alias}'.\n")
            else:
                raise ValueError("No matching column found. Please try again.")

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
                available_columns = [col for col in df.columns.tolist() if col not in selected_columns.values()]
                num_available_columns = len(available_columns)
                if num_available_columns == 0:
                    raise ValueError("No available columns left. Cannot proceed.")
                else:
                    print(f"{num_available_columns} available column{'s' if num_available_columns > 1 else ''}: "
                          f"{', '.join(available_columns)}")
                user_input = input(f"Please select the name of the column corresponding to '{alias}': ")
                matching_column = next((col for col in df.columns if col.lower() == user_input.lower()), None)
                if matching_column:
                    selected_columns[alias] = matching_column
                    print(f"Column '{matching_column}' selected for '{alias}'.\n")
                else:
                    raise ValueError("No matching column found. Please try again.")

        # Check for duplicate selections
        for col_alias, col_name in automatically_selected_columns.items():
            if col_name in selected_columns.values():
                raise ValueError(f"Column '{col_name}' was inadmissibly selected twice.")

        # Update the selected columns with the automatically matched columns
        selected_columns.update(automatically_selected_columns)

    # Select columns from DataFrame based on selected columns
    df = df[list(selected_columns.values())]
    df.columns = list(selected_columns.keys())

    # Convert Timestamp column to datetime
    df['Timestamp'] = df['Timestamp'].apply(lambda x: parser.isoparse(x) if isinstance(x, str) else x)
    df = df.sort_values(['Timestamp']).reset_index(drop=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
    df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)

    # Convert DataFrame to string type and replace spaces and hyphens with underscores
    df = df.astype(str).applymap(lambda x: x.replace(' ', '_').replace('-', '_'))

    # Create a copy of DataFrame with required attributes
    df_copy = df[[config['tf_input'], config['tf_output']]].copy()

    # Cache a copy of the DataFrame for potential future use
    set_cached_df_copy(df_copy)

    # If complete flag is set, return the DataFrame
    if complete:
        return df_copy

    # Prepare the DataFrame for sequence processing
    length = config['seq_len'] - 2
    if len(df) >= length:
        for i in range(len(df) - length + 1):
            # Concatenate sequences of input and output attributes
            df_copy.at[i, config['tf_input']] = ' '.join(df[config['tf_input']].iloc[i:i + length])
            df_copy.at[i, config['tf_output']] = ' '.join(df[config['tf_output']].iloc[i:i + length])

        # Drop rows with insufficient sequence length
        df_copy = df_copy.drop(df_copy.index[len(df) - length + 1:])
    else:
        raise ValueError(f"Length of the dataframe ({len(df)}) is less than {length}.")

    # Return the prepared DataFrame
    return df_copy


def get_ds(config: Namespace) -> Tuple[DataLoader, DataLoader, Tokenizer, Tokenizer]:
    """
    Gets training and validation DataLoaders along with tokenizers for the dataset.

    :param config: Configuration options.
    :return: Training DataLoader; Validation DataLoader; Source Tokenizer; Target Tokenizer.
    """
    df = read_log(config)
    # TODO: shuffle = false ?
    train, test = train_test_split(df, test_size=0.1)
    ds_raw = Dataset.from_pandas(train)

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['tf_input'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['tf_output'])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['tf_input'], config['tf_output'],
                                config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['tf_input'], config['tf_output'],
                              config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item[config['tf_input']]).ids
        tgt_ids = tokenizer_tgt.encode(item[config['tf_output']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config: Namespace, vocab_src_len: int, vocab_tgt_len: int) -> nn.Module:
    """
    Builds and returns the Transformer model.

    :param config: Configuration options.
    :param vocab_src_len: Length of source vocabulary.
    :param vocab_tgt_len: Length of target vocabulary.
    :return: Transformer model.
    """
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'],
                              d_model=config['d_model'])
    return model


def train_model(config: Namespace) -> None:
    """
    Trains the Transformer model.

    :param config: Configuration options.
    """
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available()\
        else "cpu"
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
    model_filename = latest_weights_file_path(config) \
        if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
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
            encoder_input = batch['encoder_input'].to(device)  # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
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
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,
                       lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


def create_log(config: Namespace, chunk_size: int = None) -> pd.DataFrame:
    """
    Creates a log with determined case IDs based on the given configuration and chunk size.

    :param config: Dictionary containing configuration parameters for log creation. It includes keys like 'tf_input',
    'tf_output', and 'seq_len'.
    :param chunk_size: Number of rows to be processed as a single chunk. Default is set to `config['seq_len']` - 2.
    :return: DataFrame representing the log with determined cases, including columns for 'Determined Case ID',
    'Actual Case ID', and 'Activity'.
    """
    if chunk_size is None:
        chunk_size = config['seq_len'] - 2

    # Read the complete log data
    data_complete = read_log(config, True)

    # Create a copy of the complete data
    df = data_complete.copy()

    # Loop through the data in chunks and concatenate values for input and output features
    for i in range(len(data_complete) // chunk_size):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size

        # Concatenate values for input feature
        df.at[start_idx, config['tf_input']] = ' '.join(data_complete[config['tf_input']].iloc[start_idx:end_idx])
        # Concatenate values for output feature
        df.at[start_idx, config['tf_output']] = ' '.join(data_complete[config['tf_output']].iloc[start_idx:end_idx])

    # Handle remaining rows if any
    remaining_rows = len(data_complete) % chunk_size
    if remaining_rows > 0:
        start_idx = len(data_complete) - remaining_rows
        # Concatenate values for input and output features for remaining rows
        df.loc[start_idx:, config['tf_input']] = ' '.join(data_complete[config['tf_input']].iloc[start_idx:])
        df.loc[start_idx:, config['tf_output']] = ' '.join(data_complete[config['tf_output']].iloc[start_idx:])

    # Drop unnecessary rows to keep only one row per chunk
    df = df.iloc[::chunk_size].reset_index(drop=True)

    # Create a raw dataset from the DataFrame
    ds_raw = Dataset.from_pandas(df)

    # Get or build tokenizers for source and target features
    vocab_src = get_or_build_tokenizer(config, ds_raw, config['tf_input'])
    vocab_tgt = get_or_build_tokenizer(config, ds_raw, config['tf_output'])

    # Create a BilingualDataset using source and target tokenizers
    ds = BilingualDataset(ds_raw, vocab_src, vocab_tgt, config['tf_input'], config['tf_output'], config['seq_len'])

    # Create a DataLoader for the BilingualDataset
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

    # Iterate over batches in the DataLoader
    for batch in ds_dataloader:
        # Get input and mask tensors
        encoder_input = batch["encoder_input"].to(device)
        encoder_mask = batch["encoder_mask"].to(device)

        # Split source and target text into lists of values
        source_text = batch["src_text"][0].split()
        target_text = batch["tgt_text"][0].split()
        batch_size = len(target_text)

        # Ensure batch size is 1 for evaluation
        assert encoder_input.size(0) == 1, "Batch size must be 1 for evaluation"

        # Perform greedy decoding using the pre-trained model
        model_out = greedy_decode(model, encoder_input, encoder_mask, vocab_src, vocab_tgt, batch_size + 2, device)
        model_out_values = vocab_tgt.decode(model_out.detach().cpu().numpy()).split()

        # Extend the determined_log list with tuples representing rows
        determined_log.extend((model_out_value, target_value, source_value)
                              for source_value, target_value, model_out_value
                              in zip(source_text, target_text, model_out_values))

    # Convert the list of tuples to a DataFrame
    determined_log = pd.DataFrame(determined_log, columns=['Determined Case ID', 'Actual Case ID', 'Activity'])

    # Save the determined log as a CSV file
    determined_log.to_csv(config['result_file_name'])

    # Return the determined log DataFrame
    return determined_log


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
    create_log(config)
