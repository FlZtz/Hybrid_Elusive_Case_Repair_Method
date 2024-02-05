import shutil

import pandas as pd
import pm4py
from pm4py.objects.conversion.log import converter as log_converter

from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

from dateutil import parser
from sklearn.model_selection import train_test_split

# Huggingface datasets and tokenizers
from datasets import load_dataset, Dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

import torchmetrics
from torch.utils.tensorboard import SummaryWriter


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
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


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer,
                   num_examples=2):
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


def get_all_sentences(ds, data):
    for item in ds:
        yield item[data]


def get_or_build_tokenizer(config, ds, data):
    tokenizer_path = Path(config['tokenizer_file'].format(config['log_name'], data))

    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        # Initialize the Tokenizer
        tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))

        # Set the pre_tokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        # Initialize the Trainer
        trainer = trainers.WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"])

        # Train the tokenizer
        tokenizer.train_from_iterator(get_all_sentences(ds, data), trainer=trainer)

        # Save the tokenizer
        tokenizer.save(str(tokenizer_path))
    else:
        # Load the tokenizer from the saved file
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def read_log(config, complete=False):
    if config['log_path'].endswith('.csv'):
        df = pd.read_csv(config['log_path'])
    elif config['log_path'].endswith('.xes'):
        file = pm4py.read_xes(config['log_path'])
        df = log_converter.apply(file, variant=log_converter.Variants.TO_DATA_FRAME)
    else:
        raise ValueError('Unknown file type. Supported types are .csv and .xes.')

    # Selecting columns
    df = df[['case:concept:name', 'concept:name', 'time:timestamp', 'org:resource']]
    df.columns = ['Case ID', 'Activity', 'Timestamp', 'Resource']

    # Replace whitespaces with underscores in column values
    df = df.map(lambda x: x.replace(' ', '_') if isinstance(x, str) else x)

    # Use dateutil.parser for ISO8601 string parsing
    df['Timestamp'] = df['Timestamp'].apply(lambda x: parser.isoparse(x) if isinstance(x, str) else x)
    df = df.sort_values(['Timestamp']).reset_index(drop=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
    df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)

    df_copy = df[[config['tf_input'], config['tf_output']]].copy()

    if complete:
        return df_copy

    length = config['seq_len'] - 2

    if len(df) >= length:
        for i in range(len(df) - length + 1):
            df_copy.at[i, config['tf_input']] = ' '.join(df[config['tf_input']].iloc[i:i + length])
            df_copy.at[i, config['tf_output']] = ' '.join(df[config['tf_output']].iloc[i:i + length])

        df_copy = df_copy.drop(df_copy.index[len(df) - length + 1:])
    else:
        raise ValueError(f"Length of the dataframe ({len(df)}) is less than {length}.")

    return df_copy


def get_ds(config):
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


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'],
                              d_model=config['d_model'])
    return model


def train_model(config):
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


def create_log(config, chunk_size=10) -> pd.DataFrame:
    """
    Creates a log with determined case IDs based on the given configuration and chunk size.

    :param config: Dictionary containing configuration parameters for log creation. It includes keys like 'tf_input',
    'tf_output', and 'seq_len'.
    :param chunk_size: Number of rows to be processed as a single chunk. Default is set to 10.
    :return: DataFrame representing the log with determined cases, including columns for 'Determined Case ID',
    'Actual Case ID', and 'Activity'.
    """
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
    state = torch.load(model_filename)
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
