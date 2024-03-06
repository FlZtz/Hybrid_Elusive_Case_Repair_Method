# dataset.py - Definition of InOutDataset class for handling input and output data and related utilities.
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class InOutDataset(Dataset):
    """
    Dataset class for handling input and output data.
    """
    def __init__(self, ds: Dataset, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, disc_input: str,
                 cont_input: str, tf_output: str, seq_len: int, num_disc_attr: int, num_cont_attr: int) -> None:
        """
        Initialize the dataset.

        :param ds: List of dictionaries containing input and output data pairs.
        :param tokenizer_src: Input tokenizer.
        :param tokenizer_tgt: Output tokenizer.
        :param disc_input: Key indicating the discrete input data in the dictionary.
        :param cont_input: Key indicating the continuous input data in the dictionary.
        :param tf_output: Key indicating the output data in the dictionary.
        :param seq_len: Maximum sequence length for inputs and outputs.
        :param num_disc_attr: Number of discrete input attributes.
        :param num_cont_attr: Number of continuous input attributes.
        """
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.disc_input = disc_input
        self.cont_input = cont_input
        self.tf_output = tf_output
        self.num_disc_attr = num_disc_attr
        self.num_cont_attr = num_cont_attr

        # Special tokens for start of sequence (SOS), end of sequence (EOS), and padding (PAD)
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self) -> int:
        """
        Get the number of items in the dataset.

        :return: Number of items in the dataset.
        """
        if hasattr(self.ds, '__len__'):
            return len(self.ds)
        else:
            raise TypeError("Dataset object does not support length operation.")

    def __getitem__(self, idx: int) -> dict:
        """
        Generate a single training example.

        :param idx: Index of the example in the dataset.
        :return: Dictionary containing encoder input, decoder input, encoder mask, decoder mask, label,
         source text, and target text.
        """
        src_target_pair = self.ds[idx]
        src_text = src_target_pair[self.disc_input]
        cont_data = src_target_pair[self.cont_input]
        tgt_text = src_target_pair[self.tf_output]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        cont_input_tokens = [float(value) for value in cont_data.split()]
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Calculate the number of padding tokens to add
        # Accounting for SOS and EOS tokens
        enc_num_padding_tokens = (self.seq_len - 2) * self.num_disc_attr - len(enc_input_tokens)
        cont_num_padding_tokens = (self.seq_len - 2) * self.num_cont_attr - len(cont_input_tokens) + 2
        dec_num_padding_tokens = self.seq_len - 1 - len(dec_input_tokens)  # Accounting for SOS token only

        # Ensure the length of the sequences does not exceed the specified maximum
        if any(num < 0 for num in [enc_num_padding_tokens, cont_num_padding_tokens, dec_num_padding_tokens]):
            raise ValueError("Sequence is too long")

        # Construct encoder input tensor with SOS, EOS, and padding tokens
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Construct continuous input tensor with padding tokens
        cont_input = torch.cat(
            [
                torch.tensor(cont_input_tokens, dtype=torch.float16),
                torch.tensor([self.pad_token] * cont_num_padding_tokens, dtype=torch.float16),
            ],
            dim=0,
        )

        # Construct decoder input tensor with SOS and padding tokens
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Construct label tensor with EOS and padding tokens
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Ensure all tensors have the correct sequence length
        assert encoder_input.size(0) == (self.seq_len - 2) * self.num_disc_attr + 2
        assert cont_input.size(0) == (self.seq_len - 2) * self.num_cont_attr + 2
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()  # (1, 1, seq_len)
        continuous_mask = (cont_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()
        continuous_mask = torch.cat([encoder_mask, continuous_mask], dim=-1)

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "continuous_input": cont_input,
            "decoder_input": decoder_input,  # (seq_len)
            # Mask indicating which tokens should participate in self-attention (excluding PAD tokens)
            "encoder_mask": encoder_mask,
            "continuous_mask": continuous_mask,
            # Mask indicating self-attention and causal mask (excluding PAD tokens)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "continuous_data": cont_data,
            "tgt_text": tgt_text
        }


def causal_mask(size: int) -> torch.Tensor:
    """
    Generate a causal mask for self-attention.

    :param size: Size of the mask.
    :return: Causal mask tensor.
    """
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
