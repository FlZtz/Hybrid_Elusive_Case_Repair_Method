# dataset.py - Definition of BilingualDataset class for handling bilingual text data and related utilities.
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class BilingualDataset(Dataset):
    """
    Dataset class for bilingual text data.
    """
    def __init__(self, ds: Dataset, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, tf_input: str, tf_output: str,
                 seq_len: int, disc_attr: int) -> None:
        """
        Initialize the dataset.

        :param ds: List of dictionaries containing source and target text pairs.
        :param tokenizer_src: Source language tokenizer.
        :param tokenizer_tgt: Target language tokenizer.
        :param tf_input: Key indicating the source text in the dictionary.
        :param tf_output: Key indicating the target text in the dictionary.
        :param seq_len: Maximum sequence length for inputs and outputs.
        :param disc_attr: Number of discrete input attributes.
        """
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.tf_input = tf_input
        self.tf_output = tf_output
        self.disc_attr = disc_attr

        # Special tokens for start of sequence (SOS), end of sequence (EOS), and padding (PAD)
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self) -> int:
        """
        Get the number of items in the dataset.

        :return: Number of items in the dataset.
        """
        try:
            return len(self.ds)
        except TypeError:
            raise TypeError("Dataset object does not support length operation.")

    def __getitem__(self, idx: int) -> dict:
        """
        Generate a single training example.

        :param idx: Index of the example in the dataset.
        :return: Dictionary containing encoder input, decoder input, encoder mask, decoder mask, label,
        source text, and target text.
        """
        src_target_pair = self.ds[idx]
        src_text = src_target_pair[self.tf_input]
        tgt_text = src_target_pair[self.tf_output]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Calculate the number of padding tokens to add
        # Accounting for SOS and EOS tokens
        enc_num_padding_tokens = (self.seq_len - 2) * self.disc_attr - len(enc_input_tokens)
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # Accounting for SOS token only

        # Ensure the length of the sequences does not exceed the specified maximum
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

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
        assert encoder_input.size(0) == (self.seq_len - 2) * self.disc_attr + 2
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            # Mask indicating which tokens should participate in self-attention (excluding PAD tokens)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
            # Mask indicating self-attention and causal mask (excluding PAD tokens)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size: int) -> torch.Tensor:
    """
    Generate a causal mask for self-attention.

    :param size: Size of the mask.
    :return: Causal mask tensor.
    """
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
