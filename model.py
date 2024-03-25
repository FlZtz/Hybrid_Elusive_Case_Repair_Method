# model.py - Definition of Transformer model and related modules.
import math
from typing import Tuple

import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    """
    Layer normalization module.
    """
    def __init__(self, features: int, eps: float = 10 ** -6) -> None:
        """
        Initialize the layer normalization module.

        :param features: Number of features in the input tensor.
        :param eps: Epsilon value to avoid division by zero in normalization.
        """
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))  # Learnable scale parameter
        self.bias = nn.Parameter(torch.zeros(features))  # Learnable bias parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer normalization module.

        :param x: Input tensor of shape (batch_size, seq_len, features).
        :return: Normalized tensor.
        """
        mean = x.mean(dim=-1, keepdim=True)  # Compute mean along the feature dimension
        std = x.std(dim=-1, keepdim=True)  # Compute standard deviation along the feature dimension
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """
    Feedforward block module.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """
        Initialize the feedforward block module.

        :param d_model: Dimensionality of the model.
        :param d_ff: Dimensionality of the feedforward layer.
        :param dropout: Dropout probability.
        """
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # First linear transformation
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # Second linear transformation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feedforward block module.

        :param x: Input tensor of shape (batch_size, seq_len, d_model).
        :return: Output tensor.
        """
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class InputEmbeddings(nn.Module):
    """
    Input embeddings module.
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        Initialize the input embeddings module.

        :param d_model: Dimensionality of the model.
        :param vocab_size: Size of the vocabulary.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the input embeddings module.

        :param x: Input tensor of shape (batch_size, seq_len).
        :return: Output tensor of shape (batch_size, seq_len, d_model).
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Positional encoding module.
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        Initialize the positional encoding module.

        :param d_model: Dimensionality of the model.
        :param seq_len: Maximum sequence length.
        :param dropout: Dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Compute positional encodings
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the positional encoding module.

        :param x: Input tensor of shape (batch_size, seq_len, d_model).
        :return: Output tensor.
        """
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class ContinuousEmbedding(nn.Module):
    """
    Continuous embedding module.
    """
    def __init__(self, d_model: int, dropout: float) -> None:
        """
        Initialize the continuous embedding module.

        :param d_model: Dimensionality of the model.
        :param dropout: Dropout probability.
        """
        super().__init__()
        self.linear = nn.Linear(1, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the continuous embedding module.

        :param x: Input tensor of shape (batch_size, seq_len).
        :return: Output tensor of shape (batch_size, seq_len, d_model).
        """
        return self.dropout(self.linear(x.unsqueeze(-1).float()))


class ResidualConnection(nn.Module):
    """
    Residual connection module.
    """
    def __init__(self, features: int, dropout: float) -> None:
        """
        Initialize the residual connection module.

        :param features: Number of features in the input tensor.
        :param dropout: Dropout probability.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        Forward pass of the residual connection module.

        :param x: Input tensor.
        :param sublayer: Sublayer module.
        :return: Output tensor.
        """
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-head attention block module.
    """
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        """
        Initialize the multi-head attention block module.

        :param d_model: Dimensionality of the model.
        :param h: Number of heads.
        :param dropout: Dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor,
                  dropout: nn.Dropout) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the scaled dot-product attention.

        :param query: Query tensor of shape (batch_size, seq_len_q, d_k).
        :param key: Key tensor of shape (batch_size, seq_len_k, d_k).
        :param value: Value tensor of shape (batch_size, seq_len_v, d_v).
        :param mask: Mask tensor indicating which entries to mask, of shape (batch_size, 1, seq_len_k).
        :param dropout: Dropout layer.
        :return: Result of the attention operation; attention scores.
        """
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-head attention block.

        :param q: Query tensor of shape (batch_size, seq_len_q, d_model).
        :param k: Key tensor of shape (batch_size, seq_len_k, d_model).
        :param v: Value tensor of shape (batch_size, seq_len_v, d_model).
        :param mask: Mask tensor indicating which entries to mask, of shape (batch_size, 1, seq_len_k).
        :return: Result of the attention operation.
        """
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)


class EncoderBlock(nn.Module):
    """
    Encoder block module.
    """
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        """
        Initialize the encoder block module.

        :param features: Number of features in the input tensor.
        :param self_attention_block: Self-attention block module.
        :param feed_forward_block: Feedforward block module.
        :param dropout: Dropout probability.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder block.

        :param x: Input tensor of shape (batch_size, seq_len, d_model).
        :param src_mask: Mask tensor for source sequence, of shape (batch_size, 1, seq_len).
        :return: Output tensor of the encoder block.
        """
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    """
    Encoder module.
    """
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        """
        Initialize the encoder module.

        :param features: Number of features in the input tensor.
        :param layers: List of encoder blocks.
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder module.

        :param x: Input tensor of shape (batch_size, seq_len, d_model).
        :param mask: Mask tensor, of shape (batch_size, 1, seq_len).
        :return: Output tensor of the encoder module.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    """
    Decoder block module.
    """
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        """
        Initialize the decoder block module.

        :param features: Number of features in the input tensor.
        :param self_attention_block: Self-attention block module.
        :param cross_attention_block: Cross-attention block module.
        :param feed_forward_block: Feedforward block module.
        :param dropout: Dropout probability.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor,
                tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder block.

        :param x: Input tensor of shape (batch_size, seq_len, d_model).
        :param encoder_output: Encoder output tensor, of shape (batch_size, seq_len, d_model).
        :param src_mask: Mask tensor for source sequence, of shape (batch_size, 1, seq_len).
        :param tgt_mask: Mask tensor for target sequence, of shape (batch_size, seq_len, seq_len).
        :return: Output tensor of the decoder block.
        """
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output,
                                                                                 src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    """
    Decoder module.
    """
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        """
        Initialize the decoder module.

        :param features: Number of features in the input tensor.
        :param layers: List of decoder blocks.
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor,
                tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder module.

        :param x: Input tensor of shape (batch_size, seq_len, d_model).
        :param encoder_output: Encoder output tensor, of shape (batch_size, seq_len, d_model).
        :param src_mask: Mask tensor for source sequence, of shape (batch_size, 1, seq_len).
        :param tgt_mask: Mask tensor for target sequence, of shape (batch_size, seq_len, seq_len).
        :return: Output tensor of the decoder module.
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    """
    Projection layer module.
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        Initialize the projection layer module.

        :param d_model: Dimensionality of the model.
        :param vocab_size: Size of the vocabulary.
        """
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the projection layer module.

        :param x: Input tensor of shape (batch_size, seq_len, d_model).
        :return: Output tensor of shape (batch_size, seq_len, vocab_size).
        """
        return self.proj(x)


class Transformer(nn.Module):
    """
    Transformer module.
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, cont_embed: ContinuousEmbedding,
                 cont_ff_block: FeedForwardBlock, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        """
        Initialize the transformer module.

        :param encoder: Encoder module.
        :param decoder: Decoder module.
        :param src_embed: Source input embeddings module.
        :param cont_embed: Continuous input embeddings module.
        :param cont_ff_block: Continuous feedforward block module.
        :param tgt_embed: Target input embeddings module.
        :param src_pos: Source positional encoding module.
        :param tgt_pos: Target positional encoding module.
        :param projection_layer: Projection layer module.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.cont_embed = cont_embed
        self.cont_ff_block = cont_ff_block
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode the source sequence.

        :param src: Source input tensor.
        :param src_mask: Mask for source sequence.
        :return: Encoded source tensor.
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def cont_enrichment(self, encoder_output: torch.Tensor, cont_input: torch.Tensor) -> torch.Tensor:
        """
        Enrich the encoder output with continuous data.

        :param encoder_output: Encoder output tensor.
        :param cont_input: Continuous input tensor.
        :return: Enriched encoder output tensor.
        """
        cont = self.cont_embed(cont_input)
        cont = self.cont_ff_block(cont)
        return torch.cat((encoder_output, cont), dim=1)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor,
               tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Decode the target sequence.

        :param encoder_output: Encoder output tensor.
        :param src_mask: Mask for source sequence.
        :param tgt: Target input tensor.
        :param tgt_mask: Mask for target sequence.
        :return: Decoded output tensor.
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project the tensor to obtain logits.

        :param x: Input tensor.
        :return: Projected tensor.
        """
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int,
                      num_layers: int, h: int, dropout: float, d_ff: int) -> Transformer:
    """
    Build the transformer model.

    :param src_vocab_size: Size of the source vocabulary.
    :param tgt_vocab_size: Size of the target vocabulary.
    :param src_seq_len: Maximum sequence length for source.
    :param tgt_seq_len: Maximum sequence length for target.
    :param d_model: Dimensionality of the model.
    :param num_layers: Number of encoder and decoder layers.
    :param h: Number of attention heads.
    :param dropout: Dropout probability.
    :param d_ff: Dimensionality of the feedforward layer.
    :return: Built transformer model.
    """
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    encoder_blocks = []
    for _ in range(num_layers):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(num_layers):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block,
                                     feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    continuous_embed = ContinuousEmbedding(d_model, dropout)
    continuous_ff_block = FeedForwardBlock(d_model, d_ff, dropout)

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, continuous_embed, continuous_ff_block, tgt_embed, src_pos,
                              tgt_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
