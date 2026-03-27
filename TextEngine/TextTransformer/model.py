import torch
import torch.nn as nn

from transformer import PositionalEncoding, TextTransformerEncoder


class EmotionTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        num_classes=28,
        max_length=64,
        dropout=0.1
    ):
        super().__init__()

        # turns token IDs into learned vectors
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=0
        )

        # adds position info to embeddings
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_length=max_length
        )

        # PyTorch transformer encoder wrapper from transformer.py
        self.transformer_encoder = TextTransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout
        )

        # dropout before classification
        self.dropout = nn.Dropout(dropout)

        # final classifier: one logit per emotion
        self.classifier = nn.Linear(d_model, num_classes)
