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

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids shape:      (batch_size, seq_length)
        attention_mask shape: (batch_size, seq_length)
        output shape:         (batch_size, num_classes)
        """

        # if no mask is given, treat non-zero tokens as real words
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()

        # embedding lookup
        x = self.embedding(input_ids)

        # add position information
        x = self.positional_encoding(x)

        # PyTorch transformer expects True where padding should be ignored
        src_key_padding_mask = (attention_mask == 0)

        # contextualize token embeddings
        x = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )

        # masked mean pooling across the sequence
        mask = attention_mask.unsqueeze(-1).float()
        x = x * mask
        summed = x.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts

        # classify the whole sentence
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        return logits

