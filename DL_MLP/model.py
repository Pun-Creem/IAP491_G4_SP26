"""
Deep Learning Model: Transformer Encoder for Malware Action Prediction.

Architecture:
    1. Feature Embedding + Type Embedding (TTP vs MBC vs Signature)
    2. Transformer Encoder (self-attention learns behavior interactions)
    3. Attention Pooling (aggregate variable-length → fixed vector)
    4. Classification Head (predict D3FEND actions + confidence)

Also includes MLP Baseline for comparison in thesis evaluation.
"""

import math
import torch
import torch.nn as nn
import config


class AttentionPooling(nn.Module):
    """
    Learned attention pooling over variable-length sequences.
    Better than mean pooling or CLS token for small datasets.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden_states, attention_mask):
        scores = self.attention(hidden_states).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = weights.masked_fill(attention_mask == 0, 0.0)
        pooled = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return pooled


class MalwareActionPredictor(nn.Module):
    """
    Main model: Transformer Encoder for malware action prediction.

    Categorical features (TTPs, MBCs, Signatures, API groups) go through
    embedding → transformer → attention pooling → classification head.
    """

    def __init__(self, vocab_size, num_actions,
                 embed_dim=None, num_heads=None, num_layers=None,
                 ff_dim=None, dropout=None):
        super().__init__()

        # Use config defaults
        embed_dim = embed_dim or config.EMBED_DIM
        num_heads = num_heads or config.NUM_ATTENTION_HEADS
        num_layers = num_layers or config.NUM_TRANSFORMER_LAYERS
        ff_dim = ff_dim or config.FEEDFORWARD_DIM
        dropout = dropout or config.TRANSFORMER_DROPOUT

        self.embed_dim = embed_dim
        self.num_actions = num_actions

        # Feature embedding (learnable vector for each TTP/MBC/Signature)
        self.feature_embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=config.PAD_IDX
        )

        # Type embedding (TTP=0, MBC=1, SIG=2 — no positional encoding)
        self.type_embedding = nn.Embedding(config.NUM_TYPES, embed_dim)

        # Embedding dropout
        self.embed_dropout = nn.Dropout(dropout)

        # Embedding layer norm
        self.embed_layer_norm = nn.LayerNorm(embed_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm (more stable for small datasets)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Attention pooling
        self.pool = AttentionPooling(embed_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, config.CLASSIFIER_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.CLASSIFIER_DROPOUT),
            nn.Linear(config.CLASSIFIER_HIDDEN_DIM, num_actions),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier/Kaiming initialization for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])

    def forward(self, feature_ids, type_ids, attention_mask):
        """
        Args:
            feature_ids: (batch, seq_len) vocab indices
            type_ids: (batch, seq_len) type indices
            attention_mask: (batch, seq_len) 1=real, 0=pad

        Returns:
            logits: (batch, num_actions) raw scores (before sigmoid)
        """
        # Embedding: feature + type (no positional, since no order)
        x = self.feature_embedding(feature_ids) + self.type_embedding(type_ids)
        x = self.embed_layer_norm(x)
        x = self.embed_dropout(x)

        # Transformer: key_padding_mask expects True for PAD positions
        key_padding_mask = (attention_mask == 0)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # Attention pooling → fixed-size vector
        pooled = self.pool(x, attention_mask)  # (batch, embed_dim)

        # Classification
        logits = self.classifier(pooled)  # (batch, num_actions)
        return logits


class MLPBaseline(nn.Module):
    """
    Simple MLP baseline for comparison.

    Uses binary feature vector (no attention, no interaction learning).
    Serves as lower bound in thesis evaluation.
    """

    def __init__(self, input_dim, num_actions, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_actions),
        )

    def forward(self, x):
        return self.model(x)
