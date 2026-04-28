"""
CNN Model for Malware Action Prediction.

Architecture designed for:
    - Small dataset (943 samples) → keep parameter count low
    - Binary sparse input (most pixels are 0) → BatchNorm helps
    - Multi-label output (33 actions) → sigmoid per action

Model: 3-layer CNN with BatchNorm + Dropout
    Input:  (batch, 1, H, W)  — single-channel binary "image"
    Conv1:  (batch, 32, H/2, W/2)   — detect low-level feature co-occurrences
    Conv2:  (batch, 64, H/4, W/4)   — combine into higher-level patterns
    Conv3:  (batch, 64, 1, 1)       — global pattern via AdaptiveAvgPool
    FC:     (batch, 128) → (batch, num_actions)

Total parameters: ~60-80K (appropriate for 943 samples).
Compare: Transformer ~200K+, MLP ~30K.
"""

import torch
import torch.nn as nn

import config


class MalwareCNN(nn.Module):
    """
    Convolutional Neural Network for malware behavioral pattern recognition.

    Treats the binary feature vector (TTPs + MBCs + Signatures) as a
    single-channel grayscale image. CNN kernels learn to detect spatial
    co-occurrence patterns among grouped features.
    """

    def __init__(self, image_size, num_actions,
                 channels=None, kernel_size=None,
                 classifier_dim=None, dropout=None):
        super().__init__()

        channels = channels or config.CNN_CHANNELS
        kernel_size = kernel_size or config.CNN_KERNEL_SIZE
        classifier_dim = classifier_dim or config.CNN_CLASSIFIER_DIM
        dropout = dropout or config.CNN_DROPOUT

        self.image_size = image_size
        self.num_actions = num_actions

        # ── Convolutional Feature Extractor ──
        # Each block: Conv2D → BatchNorm → ReLU → MaxPool
        # BatchNorm is critical for binary sparse inputs (stabilizes gradients)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, channels[0], kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(config.CNN_POOL_SIZE),

            # Block 2
            nn.Conv2d(channels[0], channels[1], kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(config.CNN_POOL_SIZE),

            # Block 3
            nn.Conv2d(channels[1], channels[2], kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),

            # Adaptive pooling → fixed output size regardless of input size
            nn.AdaptiveAvgPool2d(1),
        )

        # ── Classification Head ──
        # Dropout → Dense → ReLU → Dropout → Output
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[2], classifier_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),  # lighter dropout on second layer
            nn.Linear(classifier_dim, num_actions),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (batch, 1, H, W) binary feature image

        Returns:
            logits: (batch, num_actions) raw scores before sigmoid
        """
        x = self.features(x)
        logits = self.classifier(x)
        return logits

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
