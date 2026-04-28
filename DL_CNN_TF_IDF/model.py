"""
Conv2D Model for Malware Action Prediction.

Input: TF-IDF weighted features reshaped to 2D image (22x22).
Features grouped by type (SIG → TTP → MBC) for spatial coherence.
Category weights (0.8/0.2/0.0) emphasize discriminative features.
"""

import torch
import torch.nn as nn
import config


class MalwareCNN(nn.Module):

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

        self.features = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(config.CNN_POOL_SIZE),

            nn.Conv2d(channels[0], channels[1], kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(config.CNN_POOL_SIZE),

            nn.Conv2d(channels[1], channels[2], kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[2], classifier_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(classifier_dim, num_actions),
        )

        self._init_weights()

    def _init_weights(self):
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
        x = self.features(x)
        logits = self.classifier(x)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
