"""
Configuration for CNN-based Malware Action Recommendation System.

Hyperparameters are tuned for:
- Small dataset (943 training samples)
- Sparse multi-label classification (~33 D3FEND actions)
- TF-IDF weighted feature input (TTPs + MBCs + Signatures) reshaped to 2D image

CNN sits between Transformer (too heavy) and MLP (too simple):
- Learns local feature co-occurrence patterns via convolution kernels
- Fewer parameters than Transformer → less overfitting risk
- Captures feature interactions that MLP misses

Input encoding:
- Features are first one-hot encoded (binary 0/1)
- Then transformed via TF-IDF (sklearn TfidfTransformer)
- Result: continuous values where rare features have higher weight
- This gives CNN richer pixel intensity compared to raw binary
"""

import os

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Training data paths
TRAIN_EXCEL_DIR = os.path.join(BASE_DIR, "train_excel")
TRAIN_REPORTS_DIR = os.path.join(BASE_DIR, "train_reports")
TRAIN_ACTIONS_DIR = os.path.join(BASE_DIR, "train_actions")

# New data paths (for prediction on 201 unseen malware)
NEW_EXCEL_DIR = os.path.join(BASE_DIR, "new_excel")
NEW_REPORTS_DIR = os.path.join(BASE_DIR, "new_reports")

# Ground truth for evaluation (optional)
GROUND_TRUTH_DIR = os.path.join(BASE_DIR, "ground_truth")

# Output paths
SAVED_MODEL_DIR = os.path.join(BASE_DIR, "saved_model_cnn")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_cnn")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# =============================================================================
# D3FEND LABEL CONFIGURATION
# =============================================================================
D3FEND_MIN_SAMPLES = 3  # action must appear in >= 3 samples to be kept

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
# Feature type identifiers (for grouping in image layout)
TYPE_TTP = 0
TYPE_MBC = 1
TYPE_SIG = 2

# Feature filtering thresholds
FEATURE_MIN_FREQ = 3        # remove features appearing in < 3 samples
FEATURE_MAX_FREQ_RATIO = 0.95  # remove features appearing in > 95% samples

# Feature category weights (matching thesis Table 28, Section 4.4.2)
# These weights scale TF-IDF values by feature type before reshaping to image.
# From thesis TF-IDF+Cosine optimal: signatures dominate (1.0), TTPs secondary (0.0),
# MBC codes contribute zero to behavioral similarity in this ransomware dataset.
WEIGHT_SIG = 1.0   # signature features weight (dominant)
WEIGHT_TTP = 0.0   # ATT&CK TTP features weight (secondary)
WEIGHT_MBC = 0.0   # MBC features weight (excluded, matching thesis finding)

# =============================================================================
# CNN IMAGE CONFIGURATION
# =============================================================================
# Features are arranged into a 2D grid ("image") for CNN input.
# Layout: Signatures grouped together, then TTPs, then MBCs.
# Image size computed dynamically: ceil(sqrt(vocab_size)).
# With ~478 features: 22x22 = 484, pad 6 zeros.

# =============================================================================
# CNN MODEL ARCHITECTURE
# =============================================================================
CNN_CHANNELS = [32, 64, 64]
CNN_KERNEL_SIZE = 3
CNN_POOL_SIZE = 2
CNN_CLASSIFIER_DIM = 96
CNN_DROPOUT = 0.4

# =============================================================================
# DATA AUGMENTATION
# =============================================================================
# Feature Dropout: randomly zero-out active features during training.
# Creates variation, acts as regularization for small datasets.
AUGMENT_DROP_RATE = 0.05  # minimal: TF-IDF values are meaningful, heavy dropout destroys info

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
LEARNING_RATE = 7e-4              # balanced between 5e-4 (too slow) and 1e-3 (too fast)
WEIGHT_DECAY = 7e-4               # balanced L2 regularization
BATCH_SIZE = 32                   # 943/32 ≈ 30 steps/epoch
MAX_EPOCHS = 200                  # CNN trains fast, more epochs OK
EARLY_STOPPING_PATIENCE = 25      # stop if no improvement for 25 epochs
WARMUP_EPOCHS = 5                 # short warmup (CNN converges faster)

# Cross Validation
NUM_FOLDS = 5
RANDOM_SEED = 42

# Label Smoothing
LABEL_SMOOTHING = 0.02  # minimal smoothing for LLM-generated labels

# =============================================================================
# PREDICTION CONFIGURATION
# =============================================================================
THRESHOLD_SEARCH_MIN = 0.05
THRESHOLD_SEARCH_MAX = 0.9
THRESHOLD_SEARCH_STEP = 0.05

MAX_ACTIONS = 5  # max actions per sample
MIN_ACTIONS = 1  # always recommend at least 1

# =============================================================================
# EVALUATION
# =============================================================================
EVAL_TOP_K = [1, 3, 5]
