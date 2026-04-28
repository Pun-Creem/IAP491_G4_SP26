"""
Configuration for Malware DL Action Recommendation System.

All hyperparameters are carefully tuned for:
- Small dataset (944 samples)
- Sparse multi-label classification
- Variable-length input sequences (TTPs + MBCs + Signatures)
"""

import os

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Training data paths
TRAIN_EXCEL_DIR = os.path.join(BASE_DIR, "train_excel")
TRAIN_REPORTS_DIR = os.path.join(BASE_DIR, "train_reports")
TRAIN_ACTIONS_DIR = os.path.join(BASE_DIR, "train_actions")  # D3FEND actions (from LLM)

# New data paths (for prediction)
NEW_EXCEL_DIR = os.path.join(BASE_DIR, "new_excel")
NEW_REPORTS_DIR = os.path.join(BASE_DIR, "new_reports")

# Output paths
SAVED_MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# =============================================================================
# D3FEND LABEL CONFIGURATION
# =============================================================================
D3FEND_MIN_SAMPLES = 3           # minimum samples an action must appear in to be kept

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
# Special tokens for vocabulary
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1

# Feature types (for type embedding)
TYPE_TTP = 0
TYPE_MBC = 1
TYPE_SIG = 2
NUM_TYPES = 3

# =============================================================================
# DATA AUGMENTATION
# =============================================================================
# Feature Dropout Augmentation: randomly mask features during training
# Helps prevent overfitting on small datasets by creating variations
# Only applied during training, disabled during validation/prediction
AUGMENT_DROP_RATE = 0.15          # drop 15% of features each epoch
                                   # too high → loses important info
                                   # too low → no effect

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
# Embedding dimensions
# Rule of thumb: embed_dim ~ sqrt(vocab_size) to 4x
# vocab ~ 426 features, sqrt(426) ~ 20, using 64 for expressiveness
# while keeping small enough to avoid overfitting on 944 samples
EMBED_DIM = 64

# Transformer Encoder configuration
# 2 layers only (BERT uses 12, but we have 944 samples vs BERT's billions)
# 4 attention heads (each head sees 64/4 = 16 dims)
NUM_TRANSFORMER_LAYERS = 2
NUM_ATTENTION_HEADS = 4
FEEDFORWARD_DIM = 256             # larger FFN for learning sparse patterns
TRANSFORMER_DROPOUT = 0.2         # lower dropout, sparse labels need more capacity

# Classification head
CLASSIFIER_HIDDEN_DIM = 64
CLASSIFIER_DROPOUT = 0.2

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
# Optimizer: AdamW (standard for Transformers, includes weight decay)
LEARNING_RATE = 5e-4              # lower LR for sparse labels (less aggressive)
WEIGHT_DECAY = 1e-4               # L2 regularization

# LR Schedule: Warmup + Cosine Annealing
WARMUP_EPOCHS = 10                # longer warmup for sparse labels
# After warmup, LR follows cosine decay to near zero

# Training loop
BATCH_SIZE = 32                   # 944/32 ~ 30 steps per epoch, good balance
MAX_EPOCHS = 150                  # more epochs needed for sparse labels
EARLY_STOPPING_PATIENCE = 20     # more patience for sparse learning

# Cross Validation
NUM_FOLDS = 5                     # 5-fold CV to maximize data usage
RANDOM_SEED = 42                  # reproducibility

# Label Smoothing: softens hard labels [0,1] → [0.01, 0.99]
# LLM labels are clean, only need minimal smoothing
LABEL_SMOOTHING = 0.02

# =============================================================================
# PREDICTION CONFIGURATION
# =============================================================================
# Threshold tuning: search range for optimal threshold per action
THRESHOLD_SEARCH_MIN = 0.05
THRESHOLD_SEARCH_MAX = 0.9
THRESHOLD_SEARCH_STEP = 0.05

# Output constraints
MAX_ACTIONS = 5                   # maximum actions to recommend per sample
MIN_ACTIONS = 1                   # always recommend at least 1 action

# =============================================================================
# MLP MODEL CONFIGURATION
# =============================================================================
# MLP is simpler than Transformer — better suited for small datasets
# Input: binary feature vector (vocab_size)
# Architecture: 3-layer MLP with BatchNorm and Dropout
MLP_HIDDEN_DIM = 128              # first hidden layer
MLP_DROPOUT = 0.3                 # higher dropout for small data
MLP_LEARNING_RATE = 1e-3          # higher LR than Transformer (simpler model)
MLP_WEIGHT_DECAY = 1e-4
MLP_WARMUP_EPOCHS = 5
MLP_MAX_EPOCHS = 200              # more epochs (each epoch is much faster)
MLP_EARLY_STOPPING_PATIENCE = 25
MLP_BATCH_SIZE = 32
MLP_LABEL_SMOOTHING = 0.02

# MLP output paths (separate from Transformer)
MLP_SAVED_MODEL_DIR = os.path.join(BASE_DIR, "saved_model_mlp")
MLP_OUTPUT_DIR = os.path.join(BASE_DIR, "output_mlp")

# =============================================================================
# EVALUATION METRICS
# =============================================================================
EVAL_TOP_K = [1, 3, 5]           # Top-K accuracy evaluation
