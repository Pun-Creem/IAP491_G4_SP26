"""
Data Loading and 2D Image Feature Engineering for CNN.

Pipeline:
    1. Read Excel files → extract TTPs + MBCs per sample
    2. Read CAPE JSON reports → extract Signature names
    3. Build vocabulary with frequency-based filtering
    4. Encode each sample as a binary vector (0/1 per feature)
    5. Apply TF-IDF weighting (rare features → higher values)
    6. Sort features by type (SIG → TTP → MBC) for spatial grouping
    7. Reshape TF-IDF vector into a 2D "image" for Conv2D input

TF-IDF encoding (matching thesis Section 3.2.4.2):
    - Binary one-hot matrix is transformed via sklearn TfidfTransformer
    - TF = 1 for present features (binary input, each feature appears at most once)
    - IDF = log(N / df) penalizes features appearing in many samples
    - Result: rare features get higher pixel intensity, common ones get lower
    - CNN sees a grayscale image with continuous values instead of binary black/white
"""

import json
import math
import os
import glob
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfTransformer

import config
from logger import get_logger


# =============================================================================
# EXCEL LOADING
# =============================================================================

def find_excel_file(directory):
    pattern = os.path.join(directory, "*.xlsx")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No .xlsx file found in {directory}")
    if len(files) > 1:
        get_logger().warning(f" Multiple .xlsx files in {directory}, using: {files[0]}")
    return files[0]


def load_excel(excel_dir):
    excel_path = find_excel_file(excel_dir)
    get_logger().info(f" Loading Excel: {excel_path}")

    df = pd.read_excel(excel_path)
    df.columns = [c.strip().lower() for c in df.columns]

    samples = []
    for _, row in df.iterrows():
        filename = str(row.get("filename", ""))
        sha256 = str(row.get("sha256", ""))
        raw_ttps = str(row.get("techniqueid", ""))
        ttps = [t.strip() for t in raw_ttps.split("\n") if t.strip()]
        raw_mbcs = str(row.get("mbc", ""))
        mbcs = [m.strip() for m in raw_mbcs.split("\n") if m.strip()]

        samples.append({
            "filename": filename,
            "sha256": sha256,
            "ttps": ttps,
            "mbcs": mbcs,
        })

    get_logger().info(f" Loaded {len(samples)} samples from Excel")
    return samples


# =============================================================================
# CAPE JSON SIGNATURE EXTRACTION
# =============================================================================

def extract_signatures_from_report(filepath):
    """Extract signature names from a CAPE JSON report."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError, UnicodeDecodeError):
        return []

    signatures = []
    for sig in data.get("signatures", []):
        name = sig.get("name", "")
        if name:
            signatures.append(f"SIG_{name}")

    return signatures


def extract_all_reports(reports_dir, filenames):
    if not os.path.isdir(reports_dir):
        get_logger().warning(f" Reports directory not found: {reports_dir}")
        return {}

    all_signatures = {}
    loaded = 0
    failed = 0

    from tqdm import tqdm
    for fname in tqdm(filenames, desc="Reading CAPE reports"):
        filepath = os.path.join(reports_dir, fname)
        if not os.path.exists(filepath):
            all_signatures[fname] = []
            failed += 1
            continue
        sigs = extract_signatures_from_report(filepath)
        all_signatures[fname] = sigs
        loaded += 1

    get_logger().info(f" Extracted signatures from {loaded} reports ({failed} failed/missing)")
    return all_signatures


# =============================================================================
# VOCABULARY WITH FREQUENCY FILTERING
# =============================================================================

def build_vocabulary(samples, all_signatures=None,
                     min_freq=None, max_freq_ratio=None):
    """
    Build feature vocabulary from training data.

    Features are sorted by type: Signatures first, then TTPs, then MBCs.
    This ordering is preserved when reshaping to 2D image so that
    features of the same type are spatially grouped.

    Returns:
        vocab: dict {feature_name: index} (0-based, no special tokens)
        feature_types: dict {feature_name: type_id}
        image_size: int (side length of square image)
    """
    min_freq = min_freq or config.FEATURE_MIN_FREQ
    max_freq_ratio = max_freq_ratio or config.FEATURE_MAX_FREQ_RATIO
    n_samples = len(samples)
    log = get_logger()

    feature_counter = Counter()
    feature_type_map = {}

    for s in samples:
        for ttp in s["ttps"]:
            feature_counter[ttp] += 1
            feature_type_map[ttp] = config.TYPE_TTP
        for mbc in s["mbcs"]:
            feature_counter[mbc] += 1
            feature_type_map[mbc] = config.TYPE_MBC

    if all_signatures:
        for fname, sigs in all_signatures.items():
            for sig in set(sigs):
                feature_counter[sig] += 1
                if sig not in feature_type_map:
                    feature_type_map[sig] = config.TYPE_SIG

    # Filter by frequency
    max_freq = int(n_samples * max_freq_ratio)
    filtered = set()
    removed_rare = 0
    removed_common = 0

    for feat, count in feature_counter.items():
        if count < min_freq:
            removed_rare += 1
        elif count > max_freq:
            removed_common += 1
        else:
            filtered.add(feat)

    log.info(f" Feature filtering:")
    log.info(f"  Total features found: {len(feature_counter)}")
    log.info(f"  Removed (too rare, <{min_freq}): {removed_rare}")
    log.info(f"  Removed (too common, >{max_freq_ratio*100:.0f}%): {removed_common}")
    log.info(f"  Kept: {len(filtered)}")

    # Sort: Signatures first, then TTPs, then MBCs (alphabetical within each group)
    sigs_sorted = sorted(f for f in filtered if feature_type_map[f] == config.TYPE_SIG)
    ttps_sorted = sorted(f for f in filtered if feature_type_map[f] == config.TYPE_TTP)
    mbcs_sorted = sorted(f for f in filtered if feature_type_map[f] == config.TYPE_MBC)
    sorted_features = sigs_sorted + ttps_sorted + mbcs_sorted

    vocab = {feat: idx for idx, feat in enumerate(sorted_features)}
    feature_types = {feat: feature_type_map[feat] for feat in sorted_features}

    vocab_size = len(vocab)
    image_size = math.ceil(math.sqrt(vocab_size))

    n_ttp = len(ttps_sorted)
    n_mbc = len(mbcs_sorted)
    n_sig = len(sigs_sorted)
    log.info(f"  Breakdown: TTPs={n_ttp}, MBCs={n_mbc}, Signatures={n_sig}")
    log.info(f"  Total features: {vocab_size}")
    log.info(f"  Image size: {image_size}x{image_size} = {image_size**2} "
             f"(padding {image_size**2 - vocab_size} zeros)")

    return vocab, feature_types, image_size


# =============================================================================
# BINARY VECTOR CONSTRUCTION
# =============================================================================

def build_binary_matrix(samples, vocab, all_signatures=None):
    """
    Build binary (one-hot) feature matrix from samples.

    Returns:
        binary_matrix: np.ndarray of shape (n_samples, vocab_size), dtype float32
    """
    vocab_size = len(vocab)
    matrix = np.zeros((len(samples), vocab_size), dtype=np.float32)

    for i, s in enumerate(samples):
        for ttp in s["ttps"]:
            if ttp in vocab:
                matrix[i, vocab[ttp]] = 1.0
        for mbc in s["mbcs"]:
            if mbc in vocab:
                matrix[i, vocab[mbc]] = 1.0
        if all_signatures:
            for sig in all_signatures.get(s["filename"], []):
                if sig in vocab:
                    matrix[i, vocab[sig]] = 1.0

    return matrix


# =============================================================================
# TF-IDF TRANSFORMATION
# =============================================================================

def fit_tfidf(binary_matrix):
    """
    Fit TF-IDF transformer on training binary matrix.

    Uses sklearn's TfidfTransformer (matching thesis Section 3.2.4.2):
    - Input: binary one-hot matrix (n_samples × vocab_size)
    - Output: TF-IDF weighted matrix (same shape, continuous values)
    - TF = 1 for present features (binary, so no frequency scaling needed)
    - IDF = log(N/df) + 1 (penalizes common features, boosts rare ones)
    - L2 normalization per sample (each row has unit norm)

    Returns:
        tfidf_transformer: fitted TfidfTransformer (save this for prediction)
        tfidf_matrix: transformed matrix
    """
    log = get_logger()

    transformer = TfidfTransformer(
        norm='l2',           # L2 normalize each sample vector
        use_idf=True,        # apply IDF weighting
        smooth_idf=True,     # add 1 to df to prevent zero division
        sublinear_tf=False,  # TF is already binary, no need for log(1+tf)
    )

    tfidf_matrix = transformer.fit_transform(binary_matrix).toarray().astype(np.float32)

    # Log statistics
    nonzero_vals = tfidf_matrix[tfidf_matrix > 0]
    log.info(f" TF-IDF transformation:")
    log.info(f"  Input: binary matrix {binary_matrix.shape}")
    log.info(f"  Output: TF-IDF matrix {tfidf_matrix.shape}")
    log.info(f"  Non-zero values: min={nonzero_vals.min():.4f}, "
             f"max={nonzero_vals.max():.4f}, mean={nonzero_vals.mean():.4f}")
    log.info(f"  Sparsity: {(tfidf_matrix == 0).sum() / tfidf_matrix.size * 100:.1f}%")

    return transformer, tfidf_matrix


def apply_tfidf(binary_matrix, transformer):
    """
    Apply pre-fitted TF-IDF transformer to new data (prediction).

    Args:
        binary_matrix: binary one-hot matrix of new samples
        transformer: fitted TfidfTransformer from training

    Returns:
        tfidf_matrix: transformed matrix
    """
    tfidf_matrix = transformer.transform(binary_matrix).toarray().astype(np.float32)
    get_logger().info(f" Applied TF-IDF: {binary_matrix.shape} → {tfidf_matrix.shape}")
    return tfidf_matrix


# =============================================================================
# FEATURE CATEGORY WEIGHTING
# =============================================================================

def apply_category_weights(tfidf_matrix, vocab, feature_types):
    """
    Scale TF-IDF values by feature category weight (matching thesis Section 4.4.2).

    Each feature's TF-IDF value is multiplied by its category weight:
        - Signatures × WEIGHT_SIG
        - TTPs × WEIGHT_TTP
        - MBCs × WEIGHT_MBC

    This emphasizes more discriminative feature groups (signatures, TTPs)
    and de-emphasizes less useful ones (MBCs) based on thesis findings.

    Args:
        tfidf_matrix: np.ndarray (n_samples, vocab_size)
        vocab: dict {feature_name: index}
        feature_types: dict {feature_name: type_id}

    Returns:
        weighted_matrix: np.ndarray (same shape)
    """
    log = get_logger()

    weight_map = {
        config.TYPE_SIG: config.WEIGHT_SIG,
        config.TYPE_TTP: config.WEIGHT_TTP,
        config.TYPE_MBC: config.WEIGHT_MBC,
    }

    # Build per-column weight vector
    vocab_size = len(vocab)
    col_weights = np.ones(vocab_size, dtype=np.float32)
    for feat, idx in vocab.items():
        ftype = feature_types.get(feat, config.TYPE_TTP)
        col_weights[idx] = weight_map.get(ftype, 1.0)

    # Apply weights
    weighted_matrix = tfidf_matrix * col_weights[np.newaxis, :]

    n_sig = sum(1 for v in feature_types.values() if v == config.TYPE_SIG)
    n_ttp = sum(1 for v in feature_types.values() if v == config.TYPE_TTP)
    n_mbc = sum(1 for v in feature_types.values() if v == config.TYPE_MBC)

    log.info(f" Category weights applied:")
    log.info(f"  Signatures ({n_sig} features): weight={config.WEIGHT_SIG}")
    log.info(f"  TTPs ({n_ttp} features): weight={config.WEIGHT_TTP}")
    log.info(f"  MBCs ({n_mbc} features): weight={config.WEIGHT_MBC}")

    # Log effective feature count (non-zero weight)
    active_features = int((col_weights > 0).sum())
    log.info(f"  Active features: {active_features}/{vocab_size} "
             f"({vocab_size - active_features} zeroed out)")

    return weighted_matrix


# =============================================================================
# PYTORCH DATASET — 2D IMAGE OUTPUT
# =============================================================================

class MalwareCNNDataset(Dataset):
    """
    Dataset that produces 2D TF-IDF weighted images for Conv2D.
    """

    def __init__(self, feature_matrix, image_size, labels=None):
        self.feature_matrix = feature_matrix
        self.vocab_size = feature_matrix.shape[1]
        self.image_size = image_size
        self.labels = labels
        self.training = False

        get_logger().info(
            f" CNN Dataset: {len(feature_matrix)} samples, "
            f"{self.vocab_size} features → {image_size}x{image_size} image"
        )

    def set_training(self, mode=True):
        self.training = mode

    def __len__(self):
        return len(self.feature_matrix)

    def _to_image(self, vec):
        padded = np.zeros(self.image_size * self.image_size, dtype=np.float32)
        padded[:self.vocab_size] = vec
        image = padded.reshape(1, self.image_size, self.image_size)
        return image

    def __getitem__(self, idx):
        vec = self.feature_matrix[idx].copy()

        if self.training and config.AUGMENT_DROP_RATE > 0:
            import random
            for i in range(len(vec)):
                if vec[i] > 0 and random.random() < config.AUGMENT_DROP_RATE:
                    vec[i] = 0.0

        image = self._to_image(vec)

        result = {
            "image": torch.tensor(image, dtype=torch.float),
        }

        if self.labels is not None:
            result["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return result
