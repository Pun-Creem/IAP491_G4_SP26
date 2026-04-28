"""
Data Loading and 2D Image Feature Engineering for CNN.

Pipeline:
    1. Read Excel files → extract TTPs + MBCs per sample
    2. Read CAPE JSON reports → extract Signature names
    3. Build vocabulary with frequency-based filtering
    4. Encode each sample as a binary vector (0/1 per feature)
    5. Sort features by type (SIG → TTP → MBC) for spatial grouping
    6. Reshape binary vector into a 2D "image" for Conv2D input

Why reshape to 2D image?
    - CNN kernels slide over local neighborhoods
    - Grouping features by type (SIG/TTP/MBC) means kernels can detect
      co-occurrence patterns within each feature group
    - A 3x3 kernel over the SIG region learns "which signatures appear together"
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

    # Compute image size (smallest square that fits all features)
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
# PYTORCH DATASET — 2D IMAGE OUTPUT
# =============================================================================

class MalwareCNNDataset(Dataset):
    """
    Dataset that produces 2D binary images for CNN.

    Each sample is a binary feature vector (0/1) reshaped into a
    single-channel image of size (1, H, W).

    Feature layout in the image (row-major order):
        [SIG_1, SIG_2, ..., TTP_1, TTP_2, ..., MBC_1, MBC_2, ..., 0, 0, ...]
        → reshaped to (image_size, image_size)

    This means CNN kernels in the early rows "see" signature co-occurrences,
    middle rows see TTP patterns, later rows see MBC patterns,
    and boundary regions capture cross-type interactions.
    """

    def __init__(self, samples, vocab, feature_types, image_size,
                 all_signatures=None, labels=None):
        self.samples = samples
        self.labels = labels
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.image_size = image_size
        self.training = False

        # Build binary vectors
        self.binary_vectors = []
        for s in samples:
            vec = np.zeros(self.vocab_size, dtype=np.float32)

            for ttp in s["ttps"]:
                if ttp in vocab:
                    vec[vocab[ttp]] = 1.0
            for mbc in s["mbcs"]:
                if mbc in vocab:
                    vec[vocab[mbc]] = 1.0
            if all_signatures:
                for sig in all_signatures.get(s["filename"], []):
                    if sig in vocab:
                        vec[vocab[sig]] = 1.0

            self.binary_vectors.append(vec)

        self.binary_vectors = np.array(self.binary_vectors)

        get_logger().info(
            f" CNN Dataset: {len(samples)} samples, "
            f"{self.vocab_size} features → {image_size}x{image_size} image"
        )

    def set_training(self, mode=True):
        self.training = mode

    def __len__(self):
        return len(self.samples)

    def _to_image(self, binary_vec):
        """Reshape binary vector to 2D image with zero-padding."""
        padded = np.zeros(self.image_size * self.image_size, dtype=np.float32)
        padded[:self.vocab_size] = binary_vec
        image = padded.reshape(1, self.image_size, self.image_size)  # (C, H, W)
        return image

    def __getitem__(self, idx):
        binary = self.binary_vectors[idx].copy()

        # Feature Dropout Augmentation (training only)
        if self.training and config.AUGMENT_DROP_RATE > 0:
            import random
            for i in range(len(binary)):
                if binary[i] == 1.0 and random.random() < config.AUGMENT_DROP_RATE:
                    binary[i] = 0.0

        image = self._to_image(binary)

        result = {
            "image": torch.tensor(image, dtype=torch.float),
        }

        if self.labels is not None:
            result["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return result
