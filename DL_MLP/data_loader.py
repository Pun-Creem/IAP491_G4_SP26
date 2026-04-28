"""
Data Loading and Feature Engineering.

Reads and extracts features from:
    1. Excel files (TTPs + MBCs per sample)
    2. CAPE JSON reports (Signature names only)

Automatic feature selection:
    - Filters by frequency (removes too rare and too common features)
    - User does not need to decide which features to use
"""

import json
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
# CAPE JSON FEATURE EXTRACTION
# =============================================================================

def extract_features_from_report(filepath):
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

    all_categorical = {}
    loaded = 0
    failed = 0

    from tqdm import tqdm
    for fname in tqdm(filenames, desc="Reading CAPE reports"):
        filepath = os.path.join(reports_dir, fname)
        if not os.path.exists(filepath):
            all_categorical[fname] = []
            failed += 1
            continue
        cat = extract_features_from_report(filepath)
        all_categorical[fname] = cat
        loaded += 1

    get_logger().info(f" Extracted features from {loaded} reports ({failed} failed/missing)")
    return all_categorical


# =============================================================================
# VOCABULARY WITH AUTO-FILTERING
# =============================================================================

def build_vocabulary(samples, all_categorical=None, min_freq=3, max_freq_ratio=0.95):
    n_samples = len(samples)
    feature_counter = Counter()
    feature_type_map = {}

    for s in samples:
        for ttp in s["ttps"]:
            feature_counter[ttp] += 1
            feature_type_map[ttp] = config.TYPE_TTP
        for mbc in s["mbcs"]:
            feature_counter[mbc] += 1
            feature_type_map[mbc] = config.TYPE_MBC

    if all_categorical:
        for fname, cats in all_categorical.items():
            for cat in set(cats):
                feature_counter[cat] += 1
                if cat not in feature_type_map:
                    feature_type_map[cat] = config.TYPE_SIG

    max_freq = int(n_samples * max_freq_ratio)
    filtered_features = set()
    removed_rare = 0
    removed_common = 0

    for feat, count in feature_counter.items():
        if count < min_freq:
            removed_rare += 1
        elif count > max_freq:
            removed_common += 1
        else:
            filtered_features.add(feat)

    get_logger().info(f" Feature filtering:")
    get_logger().info(f"  Total features found: {len(feature_counter)}")
    get_logger().info(f"  Removed (too rare, <{min_freq} samples): {removed_rare}")
    get_logger().info(f"  Removed (too common, >{max_freq_ratio*100:.0f}% samples): {removed_common}")
    get_logger().info(f"  Kept: {len(filtered_features)}")

    vocab = {config.PAD_TOKEN: config.PAD_IDX, config.UNK_TOKEN: config.UNK_IDX}
    feature_types = {}
    idx = 2

    for feat in sorted(filtered_features):
        vocab[feat] = idx
        feature_types[feat] = feature_type_map[feat]
        idx += 1

    n_ttp = sum(1 for f in filtered_features if feature_type_map.get(f) == config.TYPE_TTP)
    n_mbc = sum(1 for f in filtered_features if feature_type_map.get(f) == config.TYPE_MBC)
    n_sig = sum(1 for f in filtered_features if feature_type_map.get(f) == config.TYPE_SIG)
    get_logger().info(f"  Breakdown: TTPs={n_ttp}, MBCs={n_mbc}, Signatures={n_sig}")
    get_logger().info(f"  Vocab size (with special tokens): {len(vocab)}")

    return vocab, feature_types


# =============================================================================
# SAMPLE ENCODING
# =============================================================================

def encode_sample(sample, vocab, feature_types, all_categorical=None):
    feature_ids = []
    type_ids = []

    for ttp in sample["ttps"]:
        if ttp in vocab:
            feature_ids.append(vocab[ttp])
            type_ids.append(config.TYPE_TTP)

    for mbc in sample["mbcs"]:
        if mbc in vocab:
            feature_ids.append(vocab[mbc])
            type_ids.append(config.TYPE_MBC)

    if all_categorical:
        cats = all_categorical.get(sample["filename"], [])
        seen = set()
        for cat in cats:
            if cat in vocab and cat not in seen:
                feature_ids.append(vocab[cat])
                type_ids.append(config.TYPE_SIG)
                seen.add(cat)

    if not feature_ids:
        feature_ids = [config.UNK_IDX]
        type_ids = [config.TYPE_TTP]

    return feature_ids, type_ids


# =============================================================================
# PYTORCH DATASET
# =============================================================================

class MalwareDataset(Dataset):
    def __init__(self, samples, vocab, feature_types, all_categorical=None,
                 labels=None, max_length=None):
        self.samples = samples
        self.labels = labels
        self.training = False  # toggled by trainer

        self.encoded = []
        for s in samples:
            fids, tids = encode_sample(s, vocab, feature_types, all_categorical)
            self.encoded.append((fids, tids))

        if max_length is None:
            self.max_length = max(len(e[0]) for e in self.encoded)
        else:
            self.max_length = max_length

    def set_training(self, mode=True):
        """Enable/disable augmentation. Call before each epoch."""
        self.training = mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fids, tids = list(self.encoded[idx][0]), list(self.encoded[idx][1])
        fids = fids[:self.max_length]
        tids = tids[:self.max_length]

        seq_len = len(fids)

        # ── Feature Dropout Augmentation (training only) ──
        if self.training and config.AUGMENT_DROP_RATE > 0:
            import random
            for i in range(seq_len):
                if random.random() < config.AUGMENT_DROP_RATE:
                    fids[i] = config.PAD_IDX
                    tids[i] = 0

        attention_mask = [1 if fids[i] != config.PAD_IDX else 0 for i in range(seq_len)]
        attention_mask += [0] * (self.max_length - seq_len)
        fids = fids + [config.PAD_IDX] * (self.max_length - seq_len)
        tids = tids + [config.PAD_IDX] * (self.max_length - seq_len)

        result = {
            "feature_ids": torch.tensor(fids, dtype=torch.long),
            "type_ids": torch.tensor(tids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float),
        }

        if self.labels is not None:
            result["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return result

    def get_max_length(self):
        return self.max_length


# =============================================================================
# PYTORCH DATASET FOR MLP (Binary Feature Vector)
# =============================================================================

class MalwareBinaryDataset(Dataset):
    """
    Dataset for MLP model.

    Unlike MalwareDataset (which produces token sequences for Transformer),
    this produces a flat binary vector: 1 if feature is present, 0 if not.

    Input shape: (vocab_size,)
    """

    def __init__(self, samples, vocab, feature_types, all_categorical=None,
                 labels=None):
        self.samples = samples
        self.labels = labels
        self.vocab_size = len(vocab)
        self.training = False  # toggled by trainer

        # Build binary vectors
        self.binary_vectors = []
        for s in samples:
            vec = np.zeros(self.vocab_size, dtype=np.float32)

            # TTPs
            for ttp in s["ttps"]:
                if ttp in vocab:
                    vec[vocab[ttp]] = 1.0

            # MBCs
            for mbc in s["mbcs"]:
                if mbc in vocab:
                    vec[vocab[mbc]] = 1.0

            # Signature features from CAPE reports
            if all_categorical:
                cats = all_categorical.get(s["filename"], [])
                for cat in cats:
                    if cat in vocab:
                        vec[vocab[cat]] = 1.0

            self.binary_vectors.append(vec)

        self.binary_vectors = np.array(self.binary_vectors)

        get_logger().info(
            f"MLP Binary Dataset: {len(samples)} samples, "
            f"vocab={self.vocab_size}"
        )

    def get_input_dim(self):
        """Total input dimension for MLP."""
        return self.vocab_size

    def set_training(self, mode=True):
        """Enable/disable augmentation. Call before each epoch."""
        self.training = mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        binary = self.binary_vectors[idx].copy()  # copy to avoid modifying original

        # ── Feature Dropout Augmentation (training only) ──
        if self.training and config.AUGMENT_DROP_RATE > 0:
            import random
            for i in range(len(binary)):
                if binary[i] == 1.0 and random.random() < config.AUGMENT_DROP_RATE:
                    binary[i] = 0.0

        result = {
            "features": torch.tensor(binary, dtype=torch.float),
        }

        if self.labels is not None:
            result["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return result
