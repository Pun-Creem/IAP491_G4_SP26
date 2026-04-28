"""
Visualize Malware Feature Images (Binary version, no TF-IDF).

Generates 22x22 binary (0/1) images for 20 sample training + 20 sample test.
Same 20 samples as V5 TF-IDF version for direct comparison.

Usage:
    python visualize_images.py

Output:
    output_cnn/images/
        train/           -- 20 ảnh training samples
        test/            -- 20 ảnh test samples
        grid_train.png   -- 6 mẫu train (2x3)
        grid_test.png    -- 6 mẫu test (2x3)
"""

import os
import math
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from logger import setup_logger, get_logger
from data_loader import load_excel, extract_all_reports, build_vocabulary

N_SAMPLES = 20
N_GRID = 6


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def build_binary_vectors(samples, vocab, all_signatures):
    """Build binary (0/1) feature vectors — same logic as MalwareCNNDataset."""
    vocab_size = len(vocab)
    vectors = []
    for s in samples:
        vec = np.zeros(vocab_size, dtype=np.float32)
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
        vectors.append(vec)
    return np.array(vectors)


def vector_to_image(vec, image_size):
    padded = np.zeros(image_size * image_size, dtype=np.float32)
    padded[:len(vec)] = vec
    return padded.reshape(image_size, image_size)


def save_single_image(image, title, save_path, vmax_fixed):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    im = ax.imshow(image, cmap='inferno', vmin=0, vmax=vmax_fixed,
                   interpolation='nearest', aspect='equal')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Feature Presence (Binary)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def save_grid(images, titles, save_path, grid_title, vmax_fixed):
    n = len(images)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 9))

    if rows == 1:
        axes = np.array([axes])

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            ax = axes[i][j]
            if idx < n:
                im = ax.imshow(images[idx], cmap='inferno', vmin=0, vmax=vmax_fixed,
                               interpolation='nearest', aspect='equal')
                ax.set_title(titles[idx], fontsize=11, fontweight='bold', pad=8)
            else:
                ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])

    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label('Feature Presence (Binary)', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(grid_title, fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    set_seed(config.RANDOM_SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    setup_logger(log_dir=config.LOG_DIR, mode="visualize")
    log = get_logger()

    log.info("=" * 60)
    log.info(" VISUALIZE -- Malware Feature Images (Binary, no TF-IDF)")
    log.info("=" * 60)

    img_dir = os.path.join(config.OUTPUT_DIR, "images")
    os.makedirs(os.path.join(img_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(img_dir, "test"), exist_ok=True)

    # ── Load training data ──
    log.info("\n[Step 1] Loading training data...")
    train_samples = load_excel(config.TRAIN_EXCEL_DIR)
    train_filenames = [s["filename"] for s in train_samples]
    train_signatures = extract_all_reports(config.TRAIN_REPORTS_DIR, train_filenames)

    vocab, feature_types, image_size = build_vocabulary(
        train_samples, train_signatures,
        min_freq=config.FEATURE_MIN_FREQ,
        max_freq_ratio=config.FEATURE_MAX_FREQ_RATIO,
    )

    # ── Build binary vectors ──
    log.info("\n[Step 2] Building binary vectors (training)...")
    train_vectors = build_binary_vectors(train_samples, vocab, train_signatures)
    log.info(f"  Shape: {train_vectors.shape}, values: 0 or 1")

    # Binary vmax = 1.0 (fixed, all images use same scale)
    global_vmax = 1.0

    # ── Generate training images ──
    log.info(f"\n[Step 3] Generating {N_SAMPLES} training images...")
    train_images = []
    train_titles = []
    for i in range(N_SAMPLES):
        fname = train_samples[i]["filename"].replace(".json", "")
        img = vector_to_image(train_vectors[i], image_size)
        save_single_image(img, fname, os.path.join(img_dir, "train", f"{fname}.png"), global_vmax)
        train_images.append(img)
        train_titles.append(fname)
        n_active = int(train_vectors[i].sum())
        log.info(f"  {fname}: {n_active} active features")

    save_grid(train_images[:N_GRID], train_titles[:N_GRID],
              os.path.join(img_dir, "grid_train.png"),
              "Training Samples -- Malware Feature Images (22x22, Binary)",
              global_vmax)
    log.info(f"  Saved grid_train.png ({N_GRID} samples)")

    # ── Load test data ──
    log.info(f"\n[Step 4] Loading test data...")
    test_samples = load_excel(config.NEW_EXCEL_DIR)
    test_filenames = [s["filename"] for s in test_samples]
    test_signatures = extract_all_reports(config.NEW_REPORTS_DIR, test_filenames)

    # ── Build test binary vectors (using training vocab) ──
    log.info("\n[Step 5] Building binary vectors (test)...")
    test_vectors = build_binary_vectors(test_samples, vocab, test_signatures)
    log.info(f"  Shape: {test_vectors.shape}, values: 0 or 1")

    # ── Generate test images ──
    log.info(f"\n[Step 6] Generating {N_SAMPLES} test images...")
    test_images = []
    test_titles = []
    for i in range(N_SAMPLES):
        fname = test_samples[i]["filename"].replace(".json", "")
        img = vector_to_image(test_vectors[i], image_size)
        save_single_image(img, fname, os.path.join(img_dir, "test", f"{fname}.png"), global_vmax)
        test_images.append(img)
        test_titles.append(fname)
        n_active = int(test_vectors[i].sum())
        log.info(f"  {fname}: {n_active} active features")

    save_grid(test_images[:N_GRID], test_titles[:N_GRID],
              os.path.join(img_dir, "grid_test.png"),
              "Test Samples -- Malware Feature Images (22x22, Binary)",
              global_vmax)
    log.info(f"  Saved grid_test.png ({N_GRID} samples)")

    # ── Summary ──
    log.info(f"\n{'='*60}")
    log.info(f" COMPLETE")
    log.info(f"  Train: {N_SAMPLES} individual + grid ({N_GRID} samples)")
    log.info(f"  Test:  {N_SAMPLES} individual + grid ({N_GRID} samples)")
    log.info(f"  Image size: {image_size}x{image_size}")
    log.info(f"  Colormap: inferno")
    log.info(f"  Values: Binary (0 = absent, 1 = present)")
    log.info(f"  No TF-IDF, no category weights")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
