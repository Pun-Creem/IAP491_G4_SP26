"""
Visualize Malware Feature Images (TF-IDF + Category Weights version).

Format follows malware visualization papers (Nataraj et al. 2011,
Cui et al. 2018): clean grayscale images, no axis ticks,
colorbar showing intensity, grid layout for comparison.

Usage:
    python visualize_images.py

Output:
    output_cnn/images/
        train/           -- 20 ảnh riêng lẻ training samples
        test/            -- 20 ảnh riêng lẻ test samples
        grid_train.png   -- 6 mẫu train so sánh (2x3)
        grid_test.png    -- 6 mẫu test so sánh (2x3)
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
from data_loader import (
    load_excel, extract_all_reports, build_vocabulary,
    build_binary_matrix, fit_tfidf, apply_tfidf, apply_category_weights
)

N_SAMPLES = 20   # số ảnh riêng lẻ
N_GRID = 6       # số ảnh trong grid (2x3)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def vector_to_image(vec, image_size):
    padded = np.zeros(image_size * image_size, dtype=np.float32)
    padded[:len(vec)] = vec
    return padded.reshape(image_size, image_size)


def save_single_image(image, title, save_path, vmax_fixed):
    """Lưu 1 ảnh riêng lẻ với colorbar cố định."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    im = ax.imshow(image, cmap='inferno', vmin=0, vmax=vmax_fixed,
                   interpolation='nearest', aspect='equal')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('TF-IDF Intensity', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def save_grid(images, titles, save_path, grid_title, vmax_fixed):
    """Lưu grid so sánh nhiều mẫu (2 hàng x 3 cột), theo format paper."""
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

    # 1 colorbar chung cho toàn grid
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label('TF-IDF Intensity', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(grid_title, fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    set_seed(config.RANDOM_SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    setup_logger(log_dir=config.LOG_DIR, mode="visualize")
    log = get_logger()

    log.info("=" * 60)
    log.info(" VISUALIZE -- Malware Feature Images (TF-IDF + Weights)")
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

    # ── Build training features ──
    log.info("\n[Step 2] Building training features...")
    train_binary = build_binary_matrix(train_samples, vocab, train_signatures)
    tfidf_transformer, train_tfidf = fit_tfidf(train_binary)
    train_weighted = apply_category_weights(train_tfidf, vocab, feature_types)

    # ── Generate training images ──
    log.info(f"\n[Step 3] Generating {N_SAMPLES} training images...")

    # Tính global vmax từ toàn bộ training data (dùng chung cho tất cả ảnh)
    global_vmax = float(train_weighted.max())
    log.info(f"  Global vmax (from training): {global_vmax:.4f}")

    train_images = []
    train_titles = []
    for i in range(N_SAMPLES):
        fname = train_samples[i]["filename"].replace(".json", "")
        img = vector_to_image(train_weighted[i], image_size)
        save_single_image(img, fname, os.path.join(img_dir, "train", f"{fname}.png"), global_vmax)
        train_images.append(img)
        train_titles.append(fname)
        n_active = int((train_weighted[i] > 0).sum())
        log.info(f"  {fname}: {n_active} active features")

    save_grid(train_images[:N_GRID], train_titles[:N_GRID],
              os.path.join(img_dir, "grid_train.png"),
              "Training Samples -- Malware Feature Images (22x22, TF-IDF + Category Weights)",
              global_vmax)
    log.info(f"  Saved grid_train.png ({N_GRID} samples)")

    # ── Load test data ──
    log.info(f"\n[Step 4] Loading test data...")
    test_samples = load_excel(config.NEW_EXCEL_DIR)
    test_filenames = [s["filename"] for s in test_samples]
    test_signatures = extract_all_reports(config.NEW_REPORTS_DIR, test_filenames)

    # ── Build test features ──
    log.info("\n[Step 5] Building test features...")
    test_binary = build_binary_matrix(test_samples, vocab, test_signatures)
    test_tfidf = apply_tfidf(test_binary, tfidf_transformer)
    test_weighted = apply_category_weights(test_tfidf, vocab, feature_types)

    # ── Generate test images ──
    log.info(f"\n[Step 6] Generating {N_SAMPLES} test images...")
    test_images = []
    test_titles = []
    for i in range(N_SAMPLES):
        fname = test_samples[i]["filename"].replace(".json", "")
        img = vector_to_image(test_weighted[i], image_size)
        save_single_image(img, fname, os.path.join(img_dir, "test", f"{fname}.png"), global_vmax)
        test_images.append(img)
        test_titles.append(fname)
        n_active = int((test_weighted[i] > 0).sum())
        log.info(f"  {fname}: {n_active} active features")

    save_grid(test_images[:N_GRID], test_titles[:N_GRID],
              os.path.join(img_dir, "grid_test.png"),
              "Test Samples -- Malware Feature Images (22x22, TF-IDF + Category Weights)",
              global_vmax)
    log.info(f"  Saved grid_test.png ({N_GRID} samples)")

    # ── Summary ──
    log.info(f"\n{'='*60}")
    log.info(f" COMPLETE")
    log.info(f"  Train: {N_SAMPLES} individual + grid ({N_GRID} samples)")
    log.info(f"  Test:  {N_SAMPLES} individual + grid ({N_GRID} samples)")
    log.info(f"  Image size: {image_size}x{image_size}")
    log.info(f"  Colormap: inferno (dark=absent, bright=important)")
    log.info(f"  Weights: w_sig={config.WEIGHT_SIG}, w_ttp={config.WEIGHT_TTP}, w_mbc={config.WEIGHT_MBC}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
