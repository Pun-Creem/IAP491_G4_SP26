"""
Training Logic: K-Fold CV, Early Stopping, Threshold Tuning.

Optimized for sparse multi-label classification (avg 3.2 actions out of 41).
Uses Focal Loss + pos_weight to handle severe class imbalance.
Early stopping based on validation loss (more stable than F1 with fixed threshold).
"""

import os
import copy
import json
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score

import config
from logger import get_logger
from model import MalwareActionPredictor
from data_loader import MalwareDataset


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def compute_pos_weight(labels):
    """Compute per-action positive weight for class imbalance."""
    labels_np = np.array(labels)
    n_samples = labels_np.shape[0]
    pos_counts = labels_np.sum(axis=0)
    neg_counts = n_samples - pos_counts
    pos_counts = np.maximum(pos_counts, 1)
    weights = neg_counts / pos_counts
    weights = np.clip(weights, 0.5, 200.0)
    return torch.tensor(weights, dtype=torch.float)


def smooth_labels(labels, smoothing=0.02):
    """Apply label smoothing."""
    return labels * (1 - smoothing) + smoothing / 2


class FocalLossWithLogits(nn.Module):
    """
    Focal Loss for severe class imbalance (sparse multi-label).
    Reduces loss for easy negatives, focuses on hard positives.
    """
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.pos_weight is not None:
            weight = targets * self.pos_weight.unsqueeze(0) + (1 - targets)
            loss = loss * weight

        return loss.mean()


# =============================================================================
# LR SCHEDULER
# =============================================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, dataloader, device):
    """Evaluate model, return metrics and raw predictions."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    n_batches = 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in dataloader:
            fids = batch["feature_ids"].to(device)
            tids = batch["type_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(fids, tids, mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(batch["labels"].numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    avg_loss = total_loss / max(n_batches, 1)

    return avg_loss, all_preds, all_labels


def compute_f1_at_threshold(all_preds, all_labels, threshold):
    """Compute macro F1 at a given threshold."""
    binary = (all_preds >= threshold).astype(int)
    return f1_score(all_labels, binary, average="macro", zero_division=0)


# =============================================================================
# THRESHOLD TUNING
# =============================================================================

def tune_thresholds(all_preds, all_labels):
    """Find optimal threshold per action and global."""
    n_actions = all_labels.shape[1]

    # Per-action thresholds
    per_action_thresholds = []
    for j in range(n_actions):
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(config.THRESHOLD_SEARCH_MIN, config.THRESHOLD_SEARCH_MAX + 0.01, config.THRESHOLD_SEARCH_STEP):
            preds_j = (all_preds[:, j] >= t).astype(int)
            f1 = f1_score(all_labels[:, j], preds_j, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        per_action_thresholds.append(best_t)

    # Global threshold
    best_global_t, best_global_f1 = 0.5, 0.0
    for t in np.arange(config.THRESHOLD_SEARCH_MIN, config.THRESHOLD_SEARCH_MAX + 0.01, config.THRESHOLD_SEARCH_STEP):
        binary = (all_preds >= t).astype(int)
        f1 = f1_score(all_labels, binary, average="macro", zero_division=0)
        if f1 > best_global_f1:
            best_global_f1 = f1
            best_global_t = t

    return {
        "global": float(best_global_t),
        "per_action": [float(t) for t in per_action_thresholds],
        "global_f1": float(best_global_f1),
    }


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_one_epoch(model, dataloader, optimizer, criterion, device, label_smoothing):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        fids = batch["feature_ids"].to(device)
        tids = batch["type_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if label_smoothing > 0:
            labels = smooth_labels(labels, label_smoothing)

        optimizer.zero_grad()
        logits = model(fids, tids, mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def _set_dataset_training(dataset, mode):
    """Toggle augmentation on the underlying dataset (works with Subset)."""
    if hasattr(dataset, 'set_training'):
        dataset.set_training(mode)
    elif hasattr(dataset, 'dataset'):  # Subset wraps original dataset
        if hasattr(dataset.dataset, 'set_training'):
            dataset.dataset.set_training(mode)


def train_fold(fold_idx, train_dataset, val_dataset, vocab_size, num_actions,
               pos_weight, device):
    """Train a single fold."""
    log = get_logger()
    log.info(f"\n{'='*60}")
    log.info(f"FOLD {fold_idx + 1}/{config.NUM_FOLDS}")
    log.info(f"{'='*60}")
    log.info(f"  Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = MalwareActionPredictor(
        vocab_size=vocab_size,
        num_actions=num_actions,
    ).to(device)

    criterion = FocalLossWithLogits(gamma=2.0, pos_weight=pos_weight.to(device))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = WarmupCosineScheduler(
        optimizer, config.WARMUP_EPOCHS, config.MAX_EPOCHS, config.LEARNING_RATE
    )

    # Early stopping based on VALIDATION LOSS (stable for sparse labels)
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(config.MAX_EPOCHS):
        lr = scheduler.step(epoch)

        _set_dataset_training(train_dataset, True)   # augmentation ON
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config.LABEL_SMOOTHING,
        )

        _set_dataset_training(train_dataset, False)   # augmentation OFF
        val_loss, val_preds, val_labels = evaluate_model(model, val_loader, device)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            # Compute F1 with tuned threshold for logging
            thresh_info = tune_thresholds(val_preds, val_labels)
            log.info(
                f"  Epoch {epoch+1:3d}/{config.MAX_EPOCHS} | "
                f"LR: {lr:.6f} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val F1 (tuned): {thresh_info['global_f1']:.4f} | "
                f"Thresh: {thresh_info['global']:.2f}"
            )

        # Early stopping on val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            log.info(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model and tune thresholds
    model.load_state_dict(best_model_state)
    val_loss, val_preds, val_labels = evaluate_model(model, val_loader, device)
    thresholds = tune_thresholds(val_preds, val_labels)
    best_val_f1 = thresholds["global_f1"]

    log.info(f"  Best Val Loss: {best_val_loss:.4f}")
    log.info(f"  Best Val F1 (tuned threshold): {best_val_f1:.4f}")
    log.info(f"  Optimal threshold: {thresholds['global']:.2f}")

    return model, best_val_f1, thresholds, {}


def train_kfold(dataset, vocab_size, num_actions,
                labels, device):
    """Full K-Fold Cross Validation."""
    log = get_logger()
    log.info(f"\n{'#'*60}")
    log.info(f"STARTING {config.NUM_FOLDS}-FOLD CROSS VALIDATION")
    log.info(f"{'#'*60}")

    pos_weight = compute_pos_weight(labels)
    log.info(f"Pos weight range: [{pos_weight.min():.1f}, {pos_weight.max():.1f}]")

    kfold = KFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)

    all_fold_f1s = []
    all_fold_thresholds = []
    best_overall_f1 = 0.0
    best_model = None
    best_thresholds = None

    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(range(len(dataset)))):
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        model, fold_f1, thresholds, _ = train_fold(
            fold_idx, train_subset, val_subset, vocab_size, num_actions,
            pos_weight, device,
        )

        all_fold_f1s.append(fold_f1)
        all_fold_thresholds.append(thresholds)

        if fold_f1 > best_overall_f1:
            best_overall_f1 = fold_f1
            best_model = copy.deepcopy(model)
            best_thresholds = thresholds

    mean_f1 = np.mean(all_fold_f1s)
    std_f1 = np.std(all_fold_f1s)

    log.info(f"\n{'#'*60}")
    log.info(f"CROSS VALIDATION RESULTS")
    log.info(f"{'#'*60}")
    for i, f1 in enumerate(all_fold_f1s):
        log.info(f"  Fold {i+1}: F1 = {f1:.4f}")
    log.info(f"  Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
    log.info(f"  Best Fold F1: {best_overall_f1:.4f}")

    # Average thresholds across folds
    n_actions = len(all_fold_thresholds[0]["per_action"])
    avg_per_action = [
        float(np.mean([ft["per_action"][j] for ft in all_fold_thresholds]))
        for j in range(n_actions)
    ]
    avg_global = float(np.mean([ft["global"] for ft in all_fold_thresholds]))

    final_thresholds = {"global": avg_global, "per_action": avg_per_action}

    return best_model, final_thresholds, {
        "fold_f1s": [float(f) for f in all_fold_f1s],
        "mean_f1": float(mean_f1),
        "std_f1": float(std_f1),
    }


# =============================================================================
# MLP TRAINING
# =============================================================================

def train_one_epoch_mlp(model, dataloader, optimizer, criterion, device, label_smoothing):
    """Train MLP for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)

        if label_smoothing > 0:
            labels = smooth_labels(labels, label_smoothing)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate_model_mlp(model, dataloader, device):
    """Evaluate MLP model."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    n_batches = 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(batch["labels"].numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    avg_loss = total_loss / max(n_batches, 1)

    return avg_loss, all_preds, all_labels


def train_fold_mlp(fold_idx, train_dataset, val_dataset, input_dim, num_actions,
                   pos_weight, device):
    """Train a single fold for MLP."""
    log = get_logger()
    log.info(f"\n{'='*60}")
    log.info(f"FOLD {fold_idx + 1}/{config.NUM_FOLDS}")
    log.info(f"{'='*60}")
    log.info(f"  Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=config.MLP_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.MLP_BATCH_SIZE, shuffle=False)

    from model import MLPBaseline
    model = MLPBaseline(
        input_dim=input_dim,
        num_actions=num_actions,
        hidden_dim=config.MLP_HIDDEN_DIM,
        dropout=config.MLP_DROPOUT,
    ).to(device)

    criterion = FocalLossWithLogits(gamma=2.0, pos_weight=pos_weight.to(device))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.MLP_LEARNING_RATE, weight_decay=config.MLP_WEIGHT_DECAY,
    )

    scheduler = WarmupCosineScheduler(
        optimizer, config.MLP_WARMUP_EPOCHS, config.MLP_MAX_EPOCHS, config.MLP_LEARNING_RATE
    )

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(config.MLP_MAX_EPOCHS):
        lr = scheduler.step(epoch)

        _set_dataset_training(train_dataset, True)   # augmentation ON
        train_loss = train_one_epoch_mlp(
            model, train_loader, optimizer, criterion, device, config.MLP_LABEL_SMOOTHING,
        )

        _set_dataset_training(train_dataset, False)   # augmentation OFF
        val_loss, val_preds, val_labels = evaluate_model_mlp(model, val_loader, device)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            thresh_info = tune_thresholds(val_preds, val_labels)
            log.info(
                f"  Epoch {epoch+1:3d}/{config.MLP_MAX_EPOCHS} | "
                f"LR: {lr:.6f} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val F1 (tuned): {thresh_info['global_f1']:.4f} | "
                f"Thresh: {thresh_info['global']:.2f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.MLP_EARLY_STOPPING_PATIENCE:
            log.info(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_model_state)
    val_loss, val_preds, val_labels = evaluate_model_mlp(model, val_loader, device)
    thresholds = tune_thresholds(val_preds, val_labels)
    best_val_f1 = thresholds["global_f1"]

    log.info(f"  Best Val Loss: {best_val_loss:.4f}")
    log.info(f"  Best Val F1 (tuned threshold): {best_val_f1:.4f}")
    log.info(f"  Optimal threshold: {thresholds['global']:.2f}")

    return model, best_val_f1, thresholds


def train_kfold_mlp(dataset, input_dim, num_actions, labels, device):
    """Full K-Fold Cross Validation for MLP."""
    log = get_logger()
    log.info(f"\n{'#'*60}")
    log.info(f"STARTING {config.NUM_FOLDS}-FOLD CROSS VALIDATION (MLP)")
    log.info(f"{'#'*60}")

    pos_weight = compute_pos_weight(labels)
    log.info(f"Pos weight range: [{pos_weight.min():.1f}, {pos_weight.max():.1f}]")

    kfold = KFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)

    all_fold_f1s = []
    all_fold_thresholds = []
    best_overall_f1 = 0.0
    best_model = None
    best_thresholds = None

    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(range(len(dataset)))):
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        model, fold_f1, thresholds = train_fold_mlp(
            fold_idx, train_subset, val_subset, input_dim, num_actions,
            pos_weight, device,
        )

        all_fold_f1s.append(fold_f1)
        all_fold_thresholds.append(thresholds)

        if fold_f1 > best_overall_f1:
            best_overall_f1 = fold_f1
            best_model = copy.deepcopy(model)
            best_thresholds = thresholds

    mean_f1 = np.mean(all_fold_f1s)
    std_f1 = np.std(all_fold_f1s)

    log.info(f"\n{'#'*60}")
    log.info(f"CROSS VALIDATION RESULTS (MLP)")
    log.info(f"{'#'*60}")
    for i, f1 in enumerate(all_fold_f1s):
        log.info(f"  Fold {i+1}: F1 = {f1:.4f}")
    log.info(f"  Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
    log.info(f"  Best Fold F1: {best_overall_f1:.4f}")

    n_actions = len(all_fold_thresholds[0]["per_action"])
    avg_per_action = [
        float(np.mean([ft["per_action"][j] for ft in all_fold_thresholds]))
        for j in range(n_actions)
    ]
    avg_global = float(np.mean([ft["global"] for ft in all_fold_thresholds]))

    final_thresholds = {"global": avg_global, "per_action": avg_per_action}

    return best_model, final_thresholds, {
        "fold_f1s": [float(f) for f in all_fold_f1s],
        "mean_f1": float(mean_f1),
        "std_f1": float(std_f1),
    }
