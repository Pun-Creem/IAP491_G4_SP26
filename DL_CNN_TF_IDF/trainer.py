"""
Training Logic for CNN: K-Fold CV, Focal Loss, Early Stopping, Threshold Tuning.

Optimized for sparse multi-label classification on small dataset.
Uses Focal Loss + pos_weight to handle class imbalance.
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
from model import MalwareCNN
from data_loader import MalwareCNNDataset


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
    """Apply label smoothing: [0,1] → [smoothing/2, 1-smoothing/2]."""
    return labels * (1 - smoothing) + smoothing / 2


class FocalLossWithLogits(nn.Module):
    """
    Focal Loss for sparse multi-label classification.
    Down-weights easy negatives, focuses training on hard positives.
    Critical for datasets where most labels are 0.
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
    """Warmup + Cosine Annealing LR schedule."""
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
# THRESHOLD TUNING
# =============================================================================

def find_optimal_thresholds(all_logits, all_labels, action_vocab):
    """
    Find per-action optimal threshold by searching for best F1.
    Uses validation predictions aggregated across folds.
    """
    log = get_logger()
    all_probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    all_labels = np.array(all_labels)
    n_actions = len(action_vocab)

    thresholds = np.full(n_actions, 0.5)

    for i in range(n_actions):
        best_f1 = -1
        best_t = 0.5

        for t in np.arange(config.THRESHOLD_SEARCH_MIN,
                           config.THRESHOLD_SEARCH_MAX,
                           config.THRESHOLD_SEARCH_STEP):
            preds = (all_probs[:, i] >= t).astype(int)
            if preds.sum() == 0:
                continue
            f1 = f1_score(all_labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        thresholds[i] = best_t

    log.info(f" Threshold range: [{thresholds.min():.2f}, {thresholds.max():.2f}], "
             f"mean={thresholds.mean():.2f}")

    return thresholds


# =============================================================================
# SINGLE FOLD TRAINING
# =============================================================================

def train_one_fold(model, train_loader, val_loader, device, fold_num,
                   pos_weight, total_folds):
    """Train one fold and return best model + validation predictions."""
    log = get_logger()

    # Loss
    pw = pos_weight.to(device)
    criterion = FocalLossWithLogits(gamma=2.0, pos_weight=pw)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Scheduler
    scheduler = WarmupCosineScheduler(
        optimizer, config.WARMUP_EPOCHS, config.MAX_EPOCHS, config.LEARNING_RATE
    )

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(config.MAX_EPOCHS):
        lr = scheduler.step(epoch)

        # ── Train ──
        model.train()
        if isinstance(train_loader.dataset, Subset):
            train_loader.dataset.dataset.set_training(True)
        else:
            train_loader.dataset.set_training(True)

        train_loss = 0
        train_steps = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)
            labels = smooth_labels(labels, config.LABEL_SMOOTHING)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

        avg_train_loss = train_loss / max(train_steps, 1)

        # ── Validate ──
        model.eval()
        if isinstance(val_loader.dataset, Subset):
            val_loader.dataset.dataset.set_training(False)
        else:
            val_loader.dataset.set_training(False)

        val_loss = 0
        val_steps = 0
        val_logits_list = []
        val_labels_list = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["labels"].to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                val_steps += 1
                val_logits_list.append(logits.cpu().numpy())
                val_labels_list.append(batch["labels"].numpy())

        avg_val_loss = val_loss / max(val_steps, 1)

        # Compute F1 at threshold=0.5 for logging
        val_logits_np = np.concatenate(val_logits_list, axis=0)
        val_labels_np = np.concatenate(val_labels_list, axis=0)
        val_probs = 1 / (1 + np.exp(-val_logits_np))  # sigmoid
        val_preds = (val_probs >= 0.5).astype(int)
        val_f1 = f1_score(val_labels_np, val_preds, average="macro", zero_division=0)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info(
                f"  Fold {fold_num}/{total_folds} | Epoch {epoch+1:3d} | "
                f"LR {lr:.6f} | Train Loss {avg_train_loss:.4f} | "
                f"Val Loss {avg_val_loss:.4f} | Val F1 {val_f1:.4f}"
            )

        # Early stopping on val loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                log.info(f"  Early stopping at epoch {epoch+1} (patience={config.EARLY_STOPPING_PATIENCE})")
                break

    # Load best model and get final validation predictions
    model.load_state_dict(best_model_state)
    model.eval()

    final_logits = []
    final_labels = []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            logits = model(images)
            final_logits.append(logits.cpu().numpy())
            final_labels.append(batch["labels"].numpy())

    final_logits = np.concatenate(final_logits, axis=0)
    final_labels = np.concatenate(final_labels, axis=0)

    return model, best_model_state, best_val_loss, final_logits, final_labels


# =============================================================================
# K-FOLD CROSS VALIDATION
# =============================================================================

def train_kfold(dataset, labels, image_size, num_actions, action_vocab):
    """
    Train CNN with K-Fold cross validation.

    Returns:
        fold_models: list of best model state_dicts
        thresholds: per-action optimal thresholds
        metrics: dict with aggregated evaluation metrics
    """
    log = get_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f" Device: {device}")

    pos_weight = compute_pos_weight(labels)

    kfold = KFold(
        n_splits=config.NUM_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_SEED,
    )

    fold_models = []
    all_val_logits = np.zeros((len(dataset), num_actions))
    all_val_labels = np.zeros((len(dataset), num_actions))
    fold_f1_scores = []

    indices = list(range(len(dataset)))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices), 1):
        log.info(f"\n{'='*60}")
        log.info(f" FOLD {fold}/{config.NUM_FOLDS} — Train: {len(train_idx)}, Val: {len(val_idx)}")
        log.info(f"{'='*60}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_subset, batch_size=config.BATCH_SIZE, shuffle=True,
            num_workers=0, drop_last=False,
        )
        val_loader = DataLoader(
            val_subset, batch_size=config.BATCH_SIZE, shuffle=False,
            num_workers=0,
        )

        # Create fresh model for each fold
        model = MalwareCNN(
            image_size=image_size,
            num_actions=num_actions,
        ).to(device)

        if fold == 1:
            log.info(f" Model parameters: {model.count_parameters():,}")

        model, best_state, best_loss, fold_logits, fold_labels = train_one_fold(
            model, train_loader, val_loader, device, fold, pos_weight,
            config.NUM_FOLDS,
        )

        # Store validation predictions at original indices
        all_val_logits[val_idx] = fold_logits
        all_val_labels[val_idx] = fold_labels

        # Fold-level metrics
        fold_probs = 1 / (1 + np.exp(-fold_logits))
        fold_preds = (fold_probs >= 0.5).astype(int)
        fold_f1 = f1_score(fold_labels, fold_preds, average="macro", zero_division=0)
        fold_f1_scores.append(fold_f1)

        log.info(f" Fold {fold} — Best Val Loss: {best_loss:.4f}, F1@0.5: {fold_f1:.4f}")
        fold_models.append(best_state)

    # ── Aggregate Results ──
    log.info(f"\n{'='*60}")
    log.info(f" K-FOLD RESULTS")
    log.info(f"{'='*60}")

    mean_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores)
    log.info(f" F1@0.5 per fold: {[f'{f:.4f}' for f in fold_f1_scores]}")
    log.info(f" Mean F1@0.5: {mean_f1:.4f} ± {std_f1:.4f}")

    # Find optimal thresholds using all validation predictions
    log.info(f"\n Tuning per-action thresholds...")
    thresholds = find_optimal_thresholds(all_val_logits, all_val_labels, action_vocab)

    # Evaluate with tuned thresholds
    all_probs = 1 / (1 + np.exp(-all_val_logits))
    tuned_preds = np.zeros_like(all_probs)
    for i in range(num_actions):
        tuned_preds[:, i] = (all_probs[:, i] >= thresholds[i]).astype(int)

    tuned_f1_macro = f1_score(all_val_labels, tuned_preds, average="macro", zero_division=0)
    tuned_f1_micro = f1_score(all_val_labels, tuned_preds, average="micro", zero_division=0)
    tuned_f1_samples = f1_score(all_val_labels, tuned_preds, average="samples", zero_division=0)
    tuned_precision = precision_score(all_val_labels, tuned_preds, average="macro", zero_division=0)
    tuned_recall = recall_score(all_val_labels, tuned_preds, average="macro", zero_division=0)

    log.info(f"\n With tuned thresholds:")
    log.info(f"  Macro F1:   {tuned_f1_macro:.4f}")
    log.info(f"  Micro F1:   {tuned_f1_micro:.4f}")
    log.info(f"  Sample F1:  {tuned_f1_samples:.4f}")
    log.info(f"  Precision:  {tuned_precision:.4f}")
    log.info(f"  Recall:     {tuned_recall:.4f}")

    # Per-action F1
    log.info(f"\n Per-action F1 (tuned threshold):")
    per_action_f1 = f1_score(all_val_labels, tuned_preds, average=None, zero_division=0)
    for i, aid in enumerate(action_vocab):
        support = int(all_val_labels[:, i].sum())
        log.info(f"  {aid}: F1={per_action_f1[i]:.4f} (t={thresholds[i]:.2f}, support={support})")

    metrics = {
        "fold_f1_scores": fold_f1_scores,
        "mean_f1_at_0.5": float(mean_f1),
        "std_f1_at_0.5": float(std_f1),
        "tuned_macro_f1": float(tuned_f1_macro),
        "tuned_micro_f1": float(tuned_f1_micro),
        "tuned_sample_f1": float(tuned_f1_samples),
        "tuned_precision": float(tuned_precision),
        "tuned_recall": float(tuned_recall),
        "per_action_f1": {aid: float(per_action_f1[i]) for i, aid in enumerate(action_vocab)},
    }

    return fold_models, thresholds, metrics


# =============================================================================
# SAVE MODEL ARTIFACTS
# =============================================================================

def save_model(fold_models, vocab, feature_types, image_size,
               action_vocab, action_info, thresholds, metrics,
               tfidf_transformer=None):
    """Save all artifacts needed for prediction."""
    log = get_logger()
    save_dir = config.SAVED_MODEL_DIR
    os.makedirs(save_dir, exist_ok=True)

    # Save each fold's model
    for i, state_dict in enumerate(fold_models):
        torch.save(state_dict, os.path.join(save_dir, f"model_fold_{i+1}.pt"))

    # Save vocabulary and config
    with open(os.path.join(save_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)

    with open(os.path.join(save_dir, "feature_types.json"), "w") as f:
        json.dump({k: int(v) for k, v in feature_types.items()}, f, indent=2)

    with open(os.path.join(save_dir, "action_vocab.json"), "w") as f:
        json.dump(action_vocab, f, indent=2)

    with open(os.path.join(save_dir, "action_info.json"), "w") as f:
        json.dump(action_info, f, indent=2, ensure_ascii=False)

    np.save(os.path.join(save_dir, "thresholds.npy"), thresholds)

    # Save TF-IDF transformer
    if tfidf_transformer is not None:
        import joblib
        joblib.dump(tfidf_transformer, os.path.join(save_dir, "tfidf_transformer.pkl"))
        log.info(f" TF-IDF transformer saved")

    model_config = {
        "image_size": image_size,
        "num_actions": len(action_vocab),
        "channels": config.CNN_CHANNELS,
        "kernel_size": config.CNN_KERNEL_SIZE,
        "classifier_dim": config.CNN_CLASSIFIER_DIM,
        "dropout": config.CNN_DROPOUT,
    }
    with open(os.path.join(save_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"\n Model saved to: {save_dir}")
    log.info(f" Files: {os.listdir(save_dir)}")
