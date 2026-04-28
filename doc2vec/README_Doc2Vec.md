# Doc2Vec — Malware Behavioral Similarity

A Doc2Vec-based pipeline for computing behavioral similarity between malware samples. Each malware is represented as a "document" of CAPE sandbox signature tokens, embedded into a 100-dimensional vector space using Gensim's Doc2Vec (DBOW mode). New/unseen malware samples are compared against the training corpus via cosine similarity to find the most behaviorally similar known samples.

## Overview

This model is part of the capstone thesis *"Behavior Similarity Based Malware Response Action Recommendation"* (IAP391, FPT University, Spring 2026). It implements the Doc2Vec+Cosine similarity pipeline — one of six similarity-based configurations evaluated for mapping malware behavioral evidence to MITRE D3FEND defensive actions.

### How It Works

1. **Training:** 953 ransomware samples are represented as text files, each containing a list of CAPE signature names (one per line). Doc2Vec learns a 100-dimensional vector for each document (malware sample) and each token (signature) simultaneously.
2. **Inference:** For a new malware sample, its signature list is fed to `model.infer_vector()`. To mitigate Doc2Vec's inference non-determinism, the vector is inferred 20 times with different seeds and averaged (`stable_infer_vector`). Inferred vectors are cached as `.npy` files for reuse.
3. **Comparison:** The query vector is compared against all 953 training vectors using cosine similarity. The top-10 most similar known malware are returned, optionally exported to Excel.

### Model Configuration

- **Algorithm:** DBOW (dm=0) — learns document vectors directly without word order, suitable for unordered signature sets
- **Vector size:** 100 dimensions
- **Window:** 10
- **Min count:** 1 (keep all signatures, even rare ones)
- **Epochs:** 200
- **Workers:** 1 (deterministic training)
- **Seed:** 42

### Inference Reproducibility

Doc2Vec's `infer_vector` is inherently non-deterministic. This pipeline uses multi-run averaging (20 runs, seeds 42–61) to produce stable, reproducible vectors. Inferred vectors are cached in `query_vectors/` as `.npy` files so subsequent comparisons reuse the same vector.

## Project Structure

```
doc2vec/
├── train_model_doc2vec.py         # Train Doc2Vec model on training patterns
├── compare_malware_doc2vec.py     # Compare new malware against training corpus
├── malware_doc2vec.model          # Trained Gensim Doc2Vec model
├── pattern 1-953/                 # Training data: 953 malware signature pattern files
│   ├── <sha256_1>.txt             # One file per sample, one signature per line
│   ├── <sha256_2>.txt
│   └── ...
├── pattern 1017-1217/             # Test data: 201 new malware pattern files
│   ├── <sha256_1>.txt
│   └── ...
├── query_vectors/                 # Cached inferred vectors (.npy) for test samples
│   ├── <sha256_1>.npy
│   └── ...
├── kết quả compare/               # Comparison results: one Excel per test sample
│   ├── <sha256_1>.xlsx            # Top-10 most similar training samples + scores
│   └── ...
└── logs/                          # Training and comparison logs
    ├── train_model.log
    └── compare_malware.log
```

### Pattern File Format

Each `.txt` file contains CAPE sandbox signature names, one per line:

```
dead_connect
antidebug_setunhandledexceptionfilter
language_check_registry
enumerates_running_processes
infostealer_bitcoin
```

The filename is the SHA256 hash of the malware sample (without extension).

## Usage

### 1. Train the Doc2Vec model

```bash
python train_model_doc2vec.py
```

A file dialog will prompt you to select the training pattern folder (e.g., `pattern 1-953/`). The trained model is saved as `malware_doc2vec.model`.

### 2. Compare new malware against the training corpus

```bash
python compare_malware_doc2vec.py
```

The script loads the trained model and offers two modes:

- **Mode 1 — Single compare:** Select a single `.txt` pattern file. The script infers its vector (with 20-run averaging), computes cosine similarity against all training vectors, and prints the top-10 most similar malware with similarity scores.
- **Mode 2 — Batch compare:** Select a folder of test pattern files and an output folder. For each test sample, the script infers the vector, finds the top-10 most similar training samples, and exports results as individual Excel files (one per test sample).

### Output Excel Format

Each output Excel file (`<sha256>.xlsx`) contains:

| sha256 | similarity_score |
|---|---|
| `<training_sample_hash>` | 0.9847... |
| ... | ... |

Top-10 most similar training samples, sorted by descending cosine similarity.

## Dependencies

- Python 3.10+
- gensim (Doc2Vec)
- numpy
- scikit-learn (cosine_similarity)
- openpyxl (Excel export)
- tkinter (file/folder dialogs, included with Python)

## Notes

- The model uses `workers=1` during both training and inference to ensure deterministic results.
- Inferred query vectors are cached in `query_vectors/` as `.npy` files. Delete this folder to force re-inference.
- The GUI file dialogs (tkinter) require a display environment. For headless/server use, modify the scripts to accept command-line arguments instead.
- Training and comparison logs are saved in `logs/` with full environment info for reproducibility.
