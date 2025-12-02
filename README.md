# CMPE-256 Recommendation System: User-Item Interactions

**CMPE 256 — Group 12**

## Team Members

- **Sravan Peethani** — 018280574
- **Latha Boralingaiah** — 018301361
- **Deeksha Akkati** — 018225467

---

## Overview

This project implements and compares three recommendation algorithms on a large-scale implicit user-item interaction dataset:

1. **Item-Item KNN** (Cosine Similarity)
2. **ALS** (Alternating Least Squares)
3. **NeuMF** (Neural Matrix Factorization)

The objective is to generate **Top-20 personalized recommendations** for **~52,643 users** across **~91,599 items** while maximizing **NDCG@20** performance.

---

## Problem Statement

Given a dataset where each row contains a user followed by items they interacted with, the system must:

- Generate **Top-20 item recommendations** per user
- Handle extremely sparse implicit feedback (**>99% sparsity**)
- Scale efficiently to **millions of interactions**
- Compare **classical vs. neural** recommendation algorithms
- Evaluate performance using **NDCG@20, Recall@20, and Precision@20**

---

## System Architecture

### 1. Data Preprocessing

1. **Normalize user and item IDs**
2. **Convert interactions into a CSR sparse matrix** (52,643 × 91,599)
3. **Apply train-test split:**
   - Leave-one-out for KNN
   - Stratified split for ALS & NeuMF

### 2. Models Implemented

| Model | Description | Strength |
|-------|-------------|----------|
| **Item-KNN** | Computes cosine similarity between item vectors | Best ranking performance (highest NDCG@20) |
| **ALS** | Implicit feedback matrix factorization | Scalable and robust for large sparse datasets |
| **NeuMF** | Deep hybrid GMF + MLP neural architecture | Learns nonlinear user-item patterns |

### 3. Evaluation Metrics

Each model generates **Top-20 recommendations** per user and is evaluated using:

- **NDCG@20** (ranking quality)
- **Recall@20** (coverage of relevant items)
- **Precision@20** (accuracy of recommended items)

---

## Dataset

- **Users:** 52,643
- **Items:** 91,599
- **Interactions:** ~2.38M
- **Sparsity:** 99.95%
- **Format:** Each line contains `user_id item1 item2 item3 ...`


## Key Findings

1. **Item-KNN achieved best ranking quality** (NDCG@20=0.1263) — simplicity outperforms complexity in sparse settings
2. **NeuMF excelled at prediction accuracy** (88.44%) but struggled with ranking due to pointwise optimization
3. **ALS provided balanced performance** with scalable matrix factorization
4. **Extreme sparsity (99.95%)** was the defining challenge — 78% of users had <50 interactions

---

## Technologies Used

- **Python 3.x**
- **NumPy & Pandas** — Data preprocessing and analysis
- **SciPy** — Sparse matrix operations (CSR format)
- **Scikit-learn** — Cosine similarity computation
- **Implicit** — ALS implementation for implicit feedback
- **PyTorch** — NeuMF neural network implementation
- **Matplotlib & Seaborn** — Visualization

---

## Project Structure

