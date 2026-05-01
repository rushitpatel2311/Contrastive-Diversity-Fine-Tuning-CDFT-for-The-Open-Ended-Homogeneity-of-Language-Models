# Contrastive-Diversity-Fine-Tuning-CDFT-for-The-Open-Ended-Homogeneity-of-Language-Models
Contrastive Diversity Fine-Tuning (CDFT) for The Open-Ended Homogeneity of Language Models

---

## Overview

This project implements **Contrastive Diversity Fine-Tuning (CDFT)** — a novel training objective that augments standard causal language model fine-tuning with a hinge-based diversity penalty. The goal is to encourage a GPT-2 Medium model to generate more varied and diverse responses to open-ended conversational prompts, without sacrificing text quality (fluency).

### Core Idea

The CDFT loss function is:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \max(0, \text{sim}(f(y_1), f(y_2)) - \tau)$$

For each training step, two responses ($y_1$, $y_2$) are independently sampled for the same prompt. Their cosine similarity is computed using a frozen sentence encoder. The penalty activates **only when** this similarity exceeds the threshold $\tau$, nudging the model to diversify its outputs.

---

## Dataset

### INFINITY-CHAT (`liweijiang/infinite-chats-eval`)

| Property       | Detail                                                  |
|----------------|---------------------------------------------------------|
| **Source**     | HuggingFace Hub                                         |
| **Hub ID**     | [`liweijiang/infinite-chats-eval`](https://huggingface.co/datasets/liweijiang/infinite-chats-eval) |
| **Split used** | `train`                                                 |
| **Prompts used** | 800 (700 train / 100 eval)                            |
| **Format**     | Conversational turns (multi-turn chat)                  |
| **Access**     | Public — requires a HuggingFace account & login token   |

The notebook extracts open-ended prompts of 10–40 words from the `conversation` column. If the dataset schema changes or too few suitable prompts are found, the code automatically falls back to a set of 20 built-in fallback prompts replicated to 800 entries.

#### Loading the Dataset (from within the notebook)

```python
from datasets import load_dataset

dataset = load_dataset('liweijiang/infinite-chats-eval', split='train')
print(dataset.column_names)
print(dataset[0])
```

---

## Models Used

| Model | Role | Parameters |
|---|---|---|
| `gpt2-medium` | Fine-tuned generator | 345M |
| `all-MiniLM-L6-v2` | Frozen sentence encoder (diversity signal) | 22M |
| `gpt2-large` | Frozen perplexity model (quality proxy) | 774M |

---

## Requirements

### Hardware
- **Recommended:** Google Colab with a T4 GPU (16 GB VRAM)
- The notebook is designed to fit GPT-2 Medium + GPT-2 Large within T4 VRAM at `batch_size=8`
- CPU-only execution is supported but will be very slow

### Software / Python Packages

```
transformers>=4.40.0
datasets
sentence-transformers
accelerate
matplotlib
seaborn
scikit-learn
pandas
numpy
tqdm
torch
```

---

## How to Run ?

1. **Open the notebook** in Google Colab:
   - Upload `CDFT for The Open-Ended Homogeneity of Language Models.ipynb` to [colab.research.google.com](https://colab.research.google.com)
   - Or open it directly from Google Drive

2. **Enable GPU runtime:**
   - `Runtime` → `Change runtime type` → `T4 GPU`

3. **Set your HuggingFace token as a Colab Secret:**
   - Go to the 🔑 **Secrets** panel (left sidebar)
   - Add a new secret: **Name** = `HF_TOKEN`, **Value** = your HuggingFace API token
   - Get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

4. **Run all cells** (`Runtime` → `Run all`) or step through each section manually

---

---

## Experiment Design

### Hyperparameter Grid Search

The experiment trains **10 model configurations** in total:

| Configuration | τ (tau) | λ (lambda) |
|---|---|---|
| Baseline | — | — |
| CDFT | 0.7 | 0.1 / 0.5 / 1.0 |
| CDFT | 0.8 | 0.1 / 0.5 / 1.0 |
| CDFT | 0.9 | 0.1 / 0.5 / 1.0 |

**τ** is the cosine similarity threshold above which the diversity penalty activates.  
**λ** controls how strongly the penalty influences the total loss.

### Training Settings

| Setting | Value |
|---|---|
| Epochs | 3 |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Warmup ratio | 10% |
| Penalty interval | Every 5 steps |
| Max input length | 64 tokens |
| Max output length | 80 tokens |
| Optimiser | AdamW (weight_decay=0.01) |
| Gradient clipping | max_norm=1.0 |
| Sampling | top-p=0.9, temperature=1.0 |

---

## Evaluation Metrics

| Metric | Description | Direction |
|---|---|---|
| **Mean Pairwise Cosine Similarity** | Average similarity across all C(10,2)=45 response pairs per prompt | Lower = more diverse ↓ |
| **Distinct-2** | Ratio of unique bigrams to total bigrams across responses | Higher = more lexically diverse ↑ |
| **Perplexity** | Evaluated using frozen GPT-2 Large as a fluency proxy | Lower = more fluent ↓ |

---
---
