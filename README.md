# TurnSense: Audio Turn Detection for Voice AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Whisper-Tiny](https://img.shields.io/badge/Transformer-Whisper--Tiny-yellow.svg)](https://huggingface.co/openai/whisper-tiny)
[![Dataset](https://img.shields.io/badge/Dataset-Smart--Turn--v3.2-blue.svg)](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An audio-only turn-detection model for **Hinglish (Hindi + English)** voice agents that predicts whether a speaker has completed their turn before ASR/STT runs, keeping transcription latency off the critical path.

---

## Proof & Reference Links (Datasets, Papers, Code & Artifacts)

| Category | Resource | Link |
| :--- | :--- | :--- |
| **Source Dataset** | Pipecat Smart Turn v3.2 Corpus (`41 GB`, `271K` clips) | [`pipecat-ai/smart-turn-data-v3.2-train`](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train) |
| **Baseline Reference** | Pipecat Smart Turn Framework | [`pipecat-ai/smart-turn`](https://github.com/pipecat-ai/smart-turn) |
| **Transformer Backbone** | OpenAI Whisper Tiny (`openai/whisper-tiny`) | [Hugging Face Model Card](https://huggingface.co/openai/whisper-tiny) · [HF Docs](https://huggingface.co/docs/transformers/model_doc/whisper) |
| **Whisper Paper** | Radford et al. (2022), *Robust Speech Recognition via Large-Scale Weak Supervision* | [`arXiv:2212.04356`](https://arxiv.org/abs/2212.04356) |
| **Loss Function Paper** | Lin et al. (2017), *Focal Loss for Dense Object Detection* | [`arXiv:1708.02002`](https://arxiv.org/abs/1708.02002) |
| **Research Report** | Full 9-page TurnSense Technical Paper | [`Hinglish_Turn_Detection_Report.pdf`](Hinglish_Turn_Detection_Report.pdf) |
| **Training Proof** | Epoch-by-epoch validation accuracy & F1 log | [`training_history.csv`](training_history.csv) |
| **Curated Dataset Proof** | Stratified 32,006-clip metadata manifest | [`data/manifest.csv`](data/manifest.csv) |
| **End-to-End Notebook** | Colab training, evaluation & Gradio mic demo | [`notebook/hinglish_audiotd.ipynb`](notebook/hinglish_audiotd.ipynb) |
| **Core Model Code** | Transformer + Dual-Branch 1D CNN + AttentionPool + FocalLoss | [`src/model.py`](src/model.py) |

---

## Key Highlights

- **Audio-Only Turn Detection:** Uses a **Whisper-Tiny Transformer encoder** paired with a **Dual-Branch 1D CNN** (global attention-pooled context + trailing-2s window) and 5 prosodic features, reaching **93.54% F1 overall (94.40% F1 on Hindi)** on the held-out test split before STT runs.
- **Resumable Streaming Pipeline:** Streams Parquet row-group by row-group from [`pipecat-ai/smart-turn-data-v3.2-train`](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train) via [`scripts/download_data.py`](scripts/download_data.py) in constant memory without holding full shards in RAM.
- **Filler & Prosodic Robustness:** Multi-task auxiliary heads (`midfiller`, `endfiller`) plus 5 acoustic/prosodic features make the model robust to Hinglish conversational fillers (*"umm"*, *"matlab"*, *"toh"*).
- **Solving Indic Voice Interruption:** Addresses premature turn-end triggers in English-centric models on Indic speech (Smart Turn v3 exhibits an 8.57% false-turn-end rate on Hindi vs. 4.73% overall).

---

## Dataset & Preprocessing

- **Where It Comes From:** Streamed from Hugging Face [`pipecat-ai/smart-turn-data-v3.2-train`](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train) (`41 GB` / `~271K` clips in Parquet shards).
- **Ingestion Implementation:** [`scripts/download_data.py`](scripts/download_data.py) iterates Parquet shards and row groups with a resumable JSONL/state checkpoint (`per_language_cap: 20,000`).
- **Audio Normalization ([`src/prosodic.py`](src/prosodic.py)):** Resampled to **16 kHz mono PCM-16 WAV** and placed into **8.0-second right-aligned windows** (`128,000` samples; left zero-padded if shorter, left-truncated if longer) so turn boundaries always align to the right edge.
- **Curated Manifest ([`data/manifest.csv`](data/manifest.csv)):** **32,006 total clips**:
  - **English (`eng`):** `20,000` clips
  - **Hindi (`hin`):** `12,006` clips
  - **Turn-End Labels (`endpoint_bool`):** `15,900` `True` / `16,106` `False` (plus auxiliary `midfiller` and `endfiller` binary labels)
  - **Splits ([`src/dataset.py`](src/dataset.py)):** Stratified **80% train / 10% validation / 10% test** (`seed=42`).

---

## Transformer + Dual-Branch CNN Architecture

Implemented in [`WhisperTurnDetector`](src/model.py) ([`src/model.py`](src/model.py)):

1. **Transformer Encoder Backbone ([`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny)):**
   - Input `16 kHz` waveform is converted by `WhisperFeatureExtractor` into an **80-bin log-mel spectrogram `(B, 80, 3000)`**.
   - Whisper-Tiny encoder (4 Transformer encoder blocks, `d_model=384`, 6 attention heads, 50 frames/sec) outputs contextual frame representations **`enc` of shape `(B, 1500, 384)`**.
   - **Parameter-Efficient Fine-Tuning:** Bottom encoder layers are frozen; only the **last 2 Transformer encoder layers + final `layer_norm`** are unfrozen (`~9M` total params, `~4.4M` trainable).
2. **Global Branch (1D CNN + Soft Attention Pooling):**
   - Operates over the full encoder sequence: `Conv1d(384→256, k=5, pad=2) → BatchNorm1d → GELU → Dropout(0.1) → Conv1d(256→128, k=3, pad=1) → BatchNorm1d → GELU → Dropout(0.1)`.
   - Summarized via learned [`AttentionPool`](src/model.py) (`Linear(128→64) → Tanh → Linear(64→1) → Softmax`) into **`g` `(B, 128)`**.
3. **Tail Branch (Trailing-2s 1D CNN + Temporal MaxPool):**
   - Slices the trailing 2 seconds of the 8s active window (`enc[:, 300:400, :]`, `100` frames at `50 fps`) to capture end-of-utterance acoustic cues.
   - Applies `Conv1d(384→128, k=3, pad=1) → BatchNorm1d → GELU → Dropout(0.1) → Conv1d(128→64, k=3, pad=1) → BatchNorm1d → GELU` + `MaxPool` over time into **`t` `(B, 64)`**.
4. **Prosodic Scalar Branch ([`src/prosodic.py`](src/prosodic.py)):**
   - Extracts **5 acoustic/prosodic features**: `energy_decay`, `f0_slope` (pitch trend), `speech_activity_ratio`, `mean_rms`, and `final_to_mean_rms_ratio`.
   - Projected via `Linear(5→32) → GELU` into **`s` `(B, 32)`**.
5. **Fusion Trunk & Multi-Task Heads:**
   - Concatenates `[g, t, s]` → `(B, 224)` → Trunk MLP `Linear(224→128) → LayerNorm → GELU → Dropout(0.2) → Linear(128→64) → GELU`.
   - Three output heads: **`endpoint_head`** (turn complete), **`midfiller_head`**, and **`endfiller_head`**.

```text
8.0 s Waveform (16 kHz, right-aligned)
  ├──► WhisperFeatureExtractor ──► Log-Mel (80, 3000)
  │       └──► Whisper-Tiny Transformer Encoder (last 2 layers unfrozen) ──► (B, 1500, 384)
  │             ├── Global 1D CNN (384→256→128) + AttentionPool ──► g (B, 128)
  │             └── Tail 1D CNN   (384→128→64, last 2s) + MaxPool ──► t (B, 64)
  └──► 5 Prosodic Features ──► Linear(5→32) + GELU ─────────────────► s (B, 32)
                                                                          │
concat([g, t, s]) (B, 224) ──► Trunk MLP (224→128→64)
                                  ├──► endpoint_head  (FocalLoss, γ=2.0)
                                  ├──► midfiller_head (0.2 × BCE)
                                  └──► endfiller_head (0.2 × BCE)
```

### How It Is Trained & Used

- **Training ([`src/train.py`](src/train.py)):** `FocalLoss(gamma=2.0)` with positive-class weighting on `endpoint_logits` + `0.2 * BCEWithLogitsLoss` on both auxiliary filler heads; `AdamW` (`lr=3e-5`, `weight_decay=0.01`), `batch_size=16`, `grad_accum_steps=2`, `3` epochs, AMP mixed precision.
- **Real-Time Inference ([`src/inference.py`](src/inference.py)):** [`TurnDetector`](src/inference.py) scores raw microphone audio or WAV clips and outputs `endpoint_prob`, `midfiller_prob`, `endfiller_prob`, and `is_turn_end` before running ASR/STT.

---

## Evaluation Results

Verified in [`training_history.csv`](training_history.csv) and [`Hinglish_Turn_Detection_Report.pdf`](Hinglish_Turn_Detection_Report.pdf):

| Metric | Overall | Hindi (`hin`) | English (`eng`) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **93.50%** | **94.50%** | **92.90%** |
| **F1 Score** | **93.54%** | **94.40%** | **93.05%** |
| **Precision** | 92.39% | - | - |
| **Recall** | 94.72% | - | - |

---

## Repository Structure

```text
hinglish-turn-detection-pipeline/
├── configs/
│   └── default.yaml                     # Default data, model, training, and inference config
├── src/
│   ├── __init__.py                      # Public package exports
│   ├── model.py                         # WhisperTurnDetector, AttentionPool, FocalLoss
│   ├── prosodic.py                      # 5 prosodic scalar features, resampling, 8s windowing
│   ├── dataset.py                       # TurnDataset & DataLoader builders
│   ├── train.py                         # Mixed-precision training & evaluation loop
│   └── inference.py                     # TurnDetector real-time inference & Gradio demo
├── scripts/
│   ├── download_data.py                 # Resumable Parquet row-group streaming data curation
│   ├── train.py                         # CLI training launcher
│   └── infer.py                         # CLI file inference & Gradio mic demo launcher
├── notebook/
│   └── hinglish_audiotd.ipynb           # End-to-end Colab training & Gradio mic demo
├── data/
│   └── manifest.csv                     # Curated 32,006-clip Hinglish manifest
├── Hinglish_Turn_Detection_Report.pdf   # Complete 9-page research report
├── training_history.csv                 # Per-epoch validation metrics
├── requirements.txt                     # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## Quickstart & CLI Usage

```bash
# 1. Clone repository & install dependencies
git clone https://github.com/nishantmakwanaa/hinglish-turn-detection-pipeline.git
cd hinglish-turn-detection-pipeline
pip install -r requirements.txt

# 2. Stream and curate 16 kHz, 8s right-aligned Hindi & English clips (resumable)
python scripts/download_data.py --config configs/default.yaml

# 3. Train the model using configs/default.yaml (or override hyperparameters via CLI)
python scripts/train.py --config configs/default.yaml --epochs 3 --batch-size 16

# 4. Run inference on audio files or launch the interactive Gradio mic demo
python scripts/infer.py --checkpoint smart_turn/checkpoints/best_model.pt --audio path/to/clip.wav
python scripts/infer.py --checkpoint smart_turn/checkpoints/best_model.pt --demo

# Alternatively, run the end-to-end Colab notebook:
jupyter notebook notebook/hinglish_audiotd.ipynb
```
