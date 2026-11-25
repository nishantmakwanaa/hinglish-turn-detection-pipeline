# TurnSense: Audio Turn Detection for Voice AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Whisper-Tiny-yellow.svg)](https://huggingface.co/openai/whisper-tiny)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An audio-only turn-detection model for **Hinglish (Hindi + English)** voice agents that predicts whether a speaker has completed their turn before STT runs, keeping ASR latency off the critical path.

---

## Key Highlights

- **Audio-Only Turn Detection:** Uses the **Whisper Tiny** encoder (`openai/whisper-tiny`) with a dual-branch CNN (global attention-pooled context + trailing-2s window) and 5 prosodic features, achieving **93.54% F1 (94.40% on Hindi)** on the held-out test split before STT runs.
- **Resumable Streaming Pipeline:** Streams Parquet row-group by row-group over the 41 GB / 271K-clip HuggingFace dataset ([`pipecat-ai/smart-turn-data-v3.2-train`](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train)) in constant memory without holding full shards in RAM.
- **Filler & Prosodic Robustness:** Multi-task auxiliary heads (`mid-filler`, `end-filler`) plus 5 hand-engineered prosodic scalar features make the model robust to conversational fillers (*"umm"*, *"matlab"*, *"toh"*).
- **Solving Indic Voice Interruption:** Bridges the gap where English-centric turn models over-fire on Indic speech (Smart Turn v3 exhibits an 8.57% false-turn-end rate on Hindi vs. 4.73% overall).

📄 **Full Technical Paper:** See [Hinglish_Turn_Detection_Report.pdf](Hinglish_Turn_Detection_Report.pdf) for the complete 9-page research report.

---

## Dataset & Source

- **Source Dataset:** [`pipecat-ai/smart-turn-data-v3.2-train`](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train) on Hugging Face (41 GB / ~271K multi-lingual voice clips in Parquet format).
- **Ingestion Method:** Streamed Parquet shard-by-shard and row-group-by-row-group via [`scripts/download_data.py`](scripts/download_data.py) (`per_language_cap: 20,000`), keeping memory bounded on free-tier Colab.
- **Audio Preprocessing:** Resampled to **16 kHz mono PCM-16 WAV** and normalized into **8.0-second right-aligned windows** (`128,000` samples; silence-padded on the left if shorter, truncated from the left if longer) so the end of the turn always sits at the right edge of the window.
- **Curated Manifest ([`data/manifest.csv`](data/manifest.csv)):** **32,006 clips** total:
  - **English (`eng`):** 20,000 clips
  - **Hindi (`hin`):** 12,006 clips
  - **Labels (`endpoint_bool`):** 15,900 turn-end (`True`) / 16,106 non-turn-end (`False`), plus auxiliary `midfiller` and `endfiller` flags.
  - **Splits:** Stratified **80% train / 10% validation / 10% test** (`seed: 42`).

---

## Model & Architecture

Implemented in [`src/model.py`](src/model.py) (`WhisperTurnDetector`) and [`src/prosodic.py`](src/prosodic.py):

- **Pre-trained Backbone:** [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny) encoder (`384`-dim hidden state, `50` encoder frames/sec). All layers are frozen except the **last 2 transformer encoder layers** and final `layer_norm`.
- **Dual-Branch + Prosodic Fusion (~9M total params, ~4.4M trainable):**

```text
8.0 s Audio Waveform (16 kHz, right-aligned)
  ├──► WhisperFeatureExtractor ──► Log-Mel Spectrogram (80, 3000)
  │       └──► Whisper-Tiny Encoder ──► enc (B, 1500, 384)
  │             ├── Global Branch: full enc (1500 frames)
  │             │     └── Conv1d(384→256→128) + AttentionPool ──► g (B, 128)
  │             └── Tail Branch: trailing 2 s (enc[:, 300:400, :])
  │                   └── Conv1d(384→128→64) + MaxPool      ──► t (B, 64)
  └──► Prosodic Extractor (5 scalar features)
        [energy decay, F0 slope, speech-activity ratio, mean RMS, final/mean RMS]
        └── Linear(5→32) + GELU ────────────────────────────────► s (B, 32)
                                                                      │
concat([g, t, s]) ──► (B, 224) ──► Trunk MLP (224→128→64, LayerNorm, Dropout)
                                      ├──► endpoint_head  (primary turn-end logit)
                                      ├──► midfiller_head (auxiliary mid-filler logit)
                                      └──► endfiller_head (auxiliary end-filler logit)
```

### How It Is Trained & Used

- **Training Objective ([`src/train.py`](src/train.py)):** Multi-task loss combining `FocalLoss(gamma=2.0)` with positive-class weighting on `endpoint_logits`, plus `0.2 * BCEWithLogitsLoss` on both `midfiller_logits` and `endfiller_logits`.
- **Optimization:** `AdamW` (`lr=3e-5`, `weight_decay=0.01`), `batch_size=16`, `grad_accum_steps=2` (effective batch size 32), `3` epochs, AMP mixed precision on CUDA.
- **Inference ([`src/inference.py`](src/inference.py)):** [`TurnDetector`](src/inference.py) takes raw microphone audio arrays or WAV files, applies 8s right-aligned windowing, extracts the 5 prosodic features + log-mel spectrogram, and outputs `endpoint_prob`, `midfiller_prob`, `endfiller_prob`, and `is_turn_end` before running ASR/STT.

---

## Evaluation Results

Results from the held-out test split (see [`training_history.csv`](training_history.csv) and [Hinglish_Turn_Detection_Report.pdf](Hinglish_Turn_Detection_Report.pdf)):

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
├── Hinglish_Turn_Detection_Report.pdf   # Complete research report
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
