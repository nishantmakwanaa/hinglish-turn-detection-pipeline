# TurnSense: Audio Turn Detection for Voice AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An audio-only turn-detection model for **Hinglish (Hindi + English)** voice agents that predicts whether a speaker has completed their turn before STT runs, keeping ASR latency off the critical path.

---

## 🚀 Key Highlights

- **Audio-Only Turn Detection:** Built an audio-only turn-detection model for Hinglish voice agents using Whisper Tiny encoder with a dual-branch CNN (global attention-pooled context + trailing-2s window) hitting **93.5% F1 (94.4% on Hindi)** on a held-out test split, letting the agent decide when to respond before STT runs.
- **Resumable Streaming Pipeline:** Engineered a resumable, constant-memory ingestion pipeline over a 41 GB / 271K-clip HuggingFace dataset (`pipecat-ai/smart-turn-data-v3.2-train`), streaming Parquet row-group by row-group to curate 40K Hindi + English clips (16 kHz, 8s right-aligned windows) on free-tier Colab without holding shards in RAM.
- **Filler & Prosodic Robustness:** Made the model robust to filler words (*"umm"*, *"matlab"*, *"toh"*) via multi-task auxiliary heads (`mid-filler` / `end-filler`) plus 5 hand-engineered prosodic features (energy decay, F0 slope, speech-activity ratio, mean RMS, final/mean RMS ratio), training a 9M-param model (4.4M trainable) with focal loss; shipped a Gradio mic demo for live validation.
- **Solving Indic Voice Interruption:** Interrupting a user mid-sentence is the #1 reason voice agents feel broken. English-centric turn models over-fire on Indic speech (Smart Turn v3 has an 8.57% false-turn-end rate on Hindi vs 4.73% overall) — this model directly bridges that gap.

📄 **Full Technical Paper:** See [Hinglish_Turn_Detection_Report.pdf](Hinglish_Turn_Detection_Report.pdf) for the comprehensive 9-page research report.

---

## 📊 Evaluation Results

| Metric | Overall | Hindi (`hin`) | English (`eng`) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **93.50%** | **94.50%** | **92.90%** |
| **F1 Score** | **93.54%** | **94.40%** | **93.05%** |
| **Precision** | 92.39% | - | - |
| **Recall** | 94.72% | - | - |

---

## 📁 Repository Structure

```text
ai-turn-detection/
├── notebook/
│   └── hinglish_audiotd.ipynb          # End-to-end training & Gradio mic demo
├── data/
│   └── manifest.csv                     # Stratified dataset manifest
├── Hinglish_Turn_Detection_Report.pdf   # Complete research report
├── training_history.csv                 # Per-epoch training metrics
├── requirements.txt                     # Pinned dependencies
├── .gitignore                           # Git ignore rules
└── README.md                            # Project documentation
```

---

## 🛠️ Quickstart

```bash
# 1. Clone repository & install dependencies
git clone https://github.com/nishantmakwanaa/ai-turn-detection.git
cd ai-turn-detection
pip install -r requirements.txt

# 2. Run training or demo in notebook/hinglish_audiotd.ipynb
jupyter notebook notebook/hinglish_audiotd.ipynb
```
