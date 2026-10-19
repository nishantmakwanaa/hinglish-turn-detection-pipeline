# TurnSense: Audio Turn Detection for Voice AI

Audio-only turn-detection model for Hinglish voice agents. It decides when the agent should respond before STT runs, keeping ASR latency off the critical path.

## Results
- 93.5% F1 (94.4% on Hindi) on a held-out test split
- 9M-parameter model (4.4M trainable), trained with focal loss

## How it works
- Whisper Tiny encoder with a dual-branch CNN (global attention-pooled context + trailing 2s window)
- Multi-task auxiliary heads (mid-filler / end-filler) plus 5 prosodic features (energy decay, F0 slope, speech-activity ratio) for robustness to fillers like "umm", "matlab", "toh"
- Resumable, constant-memory ingestion over a 41 GB / 271K-clip HuggingFace dataset, streaming Parquet row-group by row-group to curate 32K Hindi + English clips (16 kHz, 8s right-aligned windows)
- Gradio mic demo for live validation
