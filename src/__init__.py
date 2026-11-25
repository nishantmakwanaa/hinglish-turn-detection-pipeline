"""
TurnSense: Audio-only turn detection model for Hinglish (Hindi + English) voice agents.

This package provides modular components for the TurnSense model:
  - model: WhisperTurnDetector architecture (Whisper Tiny encoder + dual-branch CNN)
  - dataset: TurnDataset and DataLoader helpers for 16 kHz 8 s right-aligned audio
  - train: Training and evaluation loops with focal loss and auxiliary filler heads
  - inference: Real-time turn detection inference and Gradio demo helpers
  - prosodic: Prosodic feature extraction and windowing utilities
"""

from .dataset import (
    TurnDataset,
    build_dataloaders,
    collate_fn,
    get_feature_extractor,
)
from .inference import (
    TurnDetector,
    build_gradio_demo,
    launch_gradio_demo,
)
from .model import (
    AttentionPool,
    FocalLoss,
    WhisperTurnDetector,
)
from .prosodic import (
    build_scalar_cache,
    compute_scalar_features,
    prepare_fixed_window,
    resample_audio,
)
from .train import (
    evaluate,
    evaluate_detailed,
    train,
    train_one_epoch,
)

__version__ = "1.0.0"
__all__ = [
    # model
    "WhisperTurnDetector",
    "AttentionPool",
    "FocalLoss",
    # dataset
    "TurnDataset",
    "collate_fn",
    "build_dataloaders",
    "get_feature_extractor",
    # prosodic
    "compute_scalar_features",
    "build_scalar_cache",
    "prepare_fixed_window",
    "resample_audio",
    # train
    "train",
    "train_one_epoch",
    "evaluate",
    "evaluate_detailed",
    # inference
    "TurnDetector",
    "build_gradio_demo",
    "launch_gradio_demo",
]
