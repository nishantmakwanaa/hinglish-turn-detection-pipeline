"""
TurnDataset: PyTorch Dataset for Hinglish turn-detection training.

Reads pre-processed WAV files from a manifest DataFrame, applies the
8-second right-aligned windowing, extracts Whisper log-mel features, and
returns pre-cached prosodic scalar features alongside the classification labels.

Dataset schema (from ``pipecat-ai/smart-turn-data-v3.2-train``):
  - audio_path:    absolute path to the saved 16 kHz WAV clip
  - endpoint_bool: True → speaker has finished their turn
  - midfiller:     True → clip contains a mid-utterance filler word
  - endfiller:     True → clip ends with a filler word
  - language:      "hin" or "eng"
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperFeatureExtractor

try:
    import soundfile as sf
    _SOUNDFILE_AVAILABLE = True
except ImportError:
    sf = None
    _SOUNDFILE_AVAILABLE = False

from .prosodic import prepare_fixed_window, resample_audio

TARGET_SR = 16_000
FIXED_WINDOW_SECONDS = 8.0

# Shared feature extractors keyed by model_id (thread-safe for reading)
_FEATURE_EXTRACTORS: dict[str, WhisperFeatureExtractor] = {}


def get_feature_extractor(
    model_id: str = "openai/whisper-tiny",
) -> WhisperFeatureExtractor:
    """
    Return a module-level cached ``WhisperFeatureExtractor`` to avoid
    repeated HuggingFace downloads during dataset construction.

    Args:
        model_id: HuggingFace model identifier.

    Returns:
        Cached :class:`WhisperFeatureExtractor` instance.
    """
    if model_id not in _FEATURE_EXTRACTORS:
        _FEATURE_EXTRACTORS[model_id] = WhisperFeatureExtractor.from_pretrained(model_id)
    return _FEATURE_EXTRACTORS[model_id]


class TurnDataset(Dataset):
    """
    PyTorch Dataset for the TurnSense turn-detection task.

    Each item is a dictionary containing:
      - ``"input_features"``  : ``(80, 3000)`` log-mel spectrogram tensor
      - ``"scalar_features"`` : ``(5,)``  prosodic feature tensor
      - ``"endpoint_label"``  : scalar float  (0.0 or 1.0)
      - ``"midfiller_label"`` : scalar float  (0.0 or 1.0, NaN → 0.0)
      - ``"endfiller_label"`` : scalar float  (0.0 or 1.0, NaN → 0.0)
      - ``"language"``        : str  ("hin" or "eng")

    Args:
        df: Manifest DataFrame with the columns listed in the module docstring.
        scalar_cache: Pre-computed prosodic features array of shape
                      ``(len(df), 5)`` as returned by
                      :func:`~src.prosodic.build_scalar_cache`.
        whisper_model_id: HuggingFace ID of the Whisper feature extractor.
        target_sr: Expected sample rate; audio is validated (not resampled here).
        window_seconds: Duration of the right-aligned window in seconds.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        scalar_cache: np.ndarray,
        whisper_model_id: str = "openai/whisper-tiny",
        target_sr: int = TARGET_SR,
        window_seconds: float = FIXED_WINDOW_SECONDS,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.scalar_cache = scalar_cache
        self.feature_extractor = get_feature_extractor(whisper_model_id)
        self.target_sr = target_sr
        self.window_seconds = window_seconds

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        if not _SOUNDFILE_AVAILABLE:
            raise ImportError("soundfile is required to load audio files in TurnDataset.")

        row = self.df.iloc[idx]

        # Load waveform
        y, sr = sf.read(row["audio_path"], dtype="float32")

        if y.ndim == 2:
            y = y.mean(axis=1)

        if sr != self.target_sr:
            y = resample_audio(y, src_sr=sr, dst_sr=self.target_sr)

        # Apply fixed 8-second right-aligned window + normalisation
        y = prepare_fixed_window(y, sr=self.target_sr, seconds=self.window_seconds)

        # Whisper feature extraction → (1, 80, 3000) → squeeze → (80, 3000)
        features = self.feature_extractor(
            y,
            sampling_rate=self.target_sr,
            return_tensors="pt",
        )["input_features"][0]

        return {
            "input_features": features,
            "scalar_features": torch.tensor(
                self.scalar_cache[idx], dtype=torch.float32
            ),
            "endpoint_label": float(row["endpoint_bool"]),
            "midfiller_label": (
                0.0 if pd.isna(row.get("midfiller", float("nan")))
                else float(row.get("midfiller", 0.0))
            ),
            "endfiller_label": (
                0.0 if pd.isna(row.get("endfiller", float("nan")))
                else float(row.get("endfiller", 0.0))
            ),
            "language": row["language"],
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate function for :class:`TurnDataset`.

    Stacks tensors and collects language strings into a list.

    Args:
        batch: List of dicts returned by :meth:`TurnDataset.__getitem__`.

    Returns:
        Batched dictionary with the following keys:
          - ``"input_features"``  : ``(B, 80, 3000)``
          - ``"scalar_features"`` : ``(B, 5)``
          - ``"endpoint_labels"`` : ``(B,)``
          - ``"midfiller_labels"``: ``(B,)``
          - ``"endfiller_labels"``: ``(B,)``
          - ``"languages"``       : ``list[str]``
    """
    return {
        "input_features": torch.stack([x["input_features"] for x in batch]),
        "scalar_features": torch.stack([x["scalar_features"] for x in batch]),
        "endpoint_labels": torch.tensor(
            [x["endpoint_label"] for x in batch], dtype=torch.float32
        ),
        "midfiller_labels": torch.tensor(
            [x["midfiller_label"] for x in batch], dtype=torch.float32
        ),
        "endfiller_labels": torch.tensor(
            [x["endfiller_label"] for x in batch], dtype=torch.float32
        ),
        "languages": [x["language"] for x in batch],
    }


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_scalars: np.ndarray,
    val_scalars: np.ndarray,
    test_scalars: np.ndarray,
    batch_size: int = 16,
    num_workers: int = 0,
    whisper_model_id: str = "openai/whisper-tiny",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience factory that returns train / val / test DataLoaders.

    Args:
        train_df: Training split manifest.
        val_df: Validation split manifest.
        test_df: Test split manifest.
        train_scalars: Pre-computed scalar cache for the training split.
        val_scalars: Pre-computed scalar cache for the validation split.
        test_scalars: Pre-computed scalar cache for the test split.
        batch_size: Mini-batch size.
        num_workers: Number of DataLoader worker processes (0 = main process).
        whisper_model_id: Whisper feature extractor to use.

    Returns:
        Tuple of ``(train_loader, val_loader, test_loader)``.
    """
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    train_loader = DataLoader(
        TurnDataset(train_df, train_scalars, whisper_model_id),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        TurnDataset(val_df, val_scalars, whisper_model_id),
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        TurnDataset(test_df, test_scalars, whisper_model_id),
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader
