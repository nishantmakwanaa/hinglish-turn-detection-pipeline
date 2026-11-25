"""
TurnSense model architecture.

WhisperTurnDetector:
  - Backbone: Whisper Tiny encoder (frozen except the last 2 layers + LayerNorm)
  - Global branch: Conv1d(384→256→128) + Attention pooling over all 400 valid frames
  - Tail branch:   Conv1d(384→128→64)  + MaxPool over trailing 100 frames (≈ 2 s)
  - Scalar branch: Linear(5→32) projection of prosodic features
  - Trunk:         Linear(224→128→64) with LayerNorm + Dropout
  - Heads:         endpoint (primary), midfiller, endfiller (auxiliary)

Total parameters: ~9 M  |  Trainable: ~4.4 M

Loss:
  FocalLoss with gamma=2.0 on the endpoint head, plain BCE on aux heads.
  Combined: loss = focal(endpoint) + 0.2 * bce(mid) + 0.2 * bce(end)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel

# ---------------------------------------------------------------------------
# Architecture constants (must match training)
# ---------------------------------------------------------------------------
ENCODER_DIM = 384          # Whisper-tiny hidden dim
GLOBAL_DIM = 128           # Global CNN output channels
TAIL_DIM = 64              # Tail  CNN output channels
SCALAR_DIM = 32            # Scalar projection output dim
NUM_SCALAR_FEATURES = 5    # Number of prosodic features

# Encoder time-dimension constants for 8 s audio at 50 frames/s
ENCODER_FRAMES_PER_SECOND = 50
VALID_ENCODER_FRAMES = 400  # 8 s × 50 fps
TURN_END_FRAMES = 100       # trailing 2 s window


class AttentionPool(nn.Module):
    """
    Soft attention pooling over the time dimension.

    Given a sequence ``x`` of shape ``(B, T, D)``, learns a scalar attention
    weight per time step and returns a weighted sum of shape ``(B, D)``.

    Args:
        input_dim: Dimensionality of each time step.
        hidden_dim: Hidden size of the attention MLP.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, T, D)``
        Returns:
            ``(B, D)`` weighted sum.
        """
        weights = F.softmax(self.attn(x), dim=1)   # (B, T, 1)
        return (x * weights).sum(dim=1)             # (B, D)


class WhisperTurnDetector(nn.Module):
    """
    Audio-only turn-detection model for Hinglish voice agents.

    Architecture overview::

        waveform (8 s, 16 kHz)
            └─► WhisperFeatureExtractor → log-mel spectrogram (80, 3000)
                    └─► Whisper-tiny encoder  →  enc  (B, 1500, 384)
                            ┌─── Global branch ───────────────────────┐
                            │  enc[:, :400, :]                        │
                            │  Conv1d 384→256→128  + AttentionPool    │
                            │  → g  (B, 128)                          │
                            └─────────────────────────────────────────┘
                            ┌─── Tail branch (trailing ~2 s) ─────────┐
                            │  enc[:, 300:400, :]                     │
                            │  Conv1d 384→128→64  + MaxPool           │
                            │  → t  (B, 64)                           │
                            └─────────────────────────────────────────┘
                            ┌─── Scalar branch ───────────────────────┐
                            │  5 prosodic features                    │
                            │  Linear(5→32)                           │
                            │  → s  (B, 32)                           │
                            └─────────────────────────────────────────┘
                        concat([g, t, s])  → (B, 224)
                            └─► Trunk MLP 224→128→64
                                    ├─► endpoint_head   → logit (B,)
                                    ├─► midfiller_head  → logit (B,)
                                    └─► endfiller_head  → logit (B,)

    Args:
        whisper_model_id: HuggingFace model ID for the Whisper variant.
        n_unfrozen_encoder_layers: Number of *trailing* encoder layers to fine-tune.
        turn_end_frames: Number of trailing encoder frames used by the tail branch.
        dropout: Dropout probability applied in CNN and trunk layers.
    """

    def __init__(
        self,
        whisper_model_id: str = "openai/whisper-tiny",
        n_unfrozen_encoder_layers: int = 2,
        turn_end_frames: int = TURN_END_FRAMES,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------ #
        # Whisper encoder backbone
        # ------------------------------------------------------------------ #
        whisper = WhisperModel.from_pretrained(whisper_model_id)
        self.encoder = whisper.encoder
        del whisper

        self.turn_end_frames = turn_end_frames

        # Freeze all encoder parameters first
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Selectively unfreeze the last n encoder layers + final LayerNorm
        n_layers = len(self.encoder.layers)
        for layer in self.encoder.layers[max(0, n_layers - n_unfrozen_encoder_layers):]:
            for p in layer.parameters():
                p.requires_grad = True

        for p in self.encoder.layer_norm.parameters():
            p.requires_grad = True

        # ------------------------------------------------------------------ #
        # Global branch: full-context CNN + attention pooling
        # ------------------------------------------------------------------ #
        self.global_cnn = nn.Sequential(
            nn.Conv1d(ENCODER_DIM, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(256, GLOBAL_DIM, kernel_size=3, padding=1),
            nn.BatchNorm1d(GLOBAL_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.global_pool = AttentionPool(GLOBAL_DIM, hidden_dim=64)

        # ------------------------------------------------------------------ #
        # Tail branch: trailing-window CNN + max-pooling
        # ------------------------------------------------------------------ #
        self.tail_cnn = nn.Sequential(
            nn.Conv1d(ENCODER_DIM, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, TAIL_DIM, kernel_size=3, padding=1),
            nn.BatchNorm1d(TAIL_DIM),
            nn.GELU(),
        )

        # ------------------------------------------------------------------ #
        # Scalar branch: prosodic feature projection
        # ------------------------------------------------------------------ #
        self.scalar_proj = nn.Sequential(
            nn.Linear(NUM_SCALAR_FEATURES, SCALAR_DIM),
            nn.GELU(),
        )

        # ------------------------------------------------------------------ #
        # Trunk MLP
        # ------------------------------------------------------------------ #
        combined_dim = GLOBAL_DIM + TAIL_DIM + SCALAR_DIM   # 128+64+32 = 224

        self.trunk = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 2),
            nn.Linear(128, 64),
            nn.GELU(),
        )

        # ------------------------------------------------------------------ #
        # Classification heads
        # ------------------------------------------------------------------ #
        self.endpoint_head = nn.Linear(64, 1)   # primary: turn end
        self.midfiller_head = nn.Linear(64, 1)  # aux: mid-filler
        self.endfiller_head = nn.Linear(64, 1)  # aux: end-filler

    # ---------------------------------------------------------------------- #
    def forward(
        self,
        input_features: torch.Tensor,
        scalar_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_features: Log-mel spectrogram tensor of shape ``(B, 80, 3000)``
                as produced by ``WhisperFeatureExtractor``.
            scalar_features: Prosodic feature tensor of shape ``(B, 5)``
                as produced by :func:`~src.prosodic.compute_scalar_features`.

        Returns:
            Dictionary with keys:
              - ``"endpoint_logits"``  → ``(B,)``  raw logit for turn-end
              - ``"midfiller_logits"`` → ``(B,)``  raw logit for mid-filler
              - ``"endfiller_logits"`` → ``(B,)``  raw logit for end-filler

            Pass each through ``torch.sigmoid`` to obtain probabilities.
        """
        # Whisper encoder: (B, 80, 3000) → (B, 1500, 384)
        enc = self.encoder(input_features).last_hidden_state

        # --- Global branch ------------------------------------------------- #
        # Operates over full encoder output sequence (B, 1500, 384)
        g = self.global_cnn(enc.permute(0, 2, 1))       # (B, GLOBAL_DIM, T)
        g = self.global_pool(g.permute(0, 2, 1))        # (B, GLOBAL_DIM)

        # --- Tail branch --------------------------------------------------- #
        tail_start = max(0, VALID_ENCODER_FRAMES - self.turn_end_frames)  # 400 - 100 = 300
        tail = enc[:, tail_start:VALID_ENCODER_FRAMES, :]         # (B, 100, 384)
        t = self.tail_cnn(tail.permute(0, 2, 1))                  # (B, TAIL_DIM, 100)
        t = torch.max(t, dim=-1)[0]                               # (B, TAIL_DIM)

        # --- Scalar branch ------------------------------------------------- #
        s = self.scalar_proj(scalar_features)   # (B, SCALAR_DIM)

        # --- Fusion -------------------------------------------------------- #
        combined = torch.cat([g, t, s], dim=-1)   # (B, 224)
        trunk = self.trunk(combined)              # (B, 64)

        return {
            "endpoint_logits": self.endpoint_head(trunk).squeeze(-1),
            "midfiller_logits": self.midfiller_head(trunk).squeeze(-1),
            "endfiller_logits": self.endfiller_head(trunk).squeeze(-1),
        }


class FocalLoss(nn.Module):
    """
    Focal loss for binary classification with optional class imbalance weighting.

    Modulates standard binary cross-entropy by ``(1 - p_t) ** gamma`` to
    down-weight easy examples and focus training on hard negatives.

    .. math::

        FL(p_t) = (1 - p_t)^{\\gamma} \\cdot BCE(p_t)

    Args:
        gamma: Focusing parameter. Higher values → more down-weighting of easy
               examples. ``0`` reduces to standard BCE.
        pos_weight: Weight for positive class (addresses class imbalance).
                    Computed as ``(1 - pos_rate) / pos_rate`` on the training set.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer(
            "pos_weight",
            torch.tensor(float(pos_weight), dtype=torch.float32),
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw (un-sigmoided) predictions of shape ``(B,)``.
            targets: Binary labels of shape ``(B,)`` with values in ``{0, 1}``.

        Returns:
            Scalar loss tensor.
        """
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        p_t = torch.exp(-bce)
        return ((1.0 - p_t) ** self.gamma * bce).mean()
