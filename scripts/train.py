#!/usr/bin/env python3
"""
CLI training launcher for TurnSense.

Loads defaults from ``configs/default.yaml`` and allows overriding any data,
model, or training hyperparameter from the command line before calling
:func:`src.train.train`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train import train  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TurnSense Whisper turn detector.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "default.yaml",
        help="Path to YAML configuration file.",
    )
    # Data overrides
    parser.add_argument("--manifest", type=Path, default=None, help="Path to CSV or JSONL manifest.")
    parser.add_argument("--audio-dir", type=Path, default=None, help="Base directory for audio files.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Scalar feature cache directory.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--test-size", type=float, default=None, help="Test split fraction.")
    parser.add_argument("--val-size", type=float, default=None, help="Validation split fraction.")
    # Model overrides
    parser.add_argument("--whisper-model-id", type=str, default=None, help="HuggingFace Whisper model ID.")
    parser.add_argument(
        "--n-unfrozen-encoder-layers",
        type=int,
        default=None,
        help="Number of trailing Whisper encoder layers to fine-tune.",
    )
    parser.add_argument("--turn-end-frames", type=int, default=None, help="Tail branch frame count.")
    parser.add_argument("--dropout", type=float, default=None, help="Model dropout probability.")
    # Training overrides
    parser.add_argument("--output-dir", type=Path, default=None, help="Checkpoint and log directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size.")
    parser.add_argument("--grad-accum-steps", type=int, default=None, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=None, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=None, help="AdamW weight decay.")
    parser.add_argument("--aux-weight", type=float, default=None, help="Auxiliary filler loss weight.")
    parser.add_argument("--focal-gamma", type=float, default=None, help="Focal loss gamma.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker processes.")
    return parser.parse_args()


def _pick(cli_val: Any, cfg_dict: dict[str, Any], key: str, default: Any) -> Any:
    return cli_val if cli_val is not None else cfg_dict.get(key, default)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()

    cfg: dict[str, Any] = {}
    if args.config and args.config.exists():
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    manifest_path = str(_pick(args.manifest, data_cfg, "manifest_path", "smart_turn/manifest.jsonl"))
    output_dir = str(_pick(args.output_dir, train_cfg, "output_dir", "smart_turn/checkpoints"))
    audio_dir = args.audio_dir if args.audio_dir is not None else data_cfg.get("audio_dir")

    train(
        manifest_path=manifest_path,
        output_dir=output_dir,
        audio_dir=str(audio_dir) if audio_dir is not None else None,
        cache_dir=str(_pick(args.cache_dir, data_cfg, "cache_dir", "smart_turn/cache")),
        seed=int(_pick(args.seed, cfg, "seed", 42)),
        test_size=float(_pick(args.test_size, data_cfg, "test_size", 0.10)),
        val_size=float(_pick(args.val_size, data_cfg, "val_size", 0.10)),
        whisper_model_id=str(
            _pick(args.whisper_model_id, model_cfg, "whisper_model_id", "openai/whisper-tiny")
        ),
        n_unfrozen_encoder_layers=int(
            _pick(args.n_unfrozen_encoder_layers, model_cfg, "n_unfrozen_encoder_layers", 2)
        ),
        turn_end_frames=int(_pick(args.turn_end_frames, model_cfg, "turn_end_frames", 100)),
        dropout=float(_pick(args.dropout, model_cfg, "dropout", 0.1)),
        epochs=int(_pick(args.epochs, train_cfg, "epochs", 3)),
        batch_size=int(_pick(args.batch_size, train_cfg, "batch_size", 16)),
        grad_accum_steps=int(_pick(args.grad_accum_steps, train_cfg, "grad_accum_steps", 2)),
        lr=float(_pick(args.lr, train_cfg, "lr", 3e-5)),
        weight_decay=float(_pick(args.weight_decay, train_cfg, "weight_decay", 0.01)),
        aux_weight=float(_pick(args.aux_weight, train_cfg, "aux_weight", 0.2)),
        focal_gamma=float(_pick(args.focal_gamma, train_cfg, "focal_gamma", 2.0)),
        num_workers=int(_pick(args.num_workers, train_cfg, "num_workers", 0)),
    )


if __name__ == "__main__":
    main()
