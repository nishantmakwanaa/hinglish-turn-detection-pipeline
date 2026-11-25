"""
TurnSense training loop.

Trains the WhisperTurnDetector using:
  - Focal loss (gamma=2.0) on the endpoint head
  - Plain BCE on the mid-filler and end-filler auxiliary heads
  - Combined loss: focal(endpoint) + 0.2 * bce(mid) + 0.2 * bce(end)
  - AdamW optimiser with cosine annealing LR schedule
  - Gradient accumulation (default: 2 steps)
  - Mixed-precision (float16) on CUDA via torch.amp.GradScaler
  - Gradient clipping at max-norm 1.0

Typical training time: ~25 min / epoch on a T4 GPU with 40K samples.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm.auto import tqdm

from .model import WhisperTurnDetector, FocalLoss
from .dataset import build_dataloaders
from .prosodic import build_scalar_cache

log = logging.getLogger(__name__)


def train_one_epoch(
    model: WhisperTurnDetector,
    loader: torch.utils.data.DataLoader,
    endpoint_loss_fn: FocalLoss,
    aux_loss_fn: nn.BCEWithLogitsLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_accum_steps: int = 2,
    aux_weight: float = 0.2,
    epoch: int = 0,
    total_epochs: int = 1,
) -> float:
    """
    Run one full training epoch.

    Args:
        model: The TurnSense model in training mode.
        loader: Training DataLoader.
        endpoint_loss_fn: Focal loss for the endpoint head.
        aux_loss_fn: BCE loss for the auxiliary filler heads.
        optimizer: AdamW (or similar) optimiser.
        scheduler: LR scheduler; stepped after each effective update.
        scaler: GradScaler for mixed-precision training.
        device: Target device.
        grad_accum_steps: Number of micro-steps before each parameter update.
        aux_weight: Relative weight for each auxiliary head loss.
        epoch: Current epoch index (0-based, used for tqdm label).
        total_epochs: Total number of training epochs.

    Returns:
        Mean loss over all mini-batches in the epoch.
    """
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(
        tqdm(loader, desc=f"Epoch {epoch + 1}/{total_epochs}")
    ):
        feats = batch["input_features"].to(device, non_blocking=True)
        scalars = batch["scalar_features"].to(device, non_blocking=True)
        endpoint_labels = batch["endpoint_labels"].to(device)
        midfiller_labels = batch["midfiller_labels"].to(device)
        endfiller_labels = batch["endfiller_labels"].to(device)

        use_amp = device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(feats, scalars)

            loss = (
                endpoint_loss_fn(out["endpoint_logits"], endpoint_labels)
                + aux_weight * aux_loss_fn(out["midfiller_logits"], midfiller_labels)
                + aux_weight * aux_loss_fn(out["endfiller_logits"], endfiller_labels)
            )
            loss_for_backward = loss / grad_accum_steps

        scaler.scale(loss_for_backward).backward()

        should_step = (
            (step + 1) % grad_accum_steps == 0
            or (step + 1) == len(loader)
        )

        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        running_loss += float(loss.item())

    return running_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: WhisperTurnDetector,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    desc: str = "Val",
) -> dict[str, float]:
    """
    Evaluate the model on a DataLoader; returns accuracy, F1, precision, and recall.

    Args:
        model: Model in eval mode.
        loader: Val or test DataLoader.
        device: Target device.
        desc: tqdm description string.

    Returns:
        Dict with keys ``"accuracy"``, ``"f1"``, ``"precision"``, and ``"recall"``.
    """
    model.eval()
    preds, labels = [], []

    for batch in tqdm(loader, desc=desc):
        feats = batch["input_features"].to(device)
        scalars = batch["scalar_features"].to(device)

        use_amp = device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(feats, scalars)

        probs = torch.sigmoid(out["endpoint_logits"]).float().cpu().numpy()
        preds.extend((probs >= 0.5).astype(int).tolist())
        labels.extend(batch["endpoint_labels"].numpy().astype(int).tolist())

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }


@torch.no_grad()
def evaluate_detailed(
    model: WhisperTurnDetector,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    desc: str = "Test",
) -> dict:
    """
    Evaluate the model with overall metrics, confusion matrix, and per-language breakdown.
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_langs: list[str] = []

    for batch in tqdm(loader, desc=desc):
        feats = batch["input_features"].to(device)
        scalars = batch["scalar_features"].to(device)

        use_amp = device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(feats, scalars)

        probs = torch.sigmoid(out["endpoint_logits"]).float().cpu().numpy()
        all_preds.extend((probs >= 0.5).astype(int).tolist())
        all_labels.extend(batch["endpoint_labels"].numpy().astype(int).tolist())
        all_langs.extend(batch["languages"])

    preds_np = np.asarray(all_preds)
    labels_np = np.asarray(all_labels)
    langs_np = np.asarray(all_langs)

    per_language: dict[str, dict[str, float | int]] = {}
    for lang in sorted(set(all_langs)):
        mask = langs_np == lang
        per_language[lang] = {
            "count": int(mask.sum()),
            "accuracy": float(accuracy_score(labels_np[mask], preds_np[mask])),
            "f1": float(f1_score(labels_np[mask], preds_np[mask], zero_division=0)),
        }

    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        "per_language": per_language,
    }


def _resolve_audio_path(path_str: str, base_dir: Path) -> str:
    """Resolve an audio path relative to ``base_dir``, falling back to basename."""
    p = Path(path_str)
    if p.is_absolute():
        if p.exists():
            return str(p)
        return str(base_dir / p.name)
    candidate = base_dir / p
    if candidate.exists():
        return str(candidate)
    by_name = base_dir / p.name
    if by_name.exists():
        return str(by_name)
    return str(candidate)


def train(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    # Data
    audio_dir: Optional[str] = None,
    cache_dir: str = "cache",
    seed: int = 42,
    test_size: float = 0.10,
    val_size: float = 0.10,
    # Model
    whisper_model_id: str = "openai/whisper-tiny",
    n_unfrozen_encoder_layers: int = 2,
    turn_end_frames: int = 100,
    dropout: float = 0.1,
    # Training
    epochs: int = 3,
    batch_size: int = 16,
    grad_accum_steps: int = 2,
    lr: float = 3e-5,
    weight_decay: float = 0.01,
    aux_weight: float = 0.2,
    focal_gamma: float = 2.0,
    num_workers: int = 0,
) -> list[dict]:
    """
    Full end-to-end training function.

    Loads the manifest, splits into train/val/test, pre-computes prosodic
    scalar caches, instantiates the model and optimiser, trains for
    ``epochs`` epochs, and saves the best checkpoint by validation F1.

    Args:
        manifest_path: Path to the JSONL or CSV manifest with columns
                       ``audio_path``, ``endpoint_bool``, ``midfiller``,
                       ``endfiller``, ``language``.
        output_dir: Directory to store checkpoints and training history CSV.
        audio_dir: Optional base directory to prefix relative audio paths.
        cache_dir: Directory for ``.npy`` prosodic feature caches.
        seed: Random seed for reproducibility.
        test_size: Fraction of data held out as test set.
        val_size: Fraction of remaining data held out as validation set.
        whisper_model_id: HuggingFace Whisper model ID.
        n_unfrozen_encoder_layers: Number of trailing encoder layers to fine-tune.
        turn_end_frames: Trailing frame count for the tail branch.
        dropout: Dropout probability.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        grad_accum_steps: Gradient accumulation steps.
        lr: Initial learning rate.
        weight_decay: AdamW weight decay.
        aux_weight: Filler head loss weight coefficient.
        focal_gamma: Focal loss gamma parameter.
        num_workers: DataLoader worker count.

    Returns:
        Training history as a list of dicts with keys
        ``epoch``, ``train_loss``, ``val_acc``, ``val_f1``.
    """
    from sklearn.model_selection import train_test_split

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load & split manifest
    # ------------------------------------------------------------------ #
    manifest_file = Path(manifest_path)
    if manifest_file.suffix.lower() == ".jsonl":
        df = pd.read_json(manifest_file, lines=True)
    else:
        df = pd.read_csv(manifest_file)

    if audio_dir:
        base = Path(audio_dir)
        df["audio_path"] = df["audio_path"].apply(
            lambda p: _resolve_audio_path(str(p), base)
        )

    df["_strat"] = (
        df["language"].astype(str) + "_" + df["endpoint_bool"].astype(int).astype(str)
    )

    train_df, temp_df = train_test_split(
        df, test_size=test_size + val_size, random_state=seed, stratify=df["_strat"]
    )
    relative_val = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=1.0 - relative_val, random_state=seed, stratify=temp_df["_strat"]
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    log.info("Train: %d | Val: %d | Test: %d", len(train_df), len(val_df), len(test_df))

    # ------------------------------------------------------------------ #
    # Pre-compute / load prosodic scalar caches
    # ------------------------------------------------------------------ #
    train_scalars = build_scalar_cache(train_df, cache_path / "train_scalars.npy")
    val_scalars = build_scalar_cache(val_df, cache_path / "val_scalars.npy")
    test_scalars = build_scalar_cache(test_df, cache_path / "test_scalars.npy")

    # ------------------------------------------------------------------ #
    # Build DataLoaders
    # ------------------------------------------------------------------ #
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df,
        train_scalars, val_scalars, test_scalars,
        batch_size=batch_size,
        num_workers=num_workers,
        whisper_model_id=whisper_model_id,
    )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    model = WhisperTurnDetector(
        whisper_model_id=whisper_model_id,
        n_unfrozen_encoder_layers=n_unfrozen_encoder_layers,
        turn_end_frames=turn_end_frames,
        dropout=dropout,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log.info("Params: %s trainable / %s total", f"{n_trainable:,}", f"{n_total:,}")

    # ------------------------------------------------------------------ #
    # Loss, optimiser, scheduler, scaler
    # ------------------------------------------------------------------ #
    pos_rate = float(train_df["endpoint_bool"].mean())
    pos_weight = (1.0 - pos_rate) / max(pos_rate, 1e-8)
    log.info("Positive rate: %.4f | pos_weight: %.4f", pos_rate, pos_weight)

    endpoint_loss_fn = FocalLoss(gamma=focal_gamma, pos_weight=pos_weight).to(device)
    aux_loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    steps_per_epoch = max(1, (len(train_loader) + grad_accum_steps - 1) // grad_accum_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * steps_per_epoch
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    history: list[dict] = []
    best_f1 = -1.0

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            endpoint_loss_fn=endpoint_loss_fn,
            aux_loss_fn=aux_loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            grad_accum_steps=grad_accum_steps,
            aux_weight=aux_weight,
            epoch=epoch,
            total_epochs=epochs,
        )

        metrics = evaluate(model, val_loader, device, desc="Validation")
        val_acc, val_f1 = metrics["accuracy"], metrics["f1"]

        history.append(
            {"epoch": epoch + 1, "train_loss": train_loss, "val_acc": val_acc, "val_f1": val_f1}
        )

        log.info(
            "Epoch %d/%d | loss=%.4f | val_acc=%.4f | val_f1=%.4f",
            epoch + 1, epochs, train_loss, val_acc, val_f1,
        )
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"loss={train_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt_path = output_path / "best_model.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → New best F1={best_f1:.4f}  checkpoint: {ckpt_path}")

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_path / "training_history.csv", index=False)
    log.info("Training history saved to %s/training_history.csv", output_dir)

    # Final test-set evaluation using best checkpoint
    best_ckpt = output_path / "best_model.pt"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state)

    print("\n=== Final Test Evaluation ===")
    test_metrics = evaluate_detailed(model, test_loader, device, desc="Test")
    print(f"Test accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test F1:        {test_metrics['f1']:.4f}")
    print(f"Test precision: {test_metrics['precision']:.4f}")
    print(f"Test recall:    {test_metrics['recall']:.4f}")
    print(f"Confusion matrix:\n{np.array(test_metrics['confusion_matrix'])}")
    for lang, lang_m in test_metrics["per_language"].items():
        print(
            f"  [{lang}] N={lang_m['count']:,} | "
            f"acc={lang_m['accuracy']:.4f} | f1={lang_m['f1']:.4f}"
        )

    return history
