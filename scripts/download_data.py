#!/usr/bin/env python3
"""
Resumable, constant-memory dataset streaming script for TurnSense.

Streams ``pipecat-ai/smart-turn-data-v3.2-train`` shard-by-shard and
row-group-by-row-group from HuggingFace Hub, extracts embedded audio,
resamples to 16 kHz, right-aligns to 8-second windows, writes PCM_16 WAV
clips, and appends records to a resumable JSONL manifest (with optional CSV
export).
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from huggingface_hub import HfApi, hf_hub_download

try:
    import soundfile as sf
    _SOUNDFILE_AVAILABLE = True
except ImportError:
    sf = None
    _SOUNDFILE_AVAILABLE = False

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.prosodic import (  # noqa: E402
    FIXED_WINDOW_SECONDS,
    TARGET_SR,
    prepare_fixed_window,
    resample_audio,
)


def audio_value_to_bytes(audio_value: Any) -> Optional[bytes]:
    """Return embedded audio bytes when the Parquet audio field contains them."""
    if audio_value is None:
        return None
    if isinstance(audio_value, dict):
        b = audio_value.get("bytes")
        if b is not None:
            return bytes(b)
        return None
    if isinstance(audio_value, (bytes, bytearray, memoryview)):
        return bytes(audio_value)
    return None


def decode_audio_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode common audio containers from raw bytes using soundfile."""
    if not _SOUNDFILE_AVAILABLE:
        raise ImportError("soundfile is required to decode audio bytes.")
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float32), int(sr)


def write_wav(path: Path, y: np.ndarray, sr: int = TARGET_SR) -> None:
    """Write a float32 waveform to a 16-bit PCM WAV file."""
    if not _SOUNDFILE_AVAILABLE:
        raise ImportError("soundfile is required to write WAV files.")
    sf.write(str(path), y.astype(np.float32), sr, subtype="PCM_16")


def _load_existing_manifest(
    manifest_path: Path,
    languages: tuple[str, ...],
) -> tuple[dict[str, int], set[str]]:
    """Read an existing JSONL manifest so interrupted downloads can resume."""
    counts = {lang: 0 for lang in languages}
    existing_ids: set[str] = set()

    if not manifest_path.exists():
        return counts, existing_ids

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            example_id = str(rec["id"])
            existing_ids.add(example_id)
            lang = rec.get("language")
            if lang in counts:
                counts[lang] += 1

    return counts, existing_ids


def _opt_bool(value: Any) -> Optional[bool]:
    return None if value is None else bool(value)


def download_and_curate(
    dataset_id: str,
    languages: tuple[str, ...],
    per_language_cap: int,
    audio_dir: Path,
    manifest_path: Path,
    shards_dir: Path,
    csv_manifest_path: Optional[Path] = None,
    target_sr: int = TARGET_SR,
    window_seconds: float = FIXED_WINDOW_SECONDS,
) -> dict[str, int]:
    """
    Stream Parquet shards row-group by row-group and curate fixed-window WAV clips.
    """
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    files = api.list_repo_files(repo_id=dataset_id, repo_type="dataset")
    parquet_files = sorted(
        f for f in files if f.startswith("data/") and f.endswith(".parquet")
    )
    if not parquet_files:
        raise RuntimeError(f"No Parquet shards found in dataset {dataset_id!r}.")

    print(f"Found {len(parquet_files)} Parquet shards in {dataset_id}")

    counts, existing_ids = _load_existing_manifest(manifest_path, languages)
    print(f"Existing manifest counts: {counts} (unique IDs: {len(existing_ids):,})")

    for shard_num, shard_name in enumerate(parquet_files, start=1):
        if all(counts[lang] >= per_language_cap for lang in languages):
            break

        print(f"\n===== SHARD {shard_num}/{len(parquet_files)}: {shard_name} =====")
        print(f"Current counts: {counts}")

        shard_path = hf_hub_download(
            repo_id=dataset_id,
            filename=shard_name,
            repo_type="dataset",
            local_dir=str(shards_dir),
        )

        try:
            pf = pq.ParquetFile(shard_path)

            # Process one row group at a time to keep RAM constant.
            for rg in range(pf.num_row_groups):
                if all(counts[lang] >= per_language_cap for lang in languages):
                    break

                table = pf.read_row_group(rg)
                records = table.to_pylist()

                for row in records:
                    lang = row.get("language")
                    if lang not in counts or counts[lang] >= per_language_cap:
                        continue

                    example_id = str(row["id"])
                    if example_id in existing_ids:
                        continue

                    audio_bytes = audio_value_to_bytes(row.get("audio"))
                    if audio_bytes is None:
                        raise RuntimeError(
                            f"Audio field was not embedded as bytes in row {example_id}."
                        )

                    try:
                        y, sr = decode_audio_bytes(audio_bytes)
                        y = resample_audio(y, src_sr=sr, dst_sr=target_sr)
                        y = prepare_fixed_window(y, sr=target_sr, seconds=window_seconds)

                        audio_path = audio_dir / f"{example_id}.wav"
                        write_wav(audio_path, y, sr=target_sr)
                    except Exception as exc:
                        print(f"Skipping bad audio {example_id}: {type(exc).__name__}: {exc}")
                        continue

                    record = {
                        "id": example_id,
                        "audio_path": str(audio_path),
                        "language": lang,
                        "endpoint_bool": bool(row.get("endpoint_bool")),
                        "midfiller": _opt_bool(row.get("midfiller")),
                        "endfiller": _opt_bool(row.get("endfiller")),
                        "synthetic": _opt_bool(row.get("synthetic")),
                        "spoken_text": row.get("spoken_text"),
                        "dataset": row.get("dataset"),
                    }

                    with open(manifest_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    existing_ids.add(example_id)
                    counts[lang] += 1

                    total = sum(counts.values())
                    if total % 500 == 0:
                        summary = " | ".join(f"{k}={v:,}" for k, v in counts.items())
                        print(f"selected={total:,} | {summary}")

                    if all(counts[k] >= per_language_cap for k in languages):
                        break
        finally:
            try:
                os.remove(shard_path)
            except OSError:
                pass

        gc.collect()

    if csv_manifest_path is not None and manifest_path.exists():
        csv_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        pd.read_json(manifest_path, lines=True).to_csv(csv_manifest_path, index=False)
        print(f"Exported CSV manifest: {csv_manifest_path}")

    print("\n========================================")
    print("SELECTION COMPLETE / CURRENT STATE")
    print("========================================")
    for lang in languages:
        print(f"{lang:>7}: {counts[lang]:,}")
    print(f"  Total: {sum(counts.values()):,}")
    print(f"Manifest: {manifest_path}")

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream HuggingFace smart-turn Parquet shards and curate WAV clips."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument("--dataset-id", type=str, default=None, help="HuggingFace dataset ID.")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Language codes to curate (e.g. hin eng).",
    )
    parser.add_argument(
        "--per-language-cap",
        type=int,
        default=None,
        help="Maximum clips to collect per language.",
    )
    parser.add_argument("--audio-dir", type=Path, default=None, help="Output WAV directory.")
    parser.add_argument("--manifest", type=Path, default=None, help="Output JSONL manifest path.")
    parser.add_argument("--csv-manifest", type=Path, default=None, help="Optional CSV manifest path.")
    parser.add_argument("--shards-dir", type=Path, default=None, help="Temp directory for shards.")
    parser.add_argument("--target-sr", type=int, default=None, help="Target sample rate in Hz.")
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="Right-aligned clip window duration in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg: dict[str, Any] = {}
    if args.config and args.config.exists():
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    data_cfg = cfg.get("data", {})

    dataset_id = args.dataset_id or data_cfg.get(
        "dataset_id", "pipecat-ai/smart-turn-data-v3.2-train"
    )
    languages = tuple(args.languages or data_cfg.get("languages", ["hin", "eng"]))
    per_language_cap = (
        args.per_language_cap
        if args.per_language_cap is not None
        else int(data_cfg.get("per_language_cap", 20_000))
    )
    audio_dir = args.audio_dir or Path(data_cfg.get("audio_dir", "smart_turn/audio"))
    manifest_path = args.manifest or Path(data_cfg.get("manifest_path", "smart_turn/manifest.jsonl"))
    csv_manifest_path = args.csv_manifest or (
        Path(data_cfg["csv_manifest_path"]) if data_cfg.get("csv_manifest_path") else None
    )
    shards_dir = args.shards_dir or Path(data_cfg.get("shards_dir", "smart_turn/shards"))
    target_sr = args.target_sr or int(data_cfg.get("target_sr", TARGET_SR))
    window_seconds = args.window_seconds or float(
        data_cfg.get("fixed_window_seconds", FIXED_WINDOW_SECONDS)
    )

    download_and_curate(
        dataset_id=dataset_id,
        languages=languages,
        per_language_cap=per_language_cap,
        audio_dir=audio_dir,
        manifest_path=manifest_path,
        shards_dir=shards_dir,
        csv_manifest_path=csv_manifest_path,
        target_sr=target_sr,
        window_seconds=window_seconds,
    )


if __name__ == "__main__":
    main()
