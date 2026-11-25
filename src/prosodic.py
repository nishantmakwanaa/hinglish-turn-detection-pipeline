"""
Prosodic feature extraction for TurnSense.

Computes 5 hand-engineered prosodic features from raw audio waveforms:
  1. Energy decay (onset strength of the first 10 frames)
  2. Mean RMS energy
  3. Final/mean RMS ratio (trailing 10 frames vs. mean)
  4. F0 slope (pitch trajectory: rising vs. falling)
  5. Speech-activity ratio (fraction of samples above energy threshold)

These features complement Whisper encoder representations and provide
explicit prosodic signals crucial for detecting turn-completion in
Hinglish speech, where filler words (umm, matlab, toh) can mimic pauses.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False

TARGET_SR = 16_000
FIXED_WINDOW_SECONDS = 8.0


def _f0_slope(y: np.ndarray, sr: int = TARGET_SR, frame_len: int = 2048, hop: int = 512) -> float:
    """
    Estimate the fundamental frequency (F0) slope over the utterance.

    Uses autocorrelation-based pitch detection (no external library needed).
    Returns a normalised slope in [-5, 5] where:
      - Positive → rising pitch (question / mid-turn)
      - Negative → falling pitch (statement / turn-end)

    Args:
        y: Mono audio waveform (float32).
        sr: Sample rate in Hz.
        frame_len: FFT frame length in samples.
        hop: Hop size in samples.

    Returns:
        Scalar F0 slope value.
    """
    if not _LIBROSA_AVAILABLE:
        return 0.0

    try:
        frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop)

        if frames.shape[1] < 10:
            return 0.0

        # Zero-mean each frame
        frames = frames - frames.mean(axis=0, keepdims=True)

        fmin_lag = max(1, int(sr / 1100))   # ~1100 Hz upper bound
        fmax_lag = min(frame_len - 1, int(sr / 65))  # ~65 Hz lower bound

        Fx = np.fft.rfft(frames, n=2 * frame_len, axis=0)
        ac_full = np.fft.irfft(Fx * np.conj(Fx), axis=0)[:frame_len]
        ac = ac_full[fmin_lag:fmax_lag]

        if ac.size == 0:
            return 0.0

        max_power = np.abs(frames).max(axis=0)
        ac_power = ac.max(axis=0)

        valid = (max_power > 1e-4) & (ac_power > 1e-6)

        if valid.sum() < 10:
            return 0.0

        peak_lags = ac[:, valid].argmax(axis=0)
        f0 = sr / (peak_lags.astype(np.float32) + fmin_lag)

        first = float(f0[:5].mean())
        last = float(f0[-5:].mean())

        return float(np.clip((last - first) / (first + 1e-8), -5.0, 5.0))

    except Exception:
        return 0.0


def compute_scalar_features(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Compute 5 prosodic scalar features from a waveform.

    Features (index → description):
      0: Energy decay — onset strength maximum over first 10 onset frames
      1: Mean RMS — mean RMS energy of the waveform (10 ms hop)
      2: Final/mean RMS ratio — trailing-10-frame RMS divided by mean RMS
      3: F0 slope — autocorrelation-based pitch slope estimate
      4: Speech-activity ratio — fraction of samples with |amplitude| > 0.01

    Args:
        y: Mono float32 waveform at ``sr`` Hz. Should be an 8s right-aligned
           window (use :func:`prepare_fixed_window` before calling this).
        sr: Sample rate in Hz. Defaults to 16 000.

    Returns:
        NumPy array of shape ``(5,)`` with dtype float32.
        Safe defaults are returned on any error: ``[0, 0, 1, 0, 0]``.
    """
    out = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    if not _LIBROSA_AVAILABLE:
        return out

    try:
        # Feature 0: onset strength (energy attack intensity)
        onset = librosa.onset.onset_strength(y=y, sr=sr)
        if len(onset):
            out[0] = float(onset[:10].max())

        # Features 1 & 2: mean RMS and trailing/mean ratio
        rms = librosa.feature.rms(y=y, hop_length=160)[0]
        if len(rms) > 10:
            mean_rms = float(rms.mean()) + 1e-8
            out[1] = mean_rms
            out[2] = float(rms[-10:].mean() / mean_rms)

        # Feature 3: F0 slope
        out[3] = _f0_slope(y, sr)

        # Feature 4: speech activity ratio
        out[4] = float(np.mean(np.abs(y) > 0.01))

    except Exception:
        pass

    return out


def resample_audio(
    y: np.ndarray,
    src_sr: int,
    dst_sr: int = TARGET_SR,
) -> np.ndarray:
    """
    Resample a 1-D waveform from ``src_sr`` to ``dst_sr`` Hz.

    Args:
        y: Mono waveform array.
        src_sr: Source sample rate in Hz.
        dst_sr: Target sample rate in Hz.

    Returns:
        Float32 waveform at ``dst_sr`` Hz.
    """
    y = np.asarray(y, dtype=np.float32)
    if src_sr == dst_sr:
        return y
    if not _LIBROSA_AVAILABLE:
        raise ImportError(
            "librosa is required to resample audio from "
            f"{src_sr} Hz to {dst_sr} Hz."
        )
    return librosa.resample(y, orig_sr=src_sr, target_sr=dst_sr).astype(np.float32)


def build_scalar_cache(
    df,
    path: str | Path,
    target_sr: int = TARGET_SR,
    window_seconds: float = FIXED_WINDOW_SECONDS,
) -> np.ndarray:
    """
    Pre-compute and cache prosodic scalar features for every row in a manifest.

    If a cache file already exists and matches the dataframe length, it is
    loaded directly. Otherwise features are computed and the result is saved.

    Args:
        df: Pandas DataFrame with an ``audio_path`` column.
        path: Path to the ``.npy`` cache file.
        target_sr: Expected sample rate; audio is resampled if necessary.
        window_seconds: Fixed-window duration in seconds.

    Returns:
        NumPy array of shape ``(len(df), 5)`` with dtype float32.
    """
    import soundfile as sf
    from tqdm.auto import tqdm

    path = Path(path)

    if path.exists():
        arr = np.load(path)
        if arr.shape == (len(df), 5):
            print(f"Loaded scalar cache: {path}  shape={arr.shape}")
            return arr

    result = np.empty((len(df), 5), dtype=np.float32)

    for i, audio_path in enumerate(tqdm(df["audio_path"], desc=path.stem)):
        y, sr = sf.read(audio_path, dtype="float32")

        if y.ndim == 2:
            y = y.mean(axis=1)

        if sr != target_sr:
            y = resample_audio(y, src_sr=sr, dst_sr=target_sr)

        y = prepare_fixed_window(y, sr=target_sr, seconds=window_seconds)
        result[i] = compute_scalar_features(y, sr=target_sr)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, result)
    print(f"Saved scalar cache: {path}  shape={result.shape}")
    return result


def prepare_fixed_window(
    y: np.ndarray,
    sr: int = TARGET_SR,
    seconds: float = FIXED_WINDOW_SECONDS,
) -> np.ndarray:
    """
    Pad or trim ``y`` to a fixed ``seconds``-long right-aligned window.

    Right-alignment preserves the most recent speech, which is the most
    informative region for turn-end detection.  Audio is normalised to
    0.85 peak amplitude.

    Args:
        y: 1-D float32 waveform.
        sr: Sample rate.
        seconds: Target duration in seconds.

    Returns:
        1-D float32 array of length ``int(sr * seconds)``.
    """
    y = np.asarray(y, dtype=np.float32)
    n = int(sr * seconds)

    if len(y) > n:
        y = y[-n:]                                 # keep trailing (most recent)
    elif len(y) < n:
        y = np.pad(y, (n - len(y), 0))            # zero-pad at the front

    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0:
        y = y / peak * 0.85

    return y.astype(np.float32)
