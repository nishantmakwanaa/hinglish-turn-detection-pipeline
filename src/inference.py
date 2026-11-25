"""
Real-time inference utilities for TurnSense.

Provides :class:`TurnDetector` for running single-clip or live microphone
stream inference with :class:`~src.model.WhisperTurnDetector`, as well as
Gradio demo helpers matching the notebook's interactive mic interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from .dataset import get_feature_extractor
from .model import WhisperTurnDetector
from .prosodic import (
    FIXED_WINDOW_SECONDS,
    TARGET_SR,
    compute_scalar_features,
    prepare_fixed_window,
    resample_audio,
)


def _resolve_device(device: Optional[str | torch.device]) -> torch.device:
    """Resolve ``None`` or ``"auto"`` to CUDA if available, otherwise CPU."""
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class TurnDetector:
    """
    Real-time Hinglish turn-detection inference wrapper.

    Loads a trained :class:`~src.model.WhisperTurnDetector` checkpoint and
    performs end-to-end preprocessing (mono conversion, amplitude scaling,
    16 kHz resampling, 8 s right-aligned windowing, Whisper log-mel
    extraction, and 5 prosodic scalar features) to predict turn completion
    and auxiliary filler probabilities.

    Args:
        checkpoint_path: Path to a saved ``state_dict`` checkpoint
            (e.g. ``best_model.pt``). If ``None``, initializes fresh weights.
        whisper_model_id: HuggingFace Whisper backbone identifier.
        n_unfrozen_encoder_layers: Number of trailing encoder layers in model.
        turn_end_frames: Number of trailing encoder frames in the tail branch.
        dropout: Dropout rate used when constructing the architecture.
        threshold: Probability threshold for ``is_turn_end`` decisions.
        target_sr: Expected model sample rate in Hz.
        window_seconds: Right-aligned audio window length in seconds.
        device: Target torch device (``"auto"``, ``"cpu"``, ``"cuda"``, etc.).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        whisper_model_id: str = "openai/whisper-tiny",
        n_unfrozen_encoder_layers: int = 2,
        turn_end_frames: int = 100,
        dropout: float = 0.1,
        threshold: float = 0.5,
        target_sr: int = TARGET_SR,
        window_seconds: float = FIXED_WINDOW_SECONDS,
        device: Optional[str | torch.device] = "auto",
    ) -> None:
        self.device = _resolve_device(device)
        self.threshold = float(threshold)
        self.target_sr = int(target_sr)
        self.window_seconds = float(window_seconds)
        self.whisper_model_id = whisper_model_id

        self.feature_extractor = get_feature_extractor(whisper_model_id)
        self.model = WhisperTurnDetector(
            whisper_model_id=whisper_model_id,
            n_unfrozen_encoder_layers=n_unfrozen_encoder_layers,
            turn_end_frames=turn_end_frames,
            dropout=dropout,
        ).to(self.device)

        if checkpoint_path is not None:
            ckpt = Path(checkpoint_path)
            if not ckpt.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
            state = torch.load(ckpt, map_location=self.device, weights_only=True)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            self.model.load_state_dict(state)

        self.model.eval()

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        checkpoint_path: Optional[str | Path] = None,
        device: Optional[str | torch.device] = None,
        threshold: Optional[float] = None,
    ) -> "TurnDetector":
        """
        Instantiate :class:`TurnDetector` from a parsed YAML config dictionary.
        """
        data_cfg = config.get("data", {})
        model_cfg = config.get("model", {})
        infer_cfg = config.get("inference", {})

        ckpt = checkpoint_path if checkpoint_path is not None else infer_cfg.get("checkpoint_path")
        resolved_device = device if device is not None else infer_cfg.get("device", "auto")
        resolved_thresh = threshold if threshold is not None else infer_cfg.get("threshold", 0.5)

        return cls(
            checkpoint_path=ckpt,
            whisper_model_id=model_cfg.get("whisper_model_id", "openai/whisper-tiny"),
            n_unfrozen_encoder_layers=model_cfg.get("n_unfrozen_encoder_layers", 2),
            turn_end_frames=model_cfg.get("turn_end_frames", 100),
            dropout=model_cfg.get("dropout", 0.1),
            threshold=resolved_thresh,
            target_sr=data_cfg.get("target_sr", TARGET_SR),
            window_seconds=data_cfg.get("fixed_window_seconds", FIXED_WINDOW_SECONDS),
            device=resolved_device,
        )

    def _prepare_waveform(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Normalize, convert to mono, resample, and right-align to fixed window."""
        y = np.asarray(waveform)

        if y.ndim == 2:
            # Handle both (samples, channels) and (channels, samples)
            if y.shape[0] <= 8 and y.shape[1] > y.shape[0]:
                y = y.mean(axis=0)
            else:
                y = y.mean(axis=1)

        y = y.astype(np.float32, copy=False)

        # Normalize integer-scaled microphone waveforms (e.g. Gradio int16)
        max_amp = float(np.abs(y).max()) if y.size else 0.0
        if max_amp > 1.0:
            y = y / (max_amp + 1e-8)

        if sr != self.target_sr:
            y = resample_audio(y, src_sr=int(sr), dst_sr=self.target_sr)

        return prepare_fixed_window(y, sr=self.target_sr, seconds=self.window_seconds)

    @torch.no_grad()
    def predict_array(
        self,
        waveform: np.ndarray,
        sr: int = TARGET_SR,
        threshold: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Run turn-detection inference on an in-memory waveform or mic chunk.

        Args:
            waveform: 1-D or 2-D audio array.
            sr: Sample rate of ``waveform`` in Hz.
            threshold: Optional override for the classification threshold.

        Returns:
            Dictionary containing:
              - ``"is_turn_end"``: bool decision
              - ``"endpoint_prob"``: float probability in ``[0, 1]``
              - ``"midfiller_prob"``: float probability of mid-utterance filler
              - ``"endfiller_prob"``: float probability of trailing filler
              - ``"threshold"``: threshold used for ``is_turn_end``
              - ``"verdict"``: human-readable status string
        """
        thresh = self.threshold if threshold is None else float(threshold)
        prepared = self._prepare_waveform(waveform, sr=sr)

        feats = self.feature_extractor(
            prepared,
            sampling_rate=self.target_sr,
            return_tensors="pt",
        )["input_features"].to(self.device)

        scalar_values = compute_scalar_features(prepared, sr=self.target_sr)
        scalars = torch.tensor(scalar_values, dtype=torch.float32).unsqueeze(0).to(self.device)

        use_amp = self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
            out = self.model(feats, scalars)

        endpoint_prob = float(torch.sigmoid(out["endpoint_logits"]).float().item())
        midfiller_prob = float(torch.sigmoid(out["midfiller_logits"]).float().item())
        endfiller_prob = float(torch.sigmoid(out["endfiller_logits"]).float().item())
        is_turn_end = endpoint_prob >= thresh

        return {
            "is_turn_end": is_turn_end,
            "endpoint_prob": endpoint_prob,
            "midfiller_prob": midfiller_prob,
            "endfiller_prob": endfiller_prob,
            "threshold": thresh,
            "verdict": (
                "END OF TURN -- respond now"
                if is_turn_end
                else "STILL SPEAKING -- keep listening"
            ),
        }

    def predict_file(
        self,
        audio_path: str | Path,
        threshold: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Load an audio file from disk and run turn-detection inference.

        Args:
            audio_path: Path to a WAV/FLAC/OGG audio file readable by ``soundfile``.
            threshold: Optional override for the classification threshold.

        Returns:
            Prediction dictionary (see :meth:`predict_array`).
        """
        try:
            import soundfile as sf
        except ImportError as exc:
            raise ImportError(
                "soundfile is required to read audio files in TurnDetector.predict_file."
            ) from exc

        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        waveform, sr = sf.read(str(path), dtype="float32", always_2d=False)
        result = self.predict_array(waveform, sr=sr, threshold=threshold)
        result["audio_path"] = str(path)
        return result

    def predict_gradio(
        self,
        audio: Optional[tuple[int, np.ndarray]],
    ) -> tuple[str, str, str]:
        """
        Gradio callback compatible with ``gr.Audio(type="numpy")``.

        Returns:
            Tuple of ``(verdict, endpoint_probability_text, filler_probabilities_text)``.
        """
        if audio is None:
            return "No audio provided.", "", ""

        sr, waveform = audio
        res = self.predict_array(waveform, sr=sr)
        prob_text = f"P(end of turn) = {res['endpoint_prob']:.3f}"
        filler_text = (
            f"P(mid-filler) = {res['midfiller_prob']:.3f} | "
            f"P(end-filler) = {res['endfiller_prob']:.3f}"
        )
        return res["verdict"], prob_text, filler_text


def build_gradio_demo(detector: TurnDetector):
    """
    Build a Gradio Interface for live microphone and uploaded-clip testing.

    Args:
        detector: Initialized :class:`TurnDetector` instance.

    Returns:
        Configured ``gradio.Interface`` object.
    """
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError(
            "gradio is required for the interactive demo. Install it via `pip install gradio`."
        ) from exc

    return gr.Interface(
        fn=detector.predict_gradio,
        inputs=gr.Audio(
            sources=["microphone", "upload"],
            type="numpy",
            label="Speak or upload audio",
        ),
        outputs=[
            gr.Textbox(label="Verdict"),
            gr.Textbox(label="Probability"),
            gr.Textbox(label="Auxiliary Filler Predictions"),
        ],
        title="TurnSense: Hinglish Audio Turn Detector",
        description=(
            "Record speech or upload an audio file to detect whether the speaker "
            "has finished their turn (with auxiliary mid/end filler predictions)."
        ),
    )


def launch_gradio_demo(
    detector: TurnDetector,
    share: bool = False,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
) -> None:
    """Build and launch the TurnSense Gradio demo."""
    demo = build_gradio_demo(detector)
    demo.launch(share=share, server_name=server_name, server_port=server_port)
