#!/usr/bin/env python3
"""
CLI inference and Gradio demo launcher for TurnSense.

Supports running inference on one or more audio files via ``--audio`` or
launching the interactive Gradio microphone demo via ``--demo``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference import TurnDetector, launch_gradio_demo  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TurnSense inference on audio files or launch the Gradio mic demo."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--audio",
        nargs="+",
        type=Path,
        default=None,
        help="One or more audio file paths to classify.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold for turn completion (default: 0.5).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device ('auto', 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print file inference results as JSON.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Launch the interactive Gradio microphone/upload demo.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link when running --demo.",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default=None,
        help="Gradio server bind address.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=None,
        help="Gradio server port.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.audio and not args.demo:
        print("Error: specify --audio <file...> and/or --demo.", file=sys.stderr)
        sys.exit(2)

    cfg: dict[str, Any] = {}
    if args.config and args.config.exists():
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    detector = TurnDetector.from_config(
        cfg,
        checkpoint_path=args.checkpoint,
        device=args.device,
        threshold=args.threshold,
    )

    if args.audio:
        results = [detector.predict_file(path) for path in args.audio]
        if args.json:
            print(json.dumps(results if len(results) > 1 else results[0], indent=2))
        else:
            for res in results:
                print(
                    f"[{res['audio_path']}] {res['verdict']} | "
                    f"P(end)={res['endpoint_prob']:.3f} | "
                    f"P(mid_filler)={res['midfiller_prob']:.3f} | "
                    f"P(end_filler)={res['endfiller_prob']:.3f}"
                )

    if args.demo:
        demo_cfg = cfg.get("inference", {}).get("demo", {})
        share = args.share or bool(demo_cfg.get("share", False))
        server_name = args.server_name or str(demo_cfg.get("server_name", "127.0.0.1"))
        server_port = args.server_port or int(demo_cfg.get("server_port", 7860))
        launch_gradio_demo(
            detector,
            share=share,
            server_name=server_name,
            server_port=server_port,
        )


if __name__ == "__main__":
    main()
