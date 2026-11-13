#!/usr/bin/env python3
"""Check that this machine can run the project (versions, CUDA, models)."""

from __future__ import annotations

import argparse
import platform
import sys
from importlib import metadata

from _bootstrap import add_config_args

from isl.config import ConfigError, load_config

PACKAGES = [
    "torch",
    "torchvision",
    "mediapipe",
    "opencv-python",
    "opencv-contrib-python",
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "tqdm",
    "PyYAML",
]


def version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "MISSING"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Load both MediaPipe landmarkers and run one blank frame",
    )
    args = parser.parse_args()

    problems: list[str] = []
    print(f"python   : {sys.version.split()[0]}")
    print(f"platform : {platform.platform()}")
    for pkg in PACKAGES:
        v = version(pkg)
        print(f"{pkg:<22}: {v}")
        if pkg in {"torch", "mediapipe", "numpy", "PyYAML"} and v == "MISSING":
            problems.append(f"{pkg} is not installed")

    try:
        import torch

        print(f"{'cuda available':<22}: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"{'gpu':<22}: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"{'vram':<22}: {props.total_memory / 1e9:.1f} GB")
        else:
            print("  WARNING: CUDA is not available; training will be slow on CPU.")
    except Exception as exc:
        problems.append(f"torch check failed: {exc}")

    try:
        cfg = load_config(args.config, root=args.root)
        print()
        print(cfg.summary())
        print()
        for name in ("hand", "pose"):
            spec = cfg.get(f"keypoints.models.{name}")
            dest = cfg.path("keypoints.models_dir") / str(spec["file"])
            mark = "OK" if dest.is_file() and dest.stat().st_size > 1_000_000 else "MISSING"
            print(f"model {name:<5}: {mark:<7} {dest}")
            if mark == "MISSING":
                problems.append(f"MediaPipe model missing: {dest} (run scripts/fetch_models.py)")

        if args.smoke_test:
            import numpy as np
            import mediapipe as mp
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision

            image = np.zeros((480, 640, 3), dtype=np.uint8)
            image[100:300, 200:400] = 180
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

            hand_dest = cfg.path("keypoints.models_dir") / str(
                cfg.get("keypoints.models.hand.file")
            )
            pose_dest = cfg.path("keypoints.models_dir") / str(
                cfg.get("keypoints.models.pose.file")
            )
            hand = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=mpp.BaseOptions(model_asset_path=str(hand_dest)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=int(cfg.get("keypoints.num_hands", 2)),
                )
            )
            pose = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=mpp.BaseOptions(model_asset_path=str(pose_dest)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_poses=1,
                )
            )
            hand.detect_for_video(mp_image, 0)
            pose.detect_for_video(mp_image, 0)
            hand.close()
            pose.close()
            print("\nMediaPipe smoke test: OK")
    except ConfigError as exc:
        problems.append(str(exc))
    except Exception as exc:
        problems.append(f"smoke test failed: {type(exc).__name__}: {exc}")

    print()
    if problems:
        print("Problems:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("Environment check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
