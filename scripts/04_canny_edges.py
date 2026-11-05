#!/usr/bin/env python
"""scripts/04_canny_edges.py

Extract binary Canny edge maps for every clip in the manifests.

For each frame_%04d.png, compute cv2.Canny(img, CANNY_LOW, CANNY_HIGH) on the
grayscale image and write the binarized result (0/255) to the matching
<dataset>/edges/<clip>/frame_%04d.png at native resolution.

Usage:
  python scripts/04_canny_edges.py [--dataset kinetics|derf|uvg]
                                   [--split train|test] [--limit N] [--list]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import config  # noqa: E402


def process_clip(clip: dict, low: int, high: int, bar) -> None:
    frames_dir = clip["frames_dir"]
    out_dir = clip["edges_dir"]
    frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not frames:
        bar.log(f"no frames: {clip}")
        return
    os.makedirs(out_dir, exist_ok=True)

    n = 0
    for frame_path in frames:
        img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        edges = cv2.Canny(img, low, high)
        binary = (edges > 0).astype(np.uint8) * 255
        out_path = os.path.join(out_dir, os.path.basename(frame_path))
        cv2.imwrite(out_path, binary)
        n += 1

    bar.log(f"edges {clip.get('dataset', 'kinetics')}/{clip.get('split', '')}/"
            f"{clip.get('clip_id') or clip.get('seq')}: {n} frames")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["kinetics", "derf", "uvg"],
                        default=None)
    parser.add_argument("--split", choices=["train", "test"], default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    cfg = config.load_config()

    clips: list[dict] = []
    if args.dataset in (None, "kinetics"):
        clips += config.kinetics_clips(cfg, split=args.split)
    if args.dataset in (None, "derf"):
        clips += config.test_clips(cfg, "derf")
    if args.dataset in (None, "uvg"):
        clips += config.test_clips(cfg, "uvg")
    if args.limit:
        clips = clips[:args.limit]

    if args.list:
        for clip in clips:
            print(clip["frames_dir"], "->", clip["edges_dir"])
        return 0

    if not clips:
        print("no clips found; run 01_download_datasets.sh first", file=sys.stderr)
        return 1

    low = int(cfg["CANNY_LOW"])
    high = int(cfg["CANNY_HIGH"])
    bar = config.Progress(len(clips))
    try:
        for idx, clip in enumerate(clips, 1):
            bar.tick(idx)
            process_clip(clip, low, high, bar)
    finally:
        bar.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
