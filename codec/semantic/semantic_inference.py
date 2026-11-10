"""Decode a full sequence with trained semantic codec checkpoints.

The script writes one MP4 per reconstruction layer. It also writes a tiled
``grid.mp4`` with the original frame and every progressive/early-exit layer.
Frames from all available hierarchy levels are merged in frame order, so the
videos cover the whole sequence once levels 0, 1, and 2 are trained.

``stepNN`` images are cumulative: ``step04`` is the DecoderCell2 reconstruction
from the first four progressive codes, not layer 4 alone.
"""

import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/nfs/home/olidiz18/git/S-SWIFT")
SRC = REPO / "codec" / "semantic"
sys.path.insert(0, str(SRC))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

from dataset import ImageFolder
from train_options import parser
from util import eval_forward, get_models, set_eval

import network

DATASET, SEQ = "derf", "silent"
ABLATION = "C"
LEVELS = [0]
CHECKPOINT_ITER = None
OUTPUT_FPS = 10
WRITE_GRID = True
SAVE_PNG = False
COMPUTE_MSSSIM = True   # MS-SSIM is slow at 1080p; enable on short clips
METRIC_STRIDE = 1        # compute PSNR every Nth frame
MSSSIM_STRIDE = 50       # compute MS-SSIM every Nth metric frame

OUT_DIR = REPO / "codec/semantic/inference" / f"{DATASET}_{SEQ}" / ABLATION

HIERARCHIES = {
    0: dict(distance1=6, distance2=6, bits=16,
            encoder_fuse_level=1, decoder_fuse_level=1),
    1: dict(distance1=3, distance2=3, bits=16,
            encoder_fuse_level=2, decoder_fuse_level=3),
    2: dict(distance1=1, distance2=2, bits=8,
            encoder_fuse_level=1, decoder_fuse_level=1),
}

CHECKPOINT_NAMES = ["encoder", "binarizer", "decoder", "d2", "unet"]


def clip_paths():
    return (
        REPO / "data" / DATASET / "frames" / SEQ,
        REPO / "data" / DATASET / "mv" / SEQ,
        REPO / "data" / DATASET / "masks" / SEQ,
        REPO / "data" / DATASET / "edges" / SEQ,
    )


def build_args(level, frames, mv, masks, edges):
    cfg = HIERARCHIES[level]
    argv = [
        "--train", str(frames), "--eval", str(frames),
        "--train-mv", str(mv), "--eval-mv", str(mv),
        "--v-compress", "--warp", "--stack", "--fuse-encoder",
        "--bits", str(cfg["bits"]),
        "--distance1", str(cfg["distance1"]),
        "--distance2", str(cfg["distance2"]),
        "--encoder-fuse-level", str(cfg["encoder_fuse_level"]),
        "--decoder-fuse-level", str(cfg["decoder_fuse_level"]),
        "--batch-size", "1",
    ]
    if ABLATION in ("B", "C"):
        argv += [
            "--train-mask", str(masks), "--eval-mask", str(masks),
            "--train-edge", str(edges), "--eval-edge", str(edges),
            "--alpha-sem", "1.0", "--alpha-edge", "0.5",
        ]
    if ABLATION == "C":
        argv += ["--use-conditioning"]
    return parser.parse_args(argv)


def latest_iter(model_dir, level):
    prefix = f"hierarchy{level}"
    paths = [
        p for p in model_dir.glob(f"{prefix}_encoder_*.pth")
        if p.stem.rsplit("_", 1)[-1].isdigit()
    ]
    if not paths:
        raise FileNotFoundError(f"no checkpoint in {model_dir}")
    return max(int(p.stem.rsplit("_", 1)[-1]) for p in paths)


def load_models(level, args, iteration):
    model_dir = (REPO / "codec/semantic/experiments"
                 / f"hierarchy_{level}" / ABLATION / "model")
    paths = {
        name: model_dir / f"hierarchy{level}_{name}_{iteration:08d}.pth"
        for name in CHECKPOINT_NAMES
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("missing checkpoint(s): " + ", ".join(missing))

    encoder, binarizer, decoder, unet = get_models(
        args=args, v_compress=args.v_compress, bits=args.bits,
        encoder_fuse_level=args.encoder_fuse_level,
        decoder_fuse_level=args.decoder_fuse_level)
    d2 = network.DecoderCell2(
        v_compress=args.v_compress, shrink=args.shrink, bits=args.bits,
        fuse_level=args.decoder_fuse_level,
        use_conditioning=args.use_conditioning).cuda()

    nets = [encoder, binarizer, decoder, d2]
    if unet is not None:
        nets.append(unet)

    for name, net in zip(CHECKPOINT_NAMES, nets):
        net.load_state_dict(torch.load(paths[name], map_location="cuda:0"))
    set_eval(nets)
    return nets


def frame_index(path):
    return int(Path(path).stem.rsplit("_", 1)[-1])


def to_uint8_hwc(frame_chw):
    arr = np.clip(np.asarray(frame_chw), 0.0, 1.0)
    return (arr * 255.0).round().astype(np.uint8).transpose(1, 2, 0)


def bgr_from_rgb(frame_hwc):
    return cv2.cvtColor(frame_hwc, cv2.COLOR_RGB2BGR)


def psnr_value(reference, compared):
    diff = reference.astype(np.float64) - compared.astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse <= 0.0:
        return 99.99
    return float(min(99.99, 10.0 * math.log10(255.0 * 255.0 / mse)))


class VideoSet:
    def __init__(self, out_dir, fps):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.writers = {}
        self.ffmpeg = shutil.which("ffmpeg")
        self.use_h264 = False
        if self.ffmpeg:
            try:
                result = subprocess.run(
                    [self.ffmpeg, "-hide_banner", "-encoders"],
                    capture_output=True, text=True, timeout=15)
                self.use_h264 = "libx264" in result.stdout
            except Exception:
                self.use_h264 = False
        if not self.use_h264:
            print("Warning: ffmpeg/libx264 is not available; writing MPEG-4 "
                  "Part 2, which may not play in VS Code.")

    def write(self, name, frame_bgr):
        entry = self.writers.get(name)
        if entry is None:
            height, width = frame_bgr.shape[:2]
            path = self.out_dir / f"{name}.mp4"
            if self.use_h264:
                command = [
                    self.ffmpeg, "-y", "-loglevel", "error",
                    "-f", "rawvideo", "-pix_fmt", "bgr24",
                    "-s", f"{width}x{height}", "-r", str(self.fps),
                    "-i", "-", "-an",
                    "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "16", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(path),
                ]
                process = subprocess.Popen(command, stdin=subprocess.PIPE)
                entry = ("ffmpeg", process, path)
            else:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(path), fourcc, self.fps,
                                         (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"cannot open video writer: {path}")
                entry = ("opencv", writer, path)
            self.writers[name] = entry

        if entry[0] == "ffmpeg":
            entry[1].stdin.write(np.ascontiguousarray(frame_bgr).tobytes())
        else:
            entry[1].write(frame_bgr)

    def close(self):
        for kind, handle, path in self.writers.values():
            if kind == "ffmpeg":
                try:
                    handle.stdin.close()
                except BrokenPipeError:
                    pass
                code = handle.wait()
                if code != 0:
                    print(f"Warning: ffmpeg failed for {path} (code {code})")
            else:
                handle.release()
        self.writers.clear()


def make_grid(frames_bgr, labels, cols=5, panel_width=384):
    height, width = frames_bgr[0].shape[:2]
    panel_w = min(panel_width, width)
    panel_h = max(1, int(round(height * panel_w / width)))
    rows = (len(frames_bgr) + cols - 1) // cols
    grid = np.full((rows * panel_h, cols * panel_w, 3), 255, np.uint8)

    for i, (image, label) in enumerate(zip(frames_bgr, labels)):
        panel = cv2.resize(image, (panel_w, panel_h),
                           interpolation=cv2.INTER_AREA)
        cv2.putText(panel, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(panel, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y = (i // cols) * panel_h
        x = (i % cols) * panel_w
        grid[y:y + panel_h, x:x + panel_w] = panel
    return grid


def main():
    frames, mv, masks, edges = clip_paths()
    if not frames.is_dir():
        raise SystemExit(f"frames directory not found: {frames}")
    if not mv.is_dir():
        raise SystemExit(f"motion-vector directory not found: {mv}")
    if ABLATION in ("B", "C") and not masks.is_dir():
        print(f"Warning: mask directory not found (zero-fill): {masks}")
    if ABLATION in ("B", "C") and not edges.is_dir():
        print(f"Warning: edge directory not found (zero-fill): {edges}")

    models = {}
    args_by_level = {}
    datasets = {}
    items = []

    for level in LEVELS:
        args = build_args(level, frames, mv, masks, edges)
        model_dir = (REPO / "codec/semantic/experiments"
                     / f"hierarchy_{level}" / ABLATION / "model")
        try:
            iteration = (CHECKPOINT_ITER if CHECKPOINT_ITER is not None
                         else latest_iter(model_dir, level))
            nets = load_models(level, args, iteration)
        except FileNotFoundError as exc:
            print(f"Skipping hierarchy {level}: {exc}")
            continue

        dataset = ImageFolder(is_train=False, root=str(frames),
                              mv_dir=str(mv), args=args)
        if len(dataset) == 0:
            print(f"Skipping hierarchy {level}: no evaluable frames")
            continue
        dataset.imgs.sort(key=frame_index)

        models[level] = nets
        args_by_level[level] = args
        datasets[level] = dataset
        for index, path in enumerate(dataset.imgs):
            items.append((frame_index(path), level, index))
        print(f"Hierarchy {level}: {len(dataset)} frames, "
              f"checkpoint {iteration}")

    if not items:
        raise SystemExit("No hierarchy level has a usable checkpoint.")

    items.sort(key=lambda item: item[0])
    missing_levels = [level for level in LEVELS if level not in models]
    if missing_levels:
        print(f"Warning: no usable checkpoint for levels {missing_levels}; "
              "their GOP frames are absent from the videos.")

    video = VideoSet(OUT_DIR, OUTPUT_FPS)
    written = 0
    num_steps = None
    metric_frames = 0
    metric_order = []
    psnr_sum = defaultdict(float)
    psnr_count = defaultdict(int)
    msssim_sum = defaultdict(float)
    msssim_count = defaultdict(int)
    msssim_fn = None
    msssim_warned = False
    if COMPUTE_MSSSIM:
        from metric import msssim as msssim_fn
    try:
        for frame_idx, level, index in items:
            data, ctx_frames, _ = datasets[level][index]
            with torch.inference_mode():
                (original, out_imgs, ee1, ee2, ee3, ee4,
                 losses, codes) = eval_forward(
                    models[level],
                    (data.unsqueeze(0).cuda(non_blocking=True),
                     ctx_frames.unsqueeze(0)),
                    args_by_level[level])

            num_steps = out_imgs.shape[0]
            rgb_layers = {"original": to_uint8_hwc(original[0])}
            for step in range(num_steps):
                rgb_layers[f"step{step + 1:02d}"] = to_uint8_hwc(
                    out_imgs[step][0])
            for idx, ee in enumerate((ee1, ee2, ee3, ee4), 1):
                rgb_layers[f"ee{idx}"] = to_uint8_hwc(ee[-1][0])

            layers_bgr = {
                name: bgr_from_rgb(image) for name, image in rgb_layers.items()
            }

            for name, image in layers_bgr.items():
                video.write(name, image)

            if WRITE_GRID:
                labels = ["original"] + [
                    f"step{step + 1:02d}" for step in range(num_steps)
                ] + [f"ee{idx}" for idx in range(1, 5)]
                grid = make_grid([layers_bgr[name] for name in labels], labels)
                video.write("grid", grid)

            if SAVE_PNG:
                for name, image in layers_bgr.items():
                    path = (OUT_DIR / "frames" / name
                            / f"frame_{frame_idx:04d}.png")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(path), image)

            if not metric_order:
                metric_order = [
                    name for name in rgb_layers if name != "original"
                ]

            if written % METRIC_STRIDE == 0:
                reference = rgb_layers["original"]
                for name in metric_order:
                    psnr_sum[name] += psnr_value(reference, rgb_layers[name])
                    psnr_count[name] += 1
                    if msssim_fn is not None and written % MSSSIM_STRIDE == 0:
                        try:
                            msssim_sum[name] += float(
                                msssim_fn(reference, rgb_layers[name]))
                            msssim_count[name] += 1
                        except Exception as exc:
                            if not msssim_warned:
                                print(f"Warning: MS-SSIM failed: {exc}")
                                msssim_warned = True
                metric_frames += 1

            written += 1
            if written % 10 == 0:
                print(f"Decoded {written}/{len(items)} frames")

    finally:
        video.close()

    print(f"Wrote {written} frames to {OUT_DIR}")
    print("Videos:", ", ".join(sorted(p.name for p in OUT_DIR.glob("*.mp4"))))

    if metric_frames:
        print(f"\nPer-layer metrics ({metric_frames} frames):")
        header = f"{'layer':<8} {'PSNR(dB)':>10}"
        if msssim_fn is not None:
            header += f" {'MS-SSIM':>10}"
        print(header)
        for name in metric_order:
            if psnr_count[name] == 0:
                continue
            mean_psnr = psnr_sum[name] / psnr_count[name]
            line = f"{name:<8} {mean_psnr:>10.3f}"
            if msssim_fn is not None and msssim_count[name]:
                line += f" {msssim_sum[name] / msssim_count[name]:>10.5f}"
            print(line)

        step_names = [
            name for name in metric_order
            if name.startswith("step") and psnr_count[name]
        ]
        if step_names:
            first, last = step_names[0], step_names[-1]
            gain = (psnr_sum[last] / psnr_count[last]
                    - psnr_sum[first] / psnr_count[first])
            print(f"PSNR gain {first} -> {last}: {gain:+.3f} dB")
            means = [psnr_sum[name] / psnr_count[name] for name in step_names]
            monotonic = all(
                later >= earlier - 1e-9
                for earlier, later in zip(means, means[1:])
            )
            print(f"PSNR monotonic {first}..{last}: {monotonic}")


if __name__ == "__main__":
    main()
