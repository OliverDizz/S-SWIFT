#!/usr/bin/env python
"""scripts/03_sam3_masks.py

Extract binary SAM3 ROI masks for Kinetics, derf, and UVG clips.

Kinetics labels are actions; SAM3 segments noun concepts, so each clip uses the
noun prompts in scripts/kinetics_prompts.yaml (default ["person"]). derf/UVG use
scripts/prompts.yaml. Per-frame instance masks are OR-ed across prompts and
written as 0/255 PNG at the native frame resolution, 1-indexed to match
frame_0001.png.

Performance options (defaults are the fast, safe path):
  --reuse-session   reuse one SAM3 session per clip (default on)
  --forward-only    propagate forward only (default on; backward is a no-op)
  --compile         torch.compile the model (slower warmup, ~12-19% faster in
                    short tests; CUDA-graph memory grows over long runs and can OOM)

Long-run safeguards:
  --restart-every N rebuild the predictor every N processed clips (bounds leaks;
                    use 0=off with --compile to avoid recompiling)
  --offload-video   keep video frames on CPU (saves GPU memory, slower)
  --max-objects N   cap tracked instances per prompt (default 16; bounds memory;
                    many-instance concepts like "brick" otherwise scale memory up)
  per-clip gc + empty_cache, one retry on any error then skip-and-log, and no
  forced expandable_segments (a known allocator-assertion trigger with compile)

Checkpoint/resume:
  Progress is saved after every clip to data/manifests/sam3_masks_progress.json.
  On restart, completed clips (and clips whose masks already exist) are skipped.
  Use --progress-file to give each parallel shard its own file.

Multi-GPU (default):
  The script detects all visible GPUs and launches one worker per GPU (a shard
  each), then shows one combined "Clip done/total" line. Per-GPU logs go to
  data/manifests/sam3_masks_gpu<id>.log. `--limit` applies to the whole run and
  is then split across GPUs. Override with --gpus 0,1; force one GPU with
  --gpu 0 or --no-auto-gpu; or shard manually with --shard-index/--num-shards.

Usage:
  python scripts/03_sam3_masks.py [--dataset kinetics|derf|uvg] [--split ...]
      [--limit N] [--compile] [--benchmark] [--list]
      [--gpus 0,1 | --gpu 0 | --no-auto-gpu]
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import config  # noqa: E402


def load_prompts(cfg: dict) -> dict:
    import yaml

    for path in (cfg["PROMPTS_YAML"], cfg["PROMPTS_SRC"]):
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
    return {}


def load_kinetics_prompts(cfg: dict) -> dict:
    import yaml

    path = cfg.get("KINETICS_PROMPTS_YAML", "")
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


def prompts_for(clip: dict, prompts_map: dict,
                kinetics_map: dict) -> list[str]:
    if clip.get("dataset") == "kinetics":
        label = clip.get("label", "")
        return kinetics_map.get(label) or ["person"]
    seq = clip.get("seq", "")
    prompts = prompts_map.get(clip["dataset"], {}).get(seq)
    if prompts:
        return prompts
    return prompts_map.get("default", ["person", "vehicle", "animal"])


def clip_key(clip: dict) -> str:
    return "{}/{}/{}".format(
        clip.get("dataset", "kinetics"),
        clip.get("split", "") or "-",
        clip.get("clip_id") or clip.get("seq", ""),
    )


def out_dir_for(clip: dict, output_root: str | None) -> str:
    if not output_root:
        return clip["masks_dir"]
    return os.path.join(output_root, clip_key(clip).replace("/", "__"))


def count_pngs(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    return sum(1 for name in os.listdir(path) if name.endswith(".png"))


def load_progress(path: str | None) -> dict:
    if path and os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            data.setdefault("done", {})
            data.setdefault("failed", {})
            return data
        except (OSError, ValueError):
            pass
    return {"done": {}, "failed": {}}


def save_progress(path: str | None, progress: dict) -> None:
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(progress, fh)
    os.replace(tmp, path)


def _accumulate(acc: dict, resp: dict) -> None:
    masks = resp.get("outputs", {}).get("out_binary_masks")
    if masks is None:
        return
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None]
    if masks.shape[0] == 0:
        return
    union = np.any(masks, axis=0)
    frame_index = resp["frame_index"]
    if frame_index in acc:
        acc[frame_index] = np.logical_or(acc[frame_index], union)
    else:
        acc[frame_index] = union


def collect_masks(predictor, frames_dir: str, prompts: list[str], *,
                  reuse_session: bool = True, forward_only: bool = True,
                  offload_video: bool = False) -> dict:
    """Union the per-prompt SAM3 masks into a frame_index -> bool mask map.

    With reuse_session, one session (and its loaded frames) serves all prompts:
    add_prompt resets tracking state per text prompt but keeps the loaded video,
    so this avoids reloading the clip for every prompt.
    """
    acc: dict[int, np.ndarray] = {}
    direction = "forward" if forward_only else "both"

    def run_prompt(session_id: str, prompt: str) -> None:
        predictor.add_prompt(session_id=session_id, frame_idx=0, text=prompt)
        for resp in predictor.propagate_in_video(
                session_id=session_id, propagation_direction=direction):
            _accumulate(acc, resp)

    if reuse_session:
        session_id = predictor.start_session(
            resource_path=frames_dir,
            offload_video_to_cpu=offload_video)["session_id"]
        try:
            for prompt in prompts:
                run_prompt(session_id, prompt)
        finally:
            predictor.close_session(session_id=session_id)
    else:
        for prompt in prompts:
            session_id = predictor.start_session(
                resource_path=frames_dir,
                offload_video_to_cpu=offload_video)["session_id"]
            try:
                run_prompt(session_id, prompt)
            finally:
                predictor.close_session(session_id=session_id)
    return acc


def write_masks(acc: dict, frames: list[str], out_dir: str) -> tuple[int, int]:
    first = cv2.imread(frames[0], cv2.IMREAD_GRAYSCALE)
    height, width = first.shape[:2]
    os.makedirs(out_dir, exist_ok=True)
    n_empty = 0
    for frame_idx in range(len(frames)):
        mask = acc.get(frame_idx)
        if mask is None:
            mask = np.zeros((height, width), dtype=bool)
            n_empty += 1
        elif mask.shape[:2] != (height, width):
            mask = cv2.resize(mask.astype(np.uint8), (width, height),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        elif not mask.any():
            n_empty += 1
        cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx + 1:04d}.png"),
                    (mask.astype(np.uint8) * 255))
    return len(frames), n_empty


def process_clip(predictor, clip: dict, prompts_map: dict, kinetics_map: dict,
                 args) -> dict:
    frames_dir = clip["frames_dir"]
    out_dir = out_dir_for(clip, args.output_root)
    frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not frames:
        return {"status": "no_frames", "frames": 0, "empty": 0,
                "generate_s": 0.0, "write_s": 0.0}
    prompts = prompts_for(clip, prompts_map, kinetics_map)
    start = time.perf_counter()
    acc = collect_masks(predictor, frames_dir, prompts,
                        reuse_session=args.reuse_session,
                        forward_only=args.forward_only,
                        offload_video=args.offload_video)
    mid = time.perf_counter()
    n_frames, n_empty = write_masks(acc, frames, out_dir)
    end = time.perf_counter()
    return {"status": "ok", "frames": n_frames, "empty": n_empty,
            "prompts": prompts, "generate_s": mid - start,
            "write_s": end - mid}


def build_predictor(gpu: int, checkpoint: str | None, compile_model: bool):
    from sam3.model_builder import build_sam3_video_predictor

    kwargs = {"gpus_to_use": [gpu]}
    if compile_model:
        kwargs["compile"] = True
    if checkpoint:
        kwargs["checkpoint_path"] = checkpoint
    return build_sam3_video_predictor(**kwargs)


def apply_object_cap(predictor, limit: int) -> None:
    """Bound tracked instances per prompt; SAM3 defaults to 10000 (unlimited).

    Many-instance concepts (e.g. "brick") would otherwise track hundreds of
    objects and exhaust GPU memory, since tracker memory scales with objects.
    """
    if limit and limit > 0:
        model = getattr(predictor, "model", None)
        if model is not None and hasattr(model, "max_num_objects"):
            model.max_num_objects = limit


def visible_gpu_ids() -> list[int]:
    """Return CUDA-visible GPU indices (respects CUDA_VISIBLE_DEVICES)."""
    try:
        import torch

        count = torch.cuda.device_count()
        if count > 0:
            return list(range(count))
    except Exception:  # noqa: BLE001
        pass
    return []


def auto_enabled(args) -> bool:
    return (args.auto_gpu and args.gpu is None and args.num_shards == 1
            and not args.list and not args.benchmark)


def resolve_gpu_ids(args) -> list[int]:
    if args.gpus:
        ids = [int(part) for part in args.gpus.split(",") if part.strip() != ""]
    else:
        ids = visible_gpu_ids()
    return sorted(set(ids))


def load_all_clips(cfg: dict, args) -> list[dict]:
    clips: list[dict] = []
    if args.dataset in (None, "kinetics"):
        clips += config.kinetics_clips(cfg, split=args.split)
    if args.dataset in (None, "derf"):
        clips += config.test_clips(cfg, "derf")
    if args.dataset in (None, "uvg"):
        clips += config.test_clips(cfg, "uvg")
    return clips


def progress_file_path(cfg: dict, args, shard_index: int,
                       num_shards: int) -> str:
    path = args.progress_file or os.path.join(
        cfg["MANIFESTS_DIR"], "sam3_masks_progress.json")
    if num_shards > 1:
        root, ext = os.path.splitext(path)
        path = f"{root}_shard{shard_index}of{num_shards}{ext}"
    return path


def _child_command(argv: list[str], gpu_id: int, shard_index: int,
                   num_shards: int) -> list[str]:
    """Rebuild this command for one GPU shard (drops GPU/shard selectors)."""
    skip_with_value = {"--gpu", "--gpus", "--shard-index", "--num-shards"}
    skip_flag = {"--auto-gpu", "--no-auto-gpu"}
    forwarded: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in skip_with_value:
            i += 2
            continue
        if arg in skip_flag:
            i += 1
            continue
        if any(arg.startswith(opt + "=") for opt in skip_with_value):
            i += 1
            continue
        forwarded.append(arg)
        i += 1
    extra = ["--gpu", str(gpu_id), "--shard-index", str(shard_index),
             "--num-shards", str(num_shards), "--no-auto-gpu"]
    return [sys.executable, str(Path(__file__).resolve()), *forwarded, *extra]


def count_done(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return len(json.load(fh).get("done", {}))
    except (OSError, ValueError):
        return 0


def run_workers(argv: list[str], args, cfg: dict,
                active: list[tuple[int, int]], sizes: list[int],
                num_shards: int) -> int:
    """Spawn one worker per GPU and show a combined progress line."""
    total = sum(sizes)
    progress_files = [progress_file_path(cfg, args, shard_index, num_shards)
                      for shard_index, _ in active]
    os.makedirs(cfg["MANIFESTS_DIR"], exist_ok=True)
    if not args.resume:
        for path in progress_files:
            try:
                os.remove(path)
            except OSError:
                pass

    print(f"[auto-gpu] {num_shards} GPUs detected; launching {len(active)} "
          f"worker(s) for {total} clips.", flush=True)
    procs = []
    logs = []
    for shard_index, gpu_id in active:
        log_path = os.path.join(cfg["MANIFESTS_DIR"],
                                f"sam3_masks_gpu{gpu_id}.log")
        log_fh = open(log_path, "w", encoding="utf-8")
        cmd = _child_command(argv, gpu_id, shard_index, num_shards)
        print(f"[auto-gpu]   GPU {gpu_id}: shard {shard_index}/{num_shards}, "
              f"{sizes[shard_index]} clips, log {log_path}", flush=True)
        procs.append((gpu_id, subprocess.Popen(cmd, stdout=log_fh,
                                               stderr=subprocess.STDOUT),
                      log_fh))
        logs.append((gpu_id, log_path))

    bar = config.Progress(total)
    try:
        while True:
            alive = [proc for _, proc, _ in procs if proc.poll() is None]
            done = sum(count_done(path) for path in progress_files)
            bar.tick(min(done, total) if total else done)
            if not alive:
                break
            time.sleep(2)
    except KeyboardInterrupt:
        for _, proc, _ in procs:
            proc.terminate()
        for _, proc, _ in procs:
            try:
                proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                proc.kill()
        bar.close()
        print("[auto-gpu] interrupted; workers stopped.", flush=True)
        return 130
    finally:
        for _, _, log_fh in procs:
            log_fh.close()

    done = sum(count_done(path) for path in progress_files)
    bar.tick(min(done, total) if total else done)
    bar.close()
    failed = sum(1 for _, proc, _ in procs if proc.returncode != 0)
    print(f"[auto-gpu] finished: {done}/{total} clips done; "
          f"{len(procs) - failed}/{len(procs)} workers ok.", flush=True)
    for gpu_id, log_path in logs:
        print(f"[auto-gpu]   GPU {gpu_id} log: {log_path}", flush=True)
    return 0 if failed == 0 else 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None,
                        help="Use this single GPU (disables auto-launch).")
    parser.add_argument("--gpus", default=None,
                        help="Comma-separated GPU ids (default: all visible).")
    parser.add_argument("--auto-gpu", dest="auto_gpu", action="store_true",
                        default=True,
                        help="Use all visible GPUs, one worker each (default).")
    parser.add_argument("--no-auto-gpu", dest="auto_gpu",
                        action="store_false")
    parser.add_argument("--dataset", choices=["kinetics", "derf", "uvg"],
                        default=None)
    parser.add_argument("--split", choices=["train", "test"], default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--reuse-session", dest="reuse_session",
                        action="store_true", default=True)
    parser.add_argument("--no-reuse-session", dest="reuse_session",
                        action="store_false")
    parser.add_argument("--forward-only", dest="forward_only",
                        action="store_true", default=True)
    parser.add_argument("--both-directions", dest="forward_only",
                        action="store_false")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--offload-video", action="store_true")
    parser.add_argument("--max-objects", type=int, default=16,
                        help="Cap tracked objects per prompt to bound memory "
                             "(0 = library default 10000)")
    parser.add_argument("--restart-every", type=int, default=0)
    parser.add_argument("--resume", dest="resume", action="store_true",
                        default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--progress-file", default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args(argv)

    # Do not force expandable_segments: it has caused CUDA allocator assertion
    # crashes ("curr_block->next") on long torch.compile runs. Set
    # PYTORCH_CUDA_ALLOC_CONF yourself if you want a specific allocator mode.

    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        parser.error("require 0 <= --shard-index < --num-shards")
    if args.gpus and args.gpu is not None:
        parser.error("use either --gpu or --gpus, not both")

    cfg = config.load_config()
    sys.path.insert(0, cfg["SAM3_SRC"])

    if auto_enabled(args):
        gpu_ids = resolve_gpu_ids(args)
        if len(gpu_ids) == 1:
            args.gpu = gpu_ids[0]
        elif len(gpu_ids) > 1:
            clips_all = load_all_clips(cfg, args)
            base = clips_all[:args.limit] if args.limit else clips_all
            num_shards = len(gpu_ids)
            sizes = [len(base[i::num_shards]) for i in range(num_shards)]
            active = [(i, gid) for i, gid in enumerate(gpu_ids)
                      if sizes[i] > 0]
            if len(active) > 1:
                return run_workers(argv, args, cfg, active, sizes, num_shards)
            args.gpu = active[0][1] if active else gpu_ids[0]

    clips: list[dict] = []
    if args.dataset in (None, "kinetics"):
        clips += config.kinetics_clips(cfg, split=args.split)
    if args.dataset in (None, "derf"):
        clips += config.test_clips(cfg, "derf")
    if args.dataset in (None, "uvg"):
        clips += config.test_clips(cfg, "uvg")
    if args.limit:
        clips = clips[:args.limit]
    if args.num_shards > 1:
        clips = clips[args.shard_index::args.num_shards]

    if args.list:
        for clip in clips:
            print(clip["frames_dir"], "->", out_dir_for(clip, args.output_root))
        return 0
    if not clips:
        print("no clips found; run 01_download_datasets.sh first", file=sys.stderr)
        return 1

    if args.benchmark:
        if not args.output_root:
            args.output_root = tempfile.mkdtemp(prefix="sam3_masks_bench_")
        args.resume = False
        print(f"[benchmark] writing to {args.output_root}", flush=True)

    if args.benchmark:
        progress_path = None  # never record benchmark runs as real progress
    else:
        progress_path = progress_file_path(cfg, args, args.shard_index,
                                           args.num_shards)
    progress = load_progress(progress_path) if args.resume \
        else {"done": {}, "failed": {}}

    gpu = args.gpu if args.gpu is not None else int(cfg["SAM3_GPU"])
    prompts_map = load_prompts(cfg)
    kinetics_map = load_kinetics_prompts(cfg)

    build_start = time.perf_counter()
    predictor = build_predictor(gpu, args.checkpoint, args.compile)
    apply_object_cap(predictor, args.max_objects)
    build_s = time.perf_counter() - build_start

    import torch

    bar = config.Progress(len(clips))
    done = skipped = failed = 0
    gen_total = write_total = 0.0
    since_build = 0
    run_start = time.perf_counter()
    peak_reserved = 0.0

    def log_memory() -> None:
        nonlocal peak_reserved
        if not torch.cuda.is_available():
            return
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak_reserved = max(peak_reserved, reserved)
        free, _ = torch.cuda.mem_get_info()
        bar.log(f"[mem] alloc={alloc:.2f}GB reserved={reserved:.2f}GB "
                f"peak_reserved={peak_reserved:.2f}GB free={free / 1e9:.1f}GB")

    def record_ok(clip: dict, result: dict) -> None:
        nonlocal done, gen_total, write_total
        done += 1
        gen_total += result["generate_s"]
        write_total += result["write_s"]
        key = clip_key(clip)
        progress["done"][key] = time.strftime("%Y-%m-%dT%H:%M:%S")
        progress["failed"].pop(key, None)
        save_progress(progress_path, progress)
        bar.log(f"masks {key}: {result['frames']} frames, {result['empty']} empty"
                f" | gen {result['generate_s']:.1f}s write {result['write_s']:.1f}s")

    def record_fail(clip: dict, reason: str) -> None:
        nonlocal failed
        failed += 1
        key = clip_key(clip)
        progress["failed"][key] = reason
        save_progress(progress_path, progress)
        bar.log(f"FAIL {key}: {reason}")

    def run_one(clip: dict) -> None:
        def attempt():
            return process_clip(predictor, clip, prompts_map, kinetics_map,
                                args)

        try:
            result = attempt()
        except Exception as exc:  # noqa: BLE001
            # CUDA allocator / transient errors are sometimes recoverable.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            bar.log(f"[retry] {clip_key(clip)}: "
                    f"{type(exc).__name__}: {exc}")
            try:
                result = attempt()
            except Exception as exc2:  # noqa: BLE001
                record_fail(clip, f"{type(exc2).__name__}: {exc2}")
                return
        if result["status"] == "ok":
            record_ok(clip, result)
        else:
            record_fail(clip, result["status"])

    try:
        for idx, clip in enumerate(clips, 1):
            bar.tick(idx)
            key = clip_key(clip)
            out_dir = out_dir_for(clip, args.output_root)
            frames = sorted(glob.glob(
                os.path.join(clip["frames_dir"], "frame_*.png")))
            n_expected = len(frames)

            if args.resume and key in progress["done"]:
                if n_expected == 0 or count_pngs(out_dir) >= n_expected:
                    skipped += 1
                    continue

            if not args.benchmark and n_expected and \
                    count_pngs(out_dir) >= n_expected:
                progress["done"].setdefault(key, "existing")
                save_progress(progress_path, progress)
                skipped += 1
                continue

            if n_expected == 0:
                record_fail(clip, "no frames")
                continue

            run_one(clip)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_memory()

            since_build += 1
            if args.restart_every and since_build >= args.restart_every:
                try:
                    predictor.shutdown()
                except Exception:  # noqa: BLE001
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                predictor = build_predictor(gpu, args.checkpoint, args.compile)
                apply_object_cap(predictor, args.max_objects)
                since_build = 0
                bar.log(f"[maintenance] rebuilt predictor after "
                        f"{args.restart_every} clips")
    finally:
        try:
            predictor.shutdown()
        except Exception:  # noqa: BLE001
            pass
        bar.close()

    run_s = time.perf_counter() - run_start
    print(f"done={done} skipped={skipped} failed={failed} "
          f"build={build_s:.1f}s gen={gen_total:.1f}s write={write_total:.1f}s "
          f"wall={run_s:.1f}s peak_reserved={peak_reserved:.2f}GB", flush=True)
    if done:
        print(f"avg generate per processed clip: {gen_total / done:.1f}s",
              flush=True)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
