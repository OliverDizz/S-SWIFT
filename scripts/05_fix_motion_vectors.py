#!/usr/bin/env python3
"""scripts/05_fix_motion_vectors.py

Re-extract motion vectors for clips whose ``<split>_mv`` directory is missing
any of the expected flow files. Only incomplete clips are touched: complete
clips are left as-is, and clips with missing or gappy source frames are
reported (they need ``01_download_datasets.sh``) but not re-extracted.

The expected set of flow files is derived from ``flows/get_mv.sh`` itself: for
every I-frame ``idx`` (``idx % 12 == 1``) whose ``idx + 12`` frame exists, the
11 following frames each get four files
``frame_XXXX_{before,after}_flow_{x,y}_0001.jpg``. The script recomputes this
per clip from the frames actually on disk, so it works for Kinetics (100
frames) and for derf/UVG (native frame counts) without hardcoding counts.

Completeness is checked with a single directory listing per clip (set
difference), not per-file stat calls, so it stays fast on NFS.

Failures are retried and appended to ``data/manifests/mv_fix_failures.log`` so
a gap-fill run leaves an audit trail, unlike ``02_motion_vectors.sh``.

Note: re-running ``get_mv.sh`` cannot reproduce every file. The bundled
ffmpeg 2.8.6 + ``MotionVector`` pipeline deterministically drops long-distance
motion vectors (mostly the 6-frame and 3-frame references) for clips with large
motion, so those files stay missing and are reported in the failure log. Use
``--dry-run`` first to audit the gaps; the codec zero-fills missing motion.

Usage:
  python3 scripts/05_fix_motion_vectors.py --dry-run
  python3 scripts/05_fix_motion_vectors.py --dataset kinetics --limit 100
  python3 scripts/05_fix_motion_vectors.py --dataset kinetics
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import config  # noqa: E402

FLOW_SUFFIXES = ("before_flow_x", "before_flow_y",
                 "after_flow_x", "after_flow_y")
FLOW_EXT = ".jpg"

# get_mv.sh GOP deltas, keyed by i in 1..11: (prev_delta, next_delta) where
# prev_idx = cur_idx + prev_delta and next_idx = cur_idx + next_delta.
_DELTAS = {
    1: (-1, 2), 4: (-1, 2), 7: (-1, 2), 10: (-1, 2),
    2: (-2, 1), 5: (-2, 1), 8: (-2, 1), 11: (-2, 1),
    3: (-3, 3), 9: (-3, 3),
    6: (-6, 6),
}


def _list_frames(frames_dir):
    """Return the set of frame indices present in ``frames_dir`` (1-indexed)."""
    out = set()
    if not os.path.isdir(frames_dir):
        return out
    prefix = "frame_"
    for name in os.listdir(frames_dir):
        stem = os.path.splitext(name)[0]
        if not stem.startswith(prefix):
            continue
        try:
            out.add(int(stem[len(prefix):]))
        except ValueError:
            continue
    return out


def _expected_flow_names(frames):
    """Return the flow filenames a successful get_mv.sh run would produce."""
    expected = set()
    for idx in sorted(frames):
        if idx % 12 != 1:
            continue
        if idx + 12 not in frames:
            continue
        for i in range(1, 12):
            cur = idx + i
            prev = cur + _DELTAS[i][0]
            if prev not in frames or cur not in frames:
                continue
            for suffix in FLOW_SUFFIXES:
                expected.add("frame_%04d_%s_0001%s" % (cur, suffix, FLOW_EXT))
    return expected


def _existing_mv_files(mv_dir):
    if not os.path.isdir(mv_dir):
        return set()
    return set(os.listdir(mv_dir))


def clip_status(row):
    """Classify one clip: 'complete', 'no_frames', 'source_gap', or 'incomplete'."""
    frames = _list_frames(row["frames_dir"])
    if not frames:
        return ("no_frames", 0)

    if len(frames) != max(frames) - min(frames) + 1:
        # Gappy source frames: get_mv.sh (set -e) can crash on a missing
        # reference frame, so this needs 01_download_datasets.sh, not MV.
        return ("source_gap", 0)

    expected = _expected_flow_names(frames)
    existing = _existing_mv_files(row["mv_dir"])
    missing = len(expected - existing)
    if missing:
        return ("incomplete", missing)
    return ("complete", 0)


def _clip_label(row):
    return "%s/%s/%s" % (
        row["dataset"], row.get("split") or "-",
        row.get("clip_id") or row.get("seq", ""))


def _probe_dims(ffprobe, frames_dir, frames):
    """Return (width, height) of the first frame via ffprobe."""
    first = "frame_%04d.png" % min(frames)
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0",
         os.path.join(frames_dir, first)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("ffprobe failed for %s: %s"
                           % (first, proc.stderr.strip()))
    w, h = proc.stdout.strip().split("x")
    return int(w), int(h)


def _run_get_mv(cfg, row):
    """Run flows/get_mv.sh for one clip. Raises on non-zero exit."""
    frames = _list_frames(row["frames_dir"])
    w, h = _probe_dims(cfg["SYSTEM_FFPROBE"], row["frames_dir"], frames)

    env = os.environ.copy()
    mv_bin = os.path.join(cfg["MV_ENV_DIR"], "bin")
    env["PATH"] = mv_bin + os.pathsep + env.get("PATH", "")
    env["EXTRACT_MVS"] = cfg["EXTRACT_MVS_BIN"]
    env["MOTIONVECTOR"] = cfg["MOTIONVECTOR_BIN"]

    proc = subprocess.run(
        ["bash", cfg["GET_MV_SH"], row["frames_dir"], row["mv_dir"],
         str(w), str(h), "png"],
        env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        reason = tail[-1] if tail else "unknown error"
        raise RuntimeError("get_mv.sh exited %d: %s" % (proc.returncode, reason))


def fix_clip(cfg, row, retries, log_path):
    """Fix one clip, retrying. Returns (status, detail)."""
    label = _clip_label(row)
    for attempt in range(1, retries + 1):
        try:
            _run_get_mv(cfg, row)
        except RuntimeError as exc:
            detail = str(exc)
            if attempt == retries:
                _append_failure(log_path, label, detail)
                return ("failed", detail)
            continue

        status, missing = clip_status(row)
        if status == "complete":
            return ("fixed", "")
        if attempt == retries:
            detail = "still missing %d flow files after %d attempts" % (
                missing, retries)
            _append_failure(log_path, label, detail)
            return ("failed", detail)

    return ("failed", "unknown")


def _append_failure(log_path, label, detail):
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write("%s  %s  %s\n" % (
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            label, detail))


def _check_tools(cfg):
    missing = []
    for key in ("EXTRACT_MVS_BIN", "MOTIONVECTOR_BIN", "GET_MV_SH",
                "SYSTEM_FFPROBE"):
        path = cfg.get(key)
        if not path or not os.path.isfile(path):
            missing.append(key)
    if missing:
        print("Missing required tool(s): %s" % ", ".join(missing))
        print("Build the motion-vector tools first: "
              "bash scripts/02_motion_vectors.sh --smoke")
        return False
    if not os.access(cfg["EXTRACT_MVS_BIN"], os.X_OK) or \
            not os.access(cfg["MOTIONVECTOR_BIN"], os.X_OK):
        print("Motion-vector binaries are not executable; rebuild with "
              "bash scripts/02_motion_vectors.sh --rebuild")
        return False
    return True


def main(argv):
    parser = argparse.ArgumentParser(
        description="Re-extract missing motion vectors for incomplete clips.")
    parser.add_argument("--dataset", choices=["kinetics", "derf", "uvg"],
                        default="kinetics")
    parser.add_argument("--split", choices=["train", "test"], default=None)
    parser.add_argument("--only", default=None,
                        help="operate on a single clip_id (or derf/uvg seq)")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many clips (0 = all)")
    parser.add_argument("--jobs", type=int, default=0,
                        help="parallel workers (default: PARALLELISM from config)")
    parser.add_argument("--retries", type=int, default=2,
                        help="attempts per clip (default 2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="write the incomplete clip list to a file without "
                             "re-extracting")
    args = parser.parse_args(argv)

    cfg = config.load_config()
    if not _check_tools(cfg):
        return 2

    if args.dataset == "kinetics":
        clips = config.kinetics_clips(cfg, split=args.split)
    else:
        clips = config.test_clips(cfg, args.dataset)

    if args.only:
        clips = [row for row in clips
                 if (row.get("clip_id") or row.get("seq", "")) == args.only]
        if not clips:
            print("Clip not found in manifest: %s" % args.only)
            return 2

    if args.limit:
        clips = clips[: args.limit]

    jobs = args.jobs or int(cfg.get("PARALLELISM") or 1)
    log_path = os.path.join(cfg["MANIFESTS_DIR"], "mv_fix_failures.log")
    list_path = os.path.join(cfg["MANIFESTS_DIR"], "mv_incomplete.txt")

    counts = {"complete": 0, "no_frames": 0, "source_gap": 0,
              "incomplete": 0, "fixed": 0, "failed": 0}
    incomplete = []
    no_frames = []
    source_gaps = []

    for row in clips:
        status, missing = clip_status(row)
        if status == "complete":
            counts["complete"] += 1
        elif status == "no_frames":
            counts["no_frames"] += 1
            no_frames.append(_clip_label(row))
        elif status == "source_gap":
            counts["source_gap"] += 1
            source_gaps.append(_clip_label(row))
        else:
            counts["incomplete"] += 1
            incomplete.append((row, missing))

    print("Checked %d clips: %d complete, %d incomplete, %d no-frames, "
          "%d source-gaps" % (
              len(clips), counts["complete"], counts["incomplete"],
              counts["no_frames"], counts["source_gap"]))

    if no_frames:
        print("Clips with no source frames (needs 01_download_datasets.sh): %s"
              % ", ".join(no_frames))
    if source_gaps:
        print("Clips with gappy source frames (needs 01_download_datasets.sh): %s"
              % ", ".join(source_gaps))

    if args.dry_run or not incomplete:
        if incomplete:
            with open(list_path, "w", encoding="utf-8") as fh:
                for row, missing in incomplete:
                    fh.write("%d\t%s\n" % (missing, _clip_label(row)))
            print("Incomplete clip list written to %s" % list_path)
        return 0

    print("Re-extracting %d incomplete clip(s) with %d worker(s)..."
          % (len(incomplete), jobs))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(fix_clip, cfg, row, args.retries, log_path): row
            for row, _ in incomplete
        }
        for fut in as_completed(futures):
            status, _ = fut.result()
            counts[status] += 1

    print("Done: %d fixed, %d failed" % (counts["fixed"], counts["failed"]))
    if counts["failed"]:
        print("Failures logged to %s" % log_path)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
