"""Shared helper for the Swift dataset pipeline scripts.

Loads scripts/config.env, expands ``${KEY}`` references, and enumerates the
clips described by the manifests that ``01_download_datasets.sh`` writes.

Only the Python standard library is used, so the same module runs under the
system interpreter and the sam3 conda interpreter alike.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

CONFIG_ENV = Path(__file__).resolve().parent / "config.env"

# Manifest schemas.
KINETICS_HEADER = ["split", "clip_id", "youtube_id", "label", "time_start",
                   "time_end", "n_frames"]
TEST_HEADER = ["seq", "filename", "width", "height", "n_frames"]

_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Conda environment names the pipeline depends on.
CONDA_ENV_NAMES = ("sam3", "mv_env")


def _conda_info_base() -> str | None:
    """Return the conda base prefix, or None if conda is unavailable."""
    exe = shutil.which("conda")
    if not exe:
        return None
    try:
        proc = subprocess.run([exe, "info", "--base"], capture_output=True,
                              text=True, timeout=15)
        base = proc.stdout.strip()
        return base or None
    except (OSError, subprocess.SubprocessError):
        return None


def resolve_conda_envs_dir() -> str:
    """Resolve the directory that holds the sam3 and mv_env environments.

    Preference order:
      1. $CONDA_ENVS_DIR (explicit override)
      2. ~/conda_envs, ~/.conda/envs, <conda base>/envs (first that exists and
         contains sam3 or mv_env)
      3. the first existing directory from that list
      4. ~/.conda/envs
    """
    explicit = os.environ.get("CONDA_ENVS_DIR")
    if explicit:
        return explicit

    home = os.path.expanduser("~")
    base = _conda_info_base()
    candidates = [os.path.join(home, "conda_envs"),
                  os.path.join(home, ".conda", "envs")]
    if base:
        candidates.append(os.path.join(base, "envs"))

    # Prefer a directory that actually holds one of the target environments.
    for directory in candidates:
        if any(os.path.isdir(os.path.join(directory, name))
               for name in CONDA_ENV_NAMES):
            return directory
    # Otherwise the first existing directory.
    for directory in candidates:
        if os.path.isdir(directory):
            return directory
    return os.path.join(home, ".conda", "envs")


def load_config(path: Path = CONFIG_ENV) -> dict:
    """Return the config as a dict of strings with all references expanded."""
    root = Path(__file__).resolve().parent.parent
    raw = {"REPO_ROOT": str(root)}

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            raw[key] = value

    # Resolve the conda environment directory before expanding ${CONDA_ENVS_DIR}.
    raw["CONDA_ENVS_DIR"] = (
        raw.get("CONDA_ENVS_DIR") or os.environ.get("CONDA_ENVS_DIR")
        or resolve_conda_envs_dir()
    )

    # Resolve ${KEY} references (multi-pass to handle chained references).
    for _ in range(16):
        changed = False
        for key, value in list(raw.items()):
            new_value = _REF.sub(
                lambda m: raw.get(m.group(1), m.group(0)), value)
            if new_value != value:
                raw[key] = new_value
                changed = True
        if not changed:
            break

    return raw


def read_csv(path: str | os.PathLike) -> list[dict]:
    """Read a CSV file into a list of dicts keyed by header names."""
    with open(path, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: str | os.PathLike, header: list[str],
              rows: list[dict]) -> None:
    """Write rows to a CSV file, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class Progress:
    """A minimal "Clip i/Total" terminal progress line.

    Writes a carriage-return line to stderr so it updates in place. It is silent
    when stderr is not a TTY (so redirected logs stay clean), and it prints only
    "Clip i" when the total is unknown (0 or None).
    """

    def __init__(self, total: int | None = None, stream=None):
        self.total = int(total or 0)
        self.n = 0
        self.stream = stream if stream is not None else sys.stderr
        self.drawn = False

    def _is_tty(self) -> bool:
        try:
            return bool(self.stream.isatty())
        except (AttributeError, OSError, ValueError):
            return False

    def _draw(self) -> None:
        if not self._is_tty():
            return
        label = (f"Clip {self.n}/{self.total}" if self.total > 0
                 else f"Clip {self.n}")
        try:
            self.stream.write("\r" + label)
            self.stream.flush()
        except (OSError, ValueError):
            return
        self.drawn = True

    def tick(self, n: int | None = None) -> None:
        """Advance the counter (to n, or by one) and redraw."""
        self.n = int(n) if n is not None else self.n + 1
        self._draw()

    def log(self, message: str) -> None:
        """Print a normal line without corrupting the progress line."""
        if self.drawn:
            try:
                self.stream.write("\n")
                self.stream.flush()
            except (OSError, ValueError):
                pass
            self.drawn = False
        print(message, flush=True)
        self._draw()

    def close(self) -> None:
        if self.drawn:
            try:
                self.stream.write("\n")
                self.stream.flush()
            except (OSError, ValueError):
                pass
            self.drawn = False


def kinetics_clips(cfg: dict, split: str | None = None) -> list[dict]:
    """Return Kinetics clips from the split manifest.

    Each entry adds the derived frame/mv/mask/edge directories.
    """
    kin_dir = cfg["KINETICS_DIR"]
    if not os.path.isfile(cfg["KINETICS_MANIFEST"]):
        return []
    rows = read_csv(cfg["KINETICS_MANIFEST"])
    clips = []
    for row in rows:
        if split is not None and row["split"] != split:
            continue
        clip_id = row["clip_id"]
        row["dataset"] = "kinetics"
        row["frames_dir"] = os.path.join(kin_dir, row["split"], clip_id)
        row["mv_dir"] = os.path.join(kin_dir, row["split"] + "_mv", clip_id)
        row["masks_dir"] = os.path.join(kin_dir, row["split"] + "_masks", clip_id)
        row["edges_dir"] = os.path.join(kin_dir, row["split"] + "_edges", clip_id)
        clips.append(row)
    return clips


def test_clips(cfg: dict, dataset: str) -> list[dict]:
    """Return derf or UVG clips from their manifest.

    Each entry adds the derived frame/mv/mask/edge directories.
    """
    if dataset not in ("derf", "uvg"):
        raise ValueError(f"unknown test dataset: {dataset!r}")
    base = cfg["DERF_DIR"] if dataset == "derf" else cfg["UVG_DIR"]
    manifest = (cfg["DERF_MANIFEST"] if dataset == "derf"
                else cfg["UVG_MANIFEST"])
    if not os.path.isfile(manifest):
        return []
    clips = []
    for row in read_csv(manifest):
        seq = row["seq"]
        row["dataset"] = dataset
        row["frames_dir"] = os.path.join(base, "frames", seq)
        row["mv_dir"] = os.path.join(base, "mv", seq)
        row["masks_dir"] = os.path.join(base, "masks", seq)
        row["edges_dir"] = os.path.join(base, "edges", seq)
        clips.append(row)
    return clips


def _print_clip(row: dict) -> None:
    """Emit one clip as a tab-separated record for shell consumers."""
    print("\t".join([
        row.get("dataset", "kinetics"),
        row.get("split", "") or "-",
        row.get("clip_id") or row.get("seq", ""),
        row.get("frames_dir", ""),
        row.get("mv_dir", ""),
        row.get("masks_dir", ""),
        row.get("edges_dir", ""),
    ]))


def _main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Enumerate dataset clips for the Swift pipeline.")
    parser.add_argument("--dataset", choices=["kinetics", "derf", "uvg"],
                        default="kinetics")
    parser.add_argument("--split", choices=["train", "test"], default=None)
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many clips (0 = all)")
    args = parser.parse_args(argv)

    cfg = load_config()
    if args.dataset == "kinetics":
        clips = kinetics_clips(cfg, split=args.split)
    else:
        clips = test_clips(cfg, args.dataset)

    for i, row in enumerate(clips):
        if args.limit and i >= args.limit:
            break
        _print_clip(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
