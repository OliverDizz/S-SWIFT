# Dataset pipeline (`scripts/`)

These scripts download and prepare the data for the Swift codec:

- **Kinetics-400**: 8,000 train + 2,000 held-out clips, 100 frames/clip at 352x288.
- **derf** (Xiph) and **UVG** test sequences at native resolution.
- Per frame: motion vectors, binary SAM3 ROI masks, and Canny edge maps.

Run the steps in order from the repository root.

Each script prints a `Clip i/total` progress line to the terminal while it
processes clips. It updates in place; it prints only `Clip i` if the total is not
known, and it is silent when output is redirected (non-TTY).

## Environments

| Script | Environment |
|---|---|
| `01_download_datasets.sh` | system `curl`/`wget`/`tar`/`ffmpeg`/`ffprobe` + standard-library `python3` |
| `02_motion_vectors.sh` | `mv_env` (FFmpeg 2.8.6 + OpenCV 2.4.11 + gcc) |
| `03_sam3_masks.py` | `sam3` (PyTorch + SAM3 + OpenCV) |
| `04_canny_edges.py` | `sam3` (OpenCV) |

Create the environments with the files in `../environments/`.

## Scripts

### `01_download_datasets.sh`

Kinetics-400 comes from the official CVDF S3 mirror
(`https://s3.amazonaws.com/kinetics/400/`), not from YouTube. The mirror serves
`part_N.tar.gz` archives of ~1,000 clips each (~1.5 GB). The script:

1. downloads `train.csv` / `val.csv` annotations and the archive path lists;
2. downloads the first `K400_TRAIN_PARTS` train and `K400_VAL_PARTS` val archives
   with resumable `wget -c`, then extracts them with `tar`;
3. decodes 100 frames at 352x288 for the clips in the committed manifest
   `scripts/manifests/kinetics_split.csv`, then **cross-validates**: every
   manifest clip is present, its label matches the annotation CSV, and it has
   the expected 100 frames.

Because the split is committed, a fresh clone reproduces the exact same
8,000/2,000 clips. The script reports missing clips (raise `K400_TRAIN_PARTS` /
`K400_VAL_PARTS`) and any label or frame-count mismatches instead of silently
resampling.

To generate a different split, pass `--resample`; it samples with `SPLIT_SEED`
and rewrites `scripts/manifests/kinetics_split.csv` (commit the result).

It also downloads the 10 derf Y4M sequences and the 7 UVG FullHD YUV sequences,
decodes them at native resolution, and validates frame counts/dimensions against
`scripts/manifests/{derf,uvg}_sequences.csv`.

```bash
bash scripts/01_download_datasets.sh
bash scripts/01_download_datasets.sh --skip-uvg        # partial run
bash scripts/01_download_datasets.sh --dry-run         # no downloads
bash scripts/01_download_datasets.sh --resample        # new split
```

### `02_motion_vectors.sh`

Builds `MotionVector` and `extract_mvs` in `mv_env` if missing (or with
`--rebuild`), then extracts four motion-vector images per target frame. Each
frame resolution is read with `ffprobe` and passed to `flows/get_mv.sh`.

```bash
bash scripts/02_motion_vectors.sh --smoke              # one clip
bash scripts/02_motion_vectors.sh --dataset kinetics
```

### `03_sam3_masks.py`

Writes binary 0/255 SAM3 ROI masks, one per frame, 1-indexed to match the frame
files. Requires `hf auth login` and accepted access to `facebook/sam3`.

SAM3 segments noun concepts (e.g. "person", "guitar"), not actions. Kinetics
labels are actions such as "playing guitar", so the script prompts `person` (the
actor) by default. Add per-class noun prompts in `scripts/kinetics_prompts.yaml`
to also target the interacting object. derf/UVG use the noun prompts in
`scripts/prompts.yaml`.

#### Fast path (defaults)

The defaults are the fast, safe path: one SAM3 session per clip reused across
all prompts (`--reuse-session`) and forward-only propagation (`--forward-only`;
the backward pass is a no-op for SAM3). Measured on 100-frame Kinetics clips:

| Config | 3-prompt clip | 2-prompt clip | Notes |
|---|---:|---:|---|
| baseline (session per prompt, both directions) | 73.2 s | 46.8 s | old behavior |
| reuse + forward (default) | 66.0 s | 42.7 s | ~10% faster |
| reuse + forward + `--compile` | 80.9 s* | 37.7 s | ~19% faster steady state |

\* first clip includes torch.compile warmup (~15 s); later clips are faster.
Without `--compile`, GPU memory was flat across clips: 5.03 GB allocated /
5.16 GB reserved and ~15.6 GB free on a 20 GB card. With `--compile`, CUDA-graph
and private-pool memory grows over clips and can reach OOM, so compile is not
recommended for full runs.

#### Safe overnight run

Progress is checkpointed after every clip, failures are logged and skipped, and
a rerun resumes automatically. Each run first checks `--limit` to pilot.

By default the script uses **all visible GPUs**: it launches one worker per GPU
(a shard each) and shows one combined `Clip done/total` line. Per-GPU logs go to
`data/manifests/sam3_masks_gpu<id>.log`.

```bash
conda activate sam3
python scripts/03_sam3_masks.py --dataset kinetics --limit 100      # pilot
python scripts/03_sam3_masks.py --dataset kinetics --restart-every 1000
```

Do not use `--compile` for long runs: torch.compile's CUDA-graph/private-pool
memory grows over clips and has caused OOM and allocator assertion crashes
(`curr_block->next`). It is ~12–19% faster in short tests but not reliable at
scale.

Overrides: `--gpus 0,1` to choose GPUs, `--gpu 0` or `--no-auto-gpu` to force a
single process, and `--shard-index`/`--num-shards` for manual sharding. `--limit`
applies to the whole run and is then split across GPUs.

If a run is interrupted (crash, reboot, Ctrl-C), rerun the same command; completed
clips are skipped and only missing/partial clips are regenerated. For the larger
1080p sequences (UVG up to 600 frames) add `--offload-video` if you hit GPU OOM.

#### Flags

- `--auto-gpu` / `--no-auto-gpu` (default: use all visible GPUs)
- `--gpus 0,1` (choose GPUs) or `--gpu 0` (single GPU, no auto-launch)
- `--reuse-session` / `--no-reuse-session` (default reuse)
- `--forward-only` / `--both-directions` (default forward)
- `--compile` (opt-in; ~12–19% faster in short tests, but CUDA-graph memory grows
  over long runs and can OOM — avoid for full runs)
- `--restart-every N` (rebuild the predictor every N clips to bound any leak)
- `--offload-video` (keep video frames on CPU; lower GPU memory, slower)
- `--max-objects N` (cap tracked instances per prompt; default 16, 0 = library default 10000)
- `--resume` / `--no-resume`, `--progress-file PATH`
- `--shard-index I --num-shards N` (manual shards over disjoint clips)
- `--output-root DIR` (write elsewhere, e.g. a pilot)
- `--benchmark` (write to a temp dir and print per-clip timing; single process)

Checkpoint files default to `data/manifests/sam3_masks_progress.json` (one
per-shard suffix `_shardIofN.json`). Delete them plus the mask dirs to force a
full regeneration.

Safeguards active in every run: `gc.collect()` + `torch.cuda.empty_cache()` per
clip, one retry on any error then skip-and-log, and optional periodic predictor
rebuild. It caps tracked objects per prompt (`--max-objects`, default 16) because
many-instance concepts (e.g. "brick") scale tracker memory with the object count.
The script does not force `expandable_segments`, which has caused CUDA allocator
assertion crashes (`curr_block->next`) with `torch.compile` on long runs; set
`PYTORCH_CUDA_ALLOC_CONF` yourself if you want a specific mode.


### `04_canny_edges.py`

Writes binary `cv2.Canny(50, 150)` edge maps at native resolution.

```bash
conda activate sam3
python scripts/04_canny_edges.py
```

### `05_fix_motion_vectors.py`

Audits the motion-vector output and re-extracts only the clips that are missing
expected flow files (instead of redoing every clip like `02_motion_vectors.sh`).
It recomputes the expected set from `flows/get_mv.sh`, so it works for Kinetics
and derf/UVG alike, and it appends failures to `data/manifests/mv_fix_failures.log`.

```bash
python3 scripts/05_fix_motion_vectors.py --dry-run               # audit only
python3 scripts/05_fix_motion_vectors.py --dataset kinetics --limit 100   # pilot
python3 scripts/05_fix_motion_vectors.py --dataset kinetics      # re-extract gaps
```

Known limitation: the bundled ffmpeg 2.8.6 + `MotionVector` pipeline
deterministically drops long-distance motion vectors (mainly the 6-frame and
3-frame references) on clips with large motion. Those files cannot be
reproduced by re-running the extraction and stay listed in the failure log.
The codec zero-fills missing motion, so the practical impact is small.

## Configuration

`config.env` holds the shared settings. Both the shell scripts and
`config.py` read it. The conda environment directory is auto-detected:
`$CONDA_ENVS_DIR`, then `~/conda_envs`, `~/.conda/envs`, or
`<conda base>/envs`. Set `CONDA_ENVS_DIR` only for an unusual location.

The authoritative, version-controlled metadata lives in `scripts/manifests/`
(`kinetics_split.csv`, `derf_sequences.csv`, `uvg_sequences.csv`). A fresh clone
uses these to reproduce and validate the same data. Runtime downloads
(`train.csv`, `val.csv`, path lists) and `download.log` go to `data/manifests/`.

Prompts live in `scripts/prompts.yaml` (derf/UVG noun prompts) and
`scripts/kinetics_prompts.yaml` (Kinetics action-label overrides).

## Output layout

```
data/
  raw/                                  # transient downloads
    kinetics/targz/<source>/part_N.tar.gz
    kinetics/{train,test}/<clip>.mp4
    derf/<seq>.y4m
    uvg/<seq>_1920x1080_8bit.yuv
  kinetics/
    train|test/<clip>/frame_%04d.png
    train|test_mv/<clip>/frame_XXXX_{before,after}_flow_{x,y}_0001.jpg
    train|test_masks/<clip>/frame_%04d.png
    train|test_edges/<clip>/frame_%04d.png
  derf/{frames,mv,masks,edges}/<seq>/
  uvg/{frames,mv,masks,edges}/<seq>/
  manifests/                            # runtime only (gitignored)
    train.csv  val.csv  k400_*_path.txt  download.log
```

Authoritative metadata (tracked): `scripts/manifests/kinetics_split.csv`,
`scripts/manifests/derf_sequences.csv`, `scripts/manifests/uvg_sequences.csv`,
`scripts/prompts.yaml`, `scripts/kinetics_prompts.yaml`.
