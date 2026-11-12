# AGENTS.md

## Project identity

This repository is the research artifact for **Swift: Adaptive Video Streaming with
Layered Neural Codecs** (NSDI 2022). It is not Apple's Swift language.

The repository contains three code development lines:

1. An offline layered neural video codec. The codec uses PyTorch and lives in `codec/src/`.
2. An online adaptive-bitrate (ABR) streaming simulator. The simulator uses a
   reinforcement-learning agent and TensorFlow 1.x APIs. It lives in `streamer/`.
3. A semantic codec under development. It extends the Swift codec with segmentation
   masks and edge maps for a research paper. It lives in `codec/semantic/`.

The codec exposes several quality points for each video chunk. The streamer can then
use spare bandwidth to upgrade chunks that are already in the playback buffer. The
streamer does not only choose the quality of the next chunk.

The streamer implements this behavior with three parts:

- the `regret_window`,
- upgrade actions,
- "fill" actions.

The paper is `assets/nsdi-2022-paper.pdf`. The codec architecture diagram is
`assets/codec.png`. The root `README.md` is brief and somewhat stale. Read the code
when the README and the code disagree.

## Repository map

| Path | Purpose |
| --- | --- |
| `flows/` | Extracts motion vectors. It contains bundled FFmpeg 2.7.2 sources and OpenCV code; `scripts/02_motion_vectors.sh` builds `extract_mvs` against `mv_env` FFmpeg 2.8.6. It writes `_before_flow_x/y` and `_after_flow_x/y` images for the codec. |
| `codec/src/` | Main hierarchical P-frame codec. It contains DDP training, evaluation, TensorBoard logging, the encoder, the decoder, the early-exit decoder, and progressive binary codes. |
| `codec/src/training_experiments.ipynb` | Standard notebook for the three base-codec hierarchy trainings. It launches DDP training and records TensorBoard events. |
| `codec/semantic/` | Semantic Swift codec. It is an implemented copy of `codec/src/` plus SAM3 mask and Canny edge side-information. Its standard training flow is `codec/semantic/semantic_experiments.ipynb`. |
| `codec/icodec/` | Separate progressive I-frame codec. It derives from `pytorch-image-comp-rnn`. |
| `codec/laplacian/`, `codec/singleshot/` | Experimental codec variants and ablation codecs. They are not part of the main path. |
| `data/` | Codec training and evaluation data. The `scripts/` pipeline writes `data/kinetics/`, `data/derf/`, and `data/uvg/`. |
| `scripts/` | Dataset download and preparation pipeline. It downloads Kinetics-400 from the CVDF mirror, derf, and UVG. It extracts motion vectors, SAM3 masks, and Canny edges. It also audits and repairs incomplete motion-vector output. |
| `environments/` | Conda and pip environment files for `semswift_env`, `sam3`, and `mv_env`. |
| `streamer/` | Grad-style ABR simulator, actor-critic agent, mahimahi traces, example model, and chunk-size tables. |
| `baselines/svc/`, `baselines/shvc/` | MPEG reference software for H.264 SVC and H.265 SHVC comparisons. The Python code does not call this software. |
| `assets/` | Paper and architecture figure. |
| `external/sam3/` | SAM 3 library (submodule). The scripts use it to extract segmentation masks for the semantic codec. Do not modify or vendor it. |

## End-to-end data flow

```text
video frames
  -> flows/ extracts motion vectors
  -> codec/src trains and encodes hierarchical inter-frames
  -> codec/icodec encodes I-frames
  -> progressive codes and reconstructed frames
  -> five quality levels and chunk-size tables
  -> streamer/simulator.py simulates download, buffer, and rebuffering
  -> streamer/net.py actor-critic chooses the next quality and upgrades
  -> logs record quality, rebuffer time, and reward
```

The semantic codec adds this side-information path:

```text
video frames
  -> scripts/03_sam3_masks.py uses external/sam3 to extract binary object masks S
  -> scripts/04_canny_edges.py extracts binary Canny edge maps E
  -> codec/semantic trains the visual layer with S and E
  -> S and E are stored losslessly; the visual layer stays progressive
```

## Dataset pipeline (`scripts/`)

The `scripts/` pipeline downloads and prepares the codec data. Run the steps in order.

1. `01_download_datasets.sh` downloads the raw data and decodes frames.
   - Kinetics-400 comes from the CVDF S3 mirror (`https://s3.amazonaws.com/kinetics/400/`). The mirror serves `part_N.tar.gz` archives. Do not use YouTube or yt-dlp.
   - The committed manifest `scripts/manifests/kinetics_split.csv` fixes the exact 8,000 train and 2,000 test clips. A fresh clone reproduces the same split.
   - The script downloads `K400_TRAIN_PARTS` train archives and `K400_VAL_PARTS` val archives. It decodes the manifest clips and cross-validates each clip against `train.csv`/`val.csv` and the expected 100 frames.
   - It reports missing clips and label or frame-count mismatches. Raise the part counts when clips are missing. Use `--resample` to create a new split.
   - It writes 100 frames per clip at 352x288. It writes the val split as the `test` split.
   - derf and UVG frames use native resolution. The script validates them against `scripts/manifests/{derf,uvg}_sequences.csv`.
2. `02_motion_vectors.sh` builds the motion-vector tools in `mv_env`. It then extracts motion vectors for each frame.
3. `03_sam3_masks.py` writes binary SAM3 ROI masks in the `sam3` env. SAM3 segments noun concepts, not actions. Kinetics labels are actions, so the script prompts `person` by default. `scripts/kinetics_prompts.yaml` overrides this per class. It needs Hugging Face access to `facebook/sam3`. It reuses one session per clip and propagates forward by default. It uses all visible GPUs by default and launches one worker per GPU. It checkpoints progress after each clip and resumes on the next run. For long runs, use `--restart-every` and `--offload-video`. `--max-objects` caps tracked instances per prompt (default 16) so many-instance concepts cannot exhaust memory. Do not use `--compile` for long runs: its CUDA-graph memory grows and can OOM. Use `--gpus`, `--gpu`, or `--shard-index`/`--num-shards` to override GPU selection. For Kinetics it writes `data/kinetics/{train,test}_masks/<clip>/`.
4. `04_canny_edges.py` writes binary Canny edge maps in the `sam3` env. For Kinetics it writes `data/kinetics/{train,test}_edges/<clip>/`.
5. `05_fix_motion_vectors.py` audits the motion-vector directories and re-extracts only incomplete clips. It runs with system `python3` and uses the `mv_env` binaries. Use `--dry-run` first. It writes an incomplete list to `data/manifests/mv_incomplete.txt` and appends failures to `data/manifests/mv_fix_failures.log`.

`scripts/config.env` holds the shared settings. `scripts/config.py` reads it. The scripts find the conda environments automatically from `$CONDA_ENVS_DIR`, `~/conda_envs`, `~/.conda/envs`, or `<conda base>/envs`.

Use the environment files in `environments/` to create the three conda environments.

## Codec details (`codec/`)

### Temporal hierarchy

The codec uses 12-frame GOPs. `flows/get_mv.sh` documents the intended pattern:

- I-frames occur at frame indices 1 and 13.
- Frame 7 uses contexts at distance 6/6.
- Frames 4 and 10 use contexts at distance 3/3.
- The remaining frames use contexts at distance 1/2.

The repository trains three models. Each model covers one hierarchy level.
Use `codec/src/training_experiments.ipynb` for the standard training flow.
The notebook launches `train.py` with `torchrun` and `DistributedDataParallel`.

`train.sh` is the original single-GPU recipe:

```bash
cd codec/src
./train.sh 0   # distances 6/6
./train.sh 1   # distances 3/3
./train.sh 2   # distances 1/2
```

`train.py` falls back to single-GPU mode when `LOCAL_RANK` is absent.

Do not confuse these three temporal-hierarchy models with the five quality levels in
the streamer.

### Inputs and motion

Training uses 64x64 patches (`--patch 64`). For each target frame, the dataset loads:

- the target frame,
- a close context frame,
- a far context frame,
- four motion-vector channels.

A U-Net processes the context frames. The code warps the U-Net features into the
target frame coordinate system with `grid_sample`. The encoder and decoder fuse the
warped features. See `get_flows()`, `warp_unet_outputs()`, and `prepare_inputs()` in
`codec/src/util.py`.

Missing motion-vector files produce zero-filled motion and a warning. These gaps
are largely deterministic: the bundled FFmpeg 2.7.2 `extract_mvs` source, built
against `mv_env` FFmpeg 2.8.6, drops long-distance motion vectors (mostly the
6-frame and 3-frame references) on clips with large motion, so re-running
`02_motion_vectors.sh` does not fill them. 681 of 10,000 clips have at least one
missing flow file; the missing files are a small fraction (~0.1%) of all flow
files. The codec zero-fills them, so the impact on training is small.
`scripts/05_fix_motion_vectors.py` audits the gaps and re-extracts incomplete
clips.

### Progressive coding

The main training loop is in `main()` in `codec/src/train.py`:

1. The code centers the target frame around 0.5.
2. A ConvLSTM encoder reads the current target or residual. It writes a code map.
3. `Binarizer` writes `bits x H/16 x W/16` binary values.
4. The base decoder reconstructs a frame. The code subtracts this frame from the target
   to make the next residual.
5. The loop repeats for 10 iterations. The default is `--iterations 10`.

The codec does not entropy-code the codes. `run_eval()` in `codec/src/evaluate.py`
packs them with `np.packbits` into `.codes` files.

### Layered decoder and early exits

`DecoderCell2` in `codec/src/network.py` implements the layered output:

- It stacks the 10 progressive code maps.
- It applies grouped convolutions and four ConvLSTM stages.
- It writes four early-exit reconstructions and one full-resolution output.

These five outputs are the natural reconstruction points for the Swift quality
ladder.

Training zero-pads future code slots. Therefore `d2` trains on prefixes of the
progressive bitstream. Partial decoding and truncated decoding can then work.

The loss combines three parts:

- the base decoder output,
- the full `d2` output,
- all four early exits.

See the loss calculation in `codec/src/train.py`.

### I-frames

`codec/icodec/` contains a separate progressive recurrent image codec:

```bash
cd codec/icodec
python encoder.py --model checkpoint/... --input image.png --cuda --output ex --iterations 16
python decoder.py --model checkpoint/... --input ex.npz --output output_dir
```

The decoder writes one image for each progressive iteration. These images give
multiple quality levels. I-frames use this codec. Inter-frames use `codec/src/`.

### Semantic codec (`codec/semantic/`)

`codec/semantic/` is an implemented copy of `codec/src/` plus semantic
extensions. The base tree stays pristine for baseline comparisons. The two trees
are maintained separately; a change in one tree does not propagate to the other.

Each target frame gets two binary side-information layers:

- `S`: the SAM3 object mask, called the semantic base layer.
- `E`: the Canny edge map, called the structural layer.

`dataset.py` appends `S` and `E` as channels 13 and 14. The loaded tensor
therefore has 15 channels: 3 target + 6 context + 4 motion-vector + `S` + `E`.
The base codec uses 13 channels. The loader zero-fills a missing mask or edge
file, so training can start before extraction finishes. `ctx_frames` uses
channels `[0, 1, 2, 6, 7, 8]`, so `S` and `E` never enter the context U-Net.

The mask and edge layers are lossless in this version. `mask_codec.py` packs
them with `np.packbits` and `np.savez_compressed`. Evaluation reports their cost
as bytes per frame. `semantic_metrics.py` provides `mask_iou()` and `edge_f1()`,
but evaluation does not call them yet: it reports `1.0` when the corresponding
map is present.

The visual loss has two modes selected by `--loss-mode`:

- `legacy` (default) weights the loss by

  ```text
  W = 1 + alpha_sem * S + alpha_edge * E
  ```

  normalized to mean 1 per sample. `--alpha-sem` defaults to 1.0 and
  `--alpha-edge` to 0.5. With no mask or edge input, `W` is all ones, so the
  loss matches the baseline.
- `region` uses a mean-1 weighted L1 term over the feathered masks. The default
  (overlap) weight map is

  ```text
  w = w_roi * S_f + w_edge * E_f + w_bg * (1 - S_f)
  ```

  `--region-partition disjoint` switches to a mutually exclusive split, so
  "background" strictly means pixels that are neither object nor edge:

  ```text
  w = w_roi * S_f + w_edge * E_f * (1 - S_f) + w_bg * (1 - S_f) * (1 - E_f)
  ```

  The term is `sum(|x - y| * w) / (sum(w) * C + eps)`. `S_f` and `E_f` come
  from `feathered_mask(S, E, --gate-feather)`. The weights are per phase.
  `--phase-schedule` sets the phase of each progressive step. `--w-roi`,
  `--w-edge`, and `--w-bg` are per-phase weight lists. `--region-norm mean1`
  selects this normalization. `--region-norm sum` keeps the older
  sum-of-masked-means form, and its ROI/background/edge terms stay connected to
  the graph. A crop uses the region loss when it has an object mask *or* an edge
  map. A missing `S_f` with a present `E_f` becomes a zero tensor, so the edge
  and background weights still apply. Plain L1 is used only when both maps are
  empty for that sample. With `--gate-mode mask`, phase 1 gates on `S_f`, so an
  edge-only crop does not update the object phase; phase 2 gates on `E_f` and
  applies the edge weights.
  `--region-partition` defaults to `overlap`. The fallback fix changes every
  region ablation (B, C, D, D-soft, D-grouping, D-Strong). Existing B/D
  checkpoints, and any C run before the fix, used the old fallback. Retrain D as
  the control for D-Strong and treat B as an old-fallback checkpoint until it is
  retrained. D-Strong runs started after the fix use the current fallback. An A
  run that does not record `--gate-mode none` was trained with the old `mask`
  default and is not a clean ungated baseline.

The perceptual term uses the official `lpips` package with a spatial map, not
the older VGG-L2 feature distance. `--lpips-scope` (`full`/`roi`) and
`--lpips-steps` (`final`/`last3`/`all`) control the LPIPS term. `--lpips-net`
selects the `alex` or `vgg` LPIPS backbone. `--use-vgg-l2` adds the older VGG-L2
distance as an eval-only diagnostic at the final step.

`--use-conditioning` adds conditioning that reads `(S, E)` and does not change
the input channels of existing layers. `--conditioning-mode legacy` uses a
pooled `FiLM` at the four pixel-shuffle scales of `DecoderCell`/`DecoderCell2`
plus a `MaskAdd` branch in the encoder. `--conditioning-mode spatial` uses
`SpatialFiLM`. `SpatialFiLM` is spatially varying FiLM: it pools the semantic
map to the feature resolution (not `(1,1)`) and applies a per-position scale and
shift. It is not SPADE; there is no normalization layer.
`--conditioning-init zero` zero-initializes the new heads so the module is the
identity at start.

`--spatial-bit-alloc` adds a hard code-domain mask (used by D-Strong+Bit-Alloc).
After the binarizer, `code_block_mask()` in `semantic_loss.py` downsamples the
raw `S` (phase 1) or `E` (phase 2) to code resolution and zeroes masked code
positions, so those positions transmit no bits. Phases 0 and 3 return `None`
(full code map). The mask is applied identically in `train.py` and
`util.py:forward_model`, so training, in-training eval, standalone eval, and
`semantic_inference.py` all agree. `evaluate.py` counts only non-masked positions
for the per-prefix ramp and the total bytes (`code_bitrate_bytes` in
`mask_codec.py`). The `.codes` files are still written full; the sparse layout is
not materialized in a file because nothing in the repo decodes `.codes` files.

The `--gate-mode` default is `none`. D, D-grouping, D-Strong, and
D-Strong+Bit-Alloc pass `--gate-mode mask`; A, B, C, and D-soft pass `none`.
This keeps A and other legacy runs ungated because the dataset always supplies
zero-filled `S`/`E` channels, even when no mask arguments are given.

The ablation ladder is:

- **A** (`A*`): baseline. No mask or edge arguments. Legacy loss. No LPIPS. No
  gating. Its early-exit path matches `codec/src/` (see below).
- **B**: region loss (`--loss-mode region`, `--region-norm mean1`) plus official
  LPIPS. No conditioning. No gating. A flat phase schedule.
- **C**: B plus spatial conditioning (`--use-conditioning`,
  `--conditioning-mode spatial`, `--conditioning-init zero`).
- **D**: C plus residual gating (`--gate-mode mask`) and the object-first phase
  schedule (`--phase-schedule 0,1,1,1,2,2,3,3,3,3`).
- **D-soft**: C plus the object-first phase schedule with `--gate-mode none`.
- **D-grouping**: D with the ROI-late phase schedule (`0,3,3,3,2,2,1,1,1,1`)
  and ROI-late weights.
- **D-Strong**: a tuned object-first proposal. Disjoint three-way partition
  (`--region-partition disjoint`), object weights `1,8,1,0.3`, background
  `1,0.1,0.5,1`, `--gate-feather 1`, `--lpips-scope roi`, and the balanced phase
  schedule `0,1,1,1,1,2,2,3,3,3`. Trained from scratch.
- **D-Strong+Bit-Alloc**: D-Strong plus `--spatial-bit-alloc`, so object/edge
  layers transmit fewer bits. Trained from scratch.

The notebook never adds `--use-wlpips` to A. For B and later, `USE_WLPIPS`
controls the weighted LPIPS term. The notebook sets `BEST_METRIC=psnr+msssim`
for all runs.

Warm start: B starts from A. C starts from B. D-soft and D-grouping start from
C. D, D-Strong, and D-Strong+Bit-Alloc train from scratch (`LOAD_MODEL_DIR =
None`). Point `--load-model-dir` at the source `model/` directory. Load B from A
with `--load-strict false`; the refinement `extra_*` keys stay zero-init. Load C
from B with `--load-strict false`; the zero-init conditioning heads keep C
identical to B at step 0. Load D-soft and D-grouping from C with
`--load-strict true`; gating and the phase schedule do not add parameters. A
single-run resume (same `--model-dir`) uses `--load-model-name`/`--load-iter`.
Set either `RESUME_FROM` or `LOAD_MODEL_DIR`, not both; the notebook raises if
both are set. The B, C, D-soft, and D-grouping cells hardcode level 0 warm-start
sources. For levels 1 and 2, set `LOAD_MODEL_DIR` to the matching source model
directory before the run.

#### Early-exit path and LR schedule

The early-exit path for A matches `codec/src/`:

- `EarlyExit` is `PixelShuffle -> 1x1 conv` (no progressive head).
- The final decoder conv is `1x1`.
- The early-exit inputs are not detached.

The default `--ee-loss-weight` is 0.25, matching `codec/src/`. The progressive
early-exit head, detached inputs, and 3x3 output conv were a B/C-only artifact
fix. They regressed A's early exits from about 30 dB to about 16 dB and slowed
the base reconstruction, so they are reverted. Do not apply them to A.

A uses `--ee-head src`. B and all later ablations use `--ee-head refined`: `src`
plus a parallel zero-initialized refinement branch (`extra_reduce ->
PixelShuffle -> 3x3 extra_conv`). The branch is exactly zero at init, so the head
equals `src` until it learns. The existing `conv` keys and shapes are unchanged.
This keeps the B-and-later early exits from collapsing to the
checkerboard/grayscale artifact without changing A. Load B from A with
`--load-strict false` (the extra keys stay at their zero init); load
D-soft/D-grouping from C with `--load-strict true` (same refined keys). D,
D-Strong, and D-Strong+Bit-Alloc train from scratch.

The LR schedule is fraction-based. If `--schedule` is unset, `train.py` derives
the milestones from `--schedule-frac` (default `0.5,0.6,0.7,0.8,0.9`) times
`--max-train-iters`. The five milestones match the reference schedule
(`10000,12000,14000,16000,18000` at 20k) and avoid the late-training noise of
the three-milestone schedule. `--lr-schedule` selects `multistep` or `cosine`.
`--lr-warmup-iters` adds a linear warmup. `--lr-eta-min` sets the cosine floor.
`--warm-start-lr` overrides the base LR for `--load-model-dir` warm starts.

`--seed` (default `0`) seeds Python, NumPy, PyTorch, and all CUDA devices before
model and loader creation. DDP ranks use `seed + rank`. `dataset.py` passes a
top-level `seed_worker` through `functools.partial` as `worker_init_fn`, so
`spawn` workers seed from the run seed, rank, and worker id. `--deterministic`
disables cuDNN autotuning and enables deterministic kernels; without it, exact
GPU reproducibility is approximate. The notebook uses `SEED = 1234`.

The semantic notebook reads these paths:

```text
data/kinetics/train|test/<clip>/frame_%04d.png
data/kinetics/train_mv|test_mv/<clip>/...
data/kinetics/train_masks|test_masks/<clip>/frame_%04d.png
data/kinetics/train_edges|test_edges/<clip>/frame_%04d.png
```

derf and UVG use `data/<dataset>/{frames,mv,masks,edges}/<seq>/`. Use `data/`,
not `data_old/` or the original authors' paths.

The new modules are `semantic_loss.py` (weighted L1, the mean-1 region loss with
the phase schedule and the disjoint/overlap partition, the gate helpers, the
`code_block_mask` code-domain mask, and the official-LPIPS term),
`conditioning.py` (`MaskAdd`, `FiLM`, and `SpatialFiLM`), `mask_codec.py`
(lossless mask/edge bytes via `bitrate_bytes`, and sparse code accounting via
`code_bitrate_bytes`), and `semantic_metrics.py` (VMAF, official LPIPS scalars,
`lpips_map`/`lpips_map_batch`, best-effort VGG-L2, and the ROI metrics
`psnr_masked`, `bbox_from_mask`, and `msssim_bbox`).
`--use-vmaf`, `--use-lpips`, and `--use-vgg-l2` add eval-only metrics and are
best-effort. They return `None` or print nothing when the optional backend is
missing.

Evaluation reports, per progressive step and per region (full, ROI, background,
edge): PSNR, MS-SSIM (full frame and bbox-ROI), and official LPIPS. It also
reports per-early-exit PSNR and MS-SSIM, per-prefix byte counts, mask and edge
bytes per frame, `total_bytes_per_frame`, and optional VGG-L2 at the final
step. ROI and edge metrics use the raw
`S`/`E` maps, not the feathered maps, so they are comparable across
`--gate-feather` values. A sample is skipped only when both maps are empty. Edge
PSNR and LPIPS-edge include edge-only samples; ROI, background, and bbox metrics
need a nonempty object mask. A run with no mask and no edge (ablation A) reports
those metrics as `None`. Prefix accounting has three fields:
`prefix_visual_bytes_raw` (cumulative raw packed bytes before compression),
`prefix_visual_bytes` (cumulative DEFLATE code bytes, matching
`code_bitrate_bytes`), and `prefix_total_bytes` (compressed code bytes plus mask
and edge bytes). The totals are per-sample means over all evaluated samples.
`--best-metric` (`last`/`mean`/`max`/`psnr+msssim`) picks the checkpoint
criterion. New runs use `psnr+msssim`. Standalone evaluation accepts
`"best"` as the iteration. Report the final 20,000 checkpoint as the headline
number and the best checkpoint as a secondary result.
`print_hierarchy_summary` prints the per-step delta table, the group-boundary
gains in step order, raw prefix bytes, and compressed prefix bytes.

Do not modify or vendor `external/sam3`. Treat it as an external dependency.

### Codec and streamer boundary

The streamer needs the byte size of each chunk at five quality levels. The repository
includes precomputed tables:

- `streamer/new_video/.../video_size_0` through `video_size_4`,
- the legacy `streamer/bunny_chunk_size`.

The repository does **not** include the offline script that turns codec `.codes`
files into DASH tables or chunk-size tables. If a task needs a true end-to-end
encode-to-streaming run, write that bridge first.

## Codec training operations

### Notebook flow

Use `codec/src/training_experiments.ipynb` for the three hierarchy trainings.
The notebook does these steps:

1. Selects free GPUs with `PREFERRED_GPU_IDS` and `AUTO_SELECT_GPUS`.
2. Sets `CUDA_VISIBLE_DEVICES` before it imports PyTorch.
3. Writes a 352x288 frame cache to `data/prepared_352x288/`.
4. Builds a `torchrun` command for each hierarchy level.
5. Streams the output to the notebook and to `train.log`.
6. Records TensorBoard events and best checkpoints.

### Semantic notebook flow

Use `codec/semantic/semantic_experiments.ipynb` for the semantic trainings. The
notebook does these steps:

1. Selects GPUs with `PREFERRED_GPU_IDS` and `AUTO_SELECT_GPUS`. It requires at
   least 12 GiB free on each selected GPU. It sets `CUDA_VISIBLE_DEVICES` before
   it imports PyTorch.
2. Sets `HIERARCHIES`, `ABLATION` (`A`, `B`, `C`, `D`, `D-soft`,
   `D-grouping`, `D-Strong`, or `D-Strong+Bit-Alloc`), `SEED`, `RUN_LEVELS`,
   and the loss, metric, and LR flags. `semantic_arguments()` derives the gate
   mode from the ablation: only D, D-grouping, D-Strong, and
   D-Strong+Bit-Alloc use `mask`. `set_quick_test()` keeps the derived
   iteration counts and LR schedule consistent. `configure_ablation()` sets the
   D-Strong globals so the two D-Strong cells cannot drift.
3. Runs the mask/edge inspection cell. Run this cell before a long training.
4. Builds and checks the train and eval datasets for each hierarchy level.
5. Launches `codec/semantic/train.py` with `torchrun` and DDP, one cell per run.
6. Writes logs, TensorBoard events, checkpoints, and evaluation outputs under a
   new timestamped run directory
   `codec/semantic/experiments/hierarchy_<level>/<ABLATION>/<YYYYmmdd_HHMMSS>/`.
   Each `run_training` call gets its own timestamped folder, so re-runs do not
   overwrite earlier results. Set `RUN_TAG` to pin a fixed folder, or pass
   `EVAL_RUN_TAG`/`SUMMARY_RUN_TAG` to evaluate an earlier run.
   `run_config.json` records the full training argv, the base args, seed, git
   SHA, dirty-worktree state, hostname, and creation time. A stale config with
   different args or metadata raises an error; configs from before this change
   are treated as legacy and only their args are checked. A resume ignores the
   continuation flags (`--load-model-name`, `--load-iter`, and the resume
   TensorBoard directory) when comparing the recipe, so continuing the same run
   does not look like a config change. A full D-Strong run also asserts that
   `LOAD_MODEL_DIR`, `RESUME_FROM`, and `RUN_TAG` are unset.
7. Loads a saved checkpoint and runs standalone evaluation. The evaluation cell
   uses `EVAL_ABLATION` to select the ablation and evaluates `None` (latest) and
   `"best"`. The full-grid cell is disabled because it reused the previous
   ablation's globals.

Call `set_quick_test(True)` for a short check; the D-Strong cells then train
only level 0. Do not change `QUICK_TEST` without re-running the config cell,
because the derived iteration counts and LR schedule are set there. The
notebook passes `--mp-context spawn` and uses `EVAL_STRIDE_IN_TRAIN = 32`;
standalone evaluation uses `EVAL_STRIDE = 8`. In-training evaluation uses
`--eval-ee-msssim final` and `--eval-bbox-msssim final` to bound evaluation
time; standalone evaluation computes every step. Run the standalone evaluation
cell only after the selected hierarchy trainings finish. An evaluation between
trainings keeps a model and CUDA-cached memory alive during the next training.

### Distributed training

`codec/src/train.py` uses `DistributedDataParallel` with the NCCL backend.
`codec/semantic/train.py` uses the same pattern. The notebook launches it with
this command:

```bash
python -m torch.distributed.run --nproc_per_node <N> train.py ...
```

`CUDA_VISIBLE_DEVICES` selects the physical GPUs. Do not use `--gpus` for DDP.
`train.py` falls back to single-GPU mode when `LOCAL_RANK` is absent. The
semantic `dataset.py` also shards the eval split with `DistributedSampler` when
`world_size > 1`. `train.py` averages the evaluation metrics across ranks.

The DDP wrappers use `find_unused_parameters=True`. `unet.py` defines an unused
`up4` layer. Remove `up4` only after you migrate the checkpoints.

### Logs and checkpoints

- `run_training()` renames an old `train.log` to `train_<timestamp>.log`.
- The new `train.log` is empty at first. The notebook buffers it. TensorBoard is the
  real-time view.
- For the base codec, TensorBoard events accumulate in
  `experiments/hierarchy_<level>/tensorboard/`.
- `save_best()` writes `*_best.pth` when the `--best-metric` score improves.
- `RESUME_FROM` continues a level from a saved checkpoint. Do not combine it
  with `LOAD_MODEL_DIR`; the notebook raises if both are set.
- `latest_iteration()` ignores `*_best.pth` files because `best` is not an
  integer. Standalone evaluation still accepts `iteration="best"` and loads
  `{prefix}_{net}_best.pth` directly.
- `semantic_inference.py` selects the newest run whose `train.log` contains
  `Training done.` and prints the resolved tag. If only incomplete runs exist,
  it raises; set `RUN_TAG` (and `CHECKPOINT_ITER` when needed) to override.
- `semantic_inference.py` prefers `run_config.json` for the trained flags. The
  D-soft, D-grouping, D-Strong, and D-Strong+Bit-Alloc variants fail if their
  model directory has no config; their built-in fallbacks cover the architecture
  but not a missing provenance file.
- The semantic notebook writes `train.log`, `hierarchy<level>_<net>_<iter>.pth`,
  TensorBoard events, packed codes, and probe images under a timestamped run
  directory `codec/semantic/experiments/hierarchy_<level>/<ABLATION>/<run_tag>/`.
- Each semantic ablation and each run has its own `model/`, `output/`,
  `tensorboard/`, and `train.log` subtree. Ablations and repeat runs do not
  overwrite each other. The checkpoint prefix stays `hierarchy<level>`. The
  pre-repair A/B/C runs moved to `codec/semantic/old_experiments/`.
- Reusing an old checkpoint for evaluation is safe. Do not compare runs trained
  with different loss or gate code. Check `run_config.json` and `train.log`
  before you reuse an old result.
- In the semantic notebook, `--save-codes` writes packed code files under
  `output/<suffix>/codes/`. `--save-out-img` is accepted, but `evaluate.py` does
  not dump per-frame images. Evaluation always writes one probe frame per run
  under `output/probe/`.
- Semantic evaluation reports, per step, plain and per-region (ROI, background,
  edge) PSNR, full-frame and bbox-ROI MS-SSIM, official LPIPS, the four
  early-exit PSNRs and MS-SSIMs, per-prefix raw and compressed byte counts,
  mask and edge bytes per frame, and optional VMAF and VGG-L2. ROI and edge
  metrics use the raw maps; edge metrics include edge-only samples.

### Performance notes

- `torch.nn.DataParallel` was slow for this model. Its single-threaded scatter and
  gather caused step times up to about 12 seconds with `num_crops=4`.
- `DistributedDataParallel` replaces it.
- Prefix batching in the layered decoder did not improve throughput in these tests.
  The current code does not use it.
- Keep `NUM_CROPS=2` unless you measure a gain. `NUM_CROPS=4` uses more memory.
- Keep DataLoader `persistent_workers=True`. Worker churn caused a host OOM kill in an
  earlier run.
- Monitor `system/host_mem_available_gb` and `train/batch_sec` in TensorBoard.
- A 20k semantic run on two ~20 GiB GPUs takes about 3 hours of training at
  ~0.5 s/iteration.
- Evaluation is CPU-bound on MS-SSIM. `--eval-workers` (0 = auto,
  `min(cpu_count, 8)`) sets the thread count. In-training evaluation uses
  `--eval-msssim final`, `--eval-ee-msssim final`, and `--eval-bbox-msssim
  final`, and the notebook uses `EVAL_STRIDE_IN_TRAIN = 32`. Standalone
  evaluation uses `--eval-msssim all` and `EVAL_STRIDE = 8` (set 1 for the
  full set). Keep `--eval-stride` above 1 for training runs.
- `--use-wlpips` adds an official-LPIPS forward pass and extra memory. Turn it
  off when memory is tight.

## Streamer details (`streamer/`)

### Configuration

`streamer/grad.ini` is the source of runtime settings:

- `quality_levels = 5`
- `buffer_size = 30.0` seconds
- `video_chunk_size = 2.0` seconds
- `chunk_num = 48` chunks, or 96 seconds
- `history_window_size = 8`
- `feature_dim = 10`
- `agent_num = 1`
- `train_traces_path = ./datasets/train2/`
- `test_traces_path = ./datasets/test/`

`streamer/config.py` calculates two important values:

- `REGRET_WINDOW_SIZE = buffer_size / chunk_size`, which is 15 by default.
- `OUTPUT_DIM = QUALITY_DIM + REGRET_WINDOW_SIZE - 1 + 4`, which is 23 by default.

`video_bits = [1,2,3,4,5]` are **quality indices, not real bitrates**. The code uses
them for reward normalization and logging.

The tree also keeps legacy six-level settings: `quality_levels_old`,
`video_bits_old`, and `bunny_chunk_size`.

### Simulator

`streamer/simulator.py` models a DASH-like player. The simulator does five tasks:

- It reads mahimahi-style `(timestamp, bandwidth)` traces.
- It downloads one chunk at a time.
- It tracks buffer occupancy, rebuffering, and buffer drain or sleep.
- It logs played quality and update actions.
- It loops the trace when the trace ends.

`Environment.get_video_chunk()` (`streamer/simulator.py`) has two modes:

- `next_or_regret == 0`: download the next chunk at the requested quality.
- `next_or_regret > 0`: upgrade a chunk that is already in the buffer.

The simulator models the upgrade cost as the difference between composite layer
sizes. In real-video mode, the update history becomes a key such as `"0"`, `"04"`,
or `"123"`. The cost is `size(new key) - size(old key)`.

If the upgrade cannot finish before the chunk plays, the upgrade fails. The failure
value is `regret_succeed == 2`.

The third CLI argument to `test_grad.py` enables real chunk sizes:

- `0` uses synthetic sizes from `[1,2,3,4,5]`.
- `1` reads `bunny_chunk_size`.

### Actor-critic and action space

`streamer/net.py` contains the actor and the critic. The state is a `10 x 15` matrix.
The code uses eight feature rows:

1. last played quality,
2. buffer size,
3. throughput,
4. delay,
5. sizes of the next chunk at all five qualities,
6. chunks remaining,
7. number of upgrades for each buffered chunk,
8. current quality of each buffered chunk.

`OUTPUT_DIM = 23` actions have three groups:

- Actions `0..4` choose the quality for the next chunk.
- Actions `5..18` upgrade one buffered chunk by one level.
- The last four actions are automatic "fill" actions. They select a good chunk to
  upgrade from the neighboring qualities.

A chunk can receive at most two upgrades. The action mask in
`streamer/net.py:compute_mask()` enforces this limit.

### Reward

For normal downloads, the reward is:

```text
reward = log2(Q / Q0)
         - rebuf_penalty * rebuffer_time
         - smooth_penalty * |logQ - logQprev| * max(Q,Qprev)/min(Q,Qprev)
```

An upgrade reward is the difference in this objective. The difference includes the
quality gain and the smoothness difference against the previous chunk and the next
chunk. A failed upgrade gets zero reward. This is the Grad-style "regret" formulation.

### Training and testing

```bash
cd streamer
# Set python_path in grad.ini first.
python train_grad.py
python test_grad.py ./example_model/nn_model_ep_48500.ckpt ./datasets/test/ 0
```

`train_grad.py` launches one coordinator process and `agent_num` agent processes:

- The coordinator owns the actor and critic parameters. It sends them to the agents.
- Each agent streams a full episode. It sends states, actions, and rewards back.
- The coordinator calculates `R_t = r_t + gamma * V(s_{t+1})` and the TD errors.
- The coordinator calculates the actor and critic gradients. It then applies them.
- The code saves models every 500 epochs in `results/`.
- Training stops after `ENTROPY_DECAY_ITERATIONS` epochs, which is 100,000.

`train_seq_length = 100` in `grad.ini` is not used by the current code. The update
batch is one full episode.

`central_agent` ignores `NN_MODEL`. Training always runs from scratch. Testing does
load the supplied checkpoint.

`test_grad.py` loops over the trace directory. It writes one log for each trace to
`test_results/`. Action selection samples from the softmax distribution. It does not
take the argmax. Therefore test results are stochastic.

## Environment and dependencies

- The streamer uses TensorFlow 1.x-style APIs and `tflearn`. The streamer will not work unchanged on a modern TensorFlow 2-only install.
- The repository has environment files under `environments/`, one subfolder per conda environment. `semswift_env` runs the codec. `sam3` runs the masks and edges. `mv_env` runs the motion-vector tools. The download step uses system tools only.
- The codec still uses some old PyTorch idioms: `Variable`, `.data`, and
  `volatile=True`. The codec no longer uses `scipy.misc`. It uses PIL and
  `scipy.ndimage` instead.
- The codec runs in the `semswift_env` conda environment. That environment uses
  PyTorch 2.14.1+cu126, torchvision 0.29.1+cu126, TensorBoard 2.20.0, and
  setuptools<81. Do not install a CUDA 13 build on the school hosts.
- `flows/get_mv.sh` is parameterized. It takes the input directory, output directory, width, height, and frame type. `scripts/02_motion_vectors.sh` builds and runs it in `mv_env` with FFmpeg 2.8.6 and OpenCV 2.4.11.
- `codec/src/data/*` are symlinks to the original authors' NFS paths. They are broken
  in this checkout. Do not use them. Use the repository `data/` tree instead. The
  base notebook writes a 352x288 frame cache to `data/prepared_352x288/`.
- The semantic notebook reads `data/kinetics/` directly. It does not use
  `data/prepared_352x288/` and does not resize the Kinetics frames, masks, or
  edges.
- Both codec trees use OpenCV and PIL. The semantic LPIPS term and the LPIPS
  metric use the official `lpips` package (AlexNet, bundled weights). The
  optional `--use-vgg-l2` diagnostic needs torchvision VGG16. VMAF is optional
  and needs the ffmpeg `libvmaf` filter or the `vmaf` command-line tool. All run
  in `semswift_env`.
- The streamer sets `CUDA_VISIBLE_DEVICES=''`. It is CPU-only.
- `baselines/svc` and `baselines/shvc` are separate MPEG reference codebases. They are
  not integrated with the Python pipeline.
- The base codec and the streamer do not use `external/sam3`. The scripts use it to
  extract segmentation masks. The semantic codec reads the extracted masks.

## Working conventions for agents

- Read the implementation first. Use the README only when it agrees with the code.
- Do not modify the old TensorFlow or PyTorch idioms globally. Do this only when the
  task explicitly asks for modernization. This repo is a research artifact, and
  behavior compatibility matters.
- Treat `grad.ini` as the intended configuration surface. Do not hardcode values that
  already have a config entry.
- Some tasks need updates in four files. If you modify the state or action layout,
  update all of these files:
  - `streamer/config.py`,
  - `streamer/net.py` for the input splits, gradients, and mask,
  - `streamer/train_grad.py`,
  - `streamer/test_grad.py`.
- Keep training and testing behavior in sync when you modify the reward or upgrade
  logic. The two files duplicate a large action-selection block.
- Do not modify or vendor `external/sam3`. Use it as an external dependency for
  mask extraction.
- Keep `codec/src/` and `codec/semantic/` separate. The semantic tree is a
  maintained copy, not an import shim. Re-apply valid baseline fixes to
  `codec/semantic/` manually, and make semantic-only changes only in
  `codec/semantic/`.
- Preserve semantic checkpoint compatibility when you change the architecture.
  The early-exit path matches `codec/src/`, so the pre-repair A/B/C checkpoints
  load again. A and the base of B share keys; B adds the zero-initialized
  `extra_*` refinement keys. C adds the `mask_add` and `film*` keys, so strict
  resume fails across the A/B-to-C boundary. Warm-start B from A with
  `--load-model-dir` and `--load-strict false`; the refinement heads are
  zero-initialized. Warm-start C from B with `--load-strict false`; the
  conditioning heads are zero-initialized. Warm-start D-soft and D-grouping from
  C with `--load-strict true`. D, D-Strong, and D-Strong+Bit-Alloc train from
  scratch. Do not combine a warm start with `RESUME_FROM`.
- Do not commit `codec/semantic/experiments/`,
  `codec/semantic/old_experiments/`, checkpoints, logs, TensorBoard events,
  packed `.codes` files, or probe images.
- `codec/semantic/train.sh` and `codec/semantic/train_kinetics.sh` are legacy
  recipes with demo data or old NFS paths. Use the notebook for new runs.
- Use `torchrun` or `python -m torch.distributed.run` with
  `DistributedDataParallel` for multi-GPU codec training. Do not use
  `torch.nn.DataParallel`. DataParallel is single-process and it caused severe
  slowdowns on these models.
- Select devices with `CUDA_VISIBLE_DEVICES`. `--gpus` is a legacy option and it does
  not control DDP device selection.
- Keep `find_unused_parameters=True` on the DDP wrappers. `unet.py` defines an unused
  `up4` layer. Removing the flag breaks DDP. Removing `up4` breaks checkpoint
  compatibility.
- Use `--mp-context spawn` for DDP DataLoader workers. Use `fork` only for the
  in-kernel evaluation loader.
- Keep DataLoader `persistent_workers=True`. Worker churn caused a host OOM kill in an
  earlier run.
- Write `AGENTS.md` in ASD-STE100 Simplified Technical English. Use short sentences
  and active voice.
- The repo has no formal lint, typecheck, or test harness. Use targeted smoke tests:
  - For the base codec, run `python -m py_compile`. Then run a short DDP training
    with `--max-train-iters 20`, `--eval-iters 15`, `--num-workers 2`, and
    `--mp-context spawn`. Check the exit code, the loss lines, and the best
    checkpoint.
  - For the semantic codec, compile the changed modules and run the notebook
    with `set_quick_test(True)` on level 0. Run the mask/edge inspection cell
    first. Check the loss lines, the mask and edge byte counts, the ROI scalars,
    and the best checkpoint. Run the lightweight unit tests with
    `python -m unittest tests.test_semantic -v` from `codec/semantic/`.
  - For the streamer, run a short `test_grad.py` call against one trace. Then check the
    logs and the reward.
- Use `python`, not `python3`, only when the configured `python_path` points there.
  The streamer's `grad.ini` controls the executable for subprocess tests.
