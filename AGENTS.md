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
| `flows/` | Extracts motion vectors with a bundled FFmpeg 2.7.2 and OpenCV code. It writes `_before_flow_x/y` and `_after_flow_x/y` images for the codec. |
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
are largely deterministic: the bundled ffmpeg 2.8.6 + `MotionVector` pipeline
drops long-distance motion vectors (mostly the 6-frame and 3-frame references)
on clips with large motion, so re-running `02_motion_vectors.sh` does not fill
them. They affect ~0.1% of flow files (681/10,000 clips have at least one) and
are zero-filled by the codec, so the impact on training is small.
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
as bytes per frame. `mask_iou()` and `edge_f1()` in `semantic_metrics.py` are
placeholders and are not used yet.

The visual loss has two modes selected by `--loss-mode`:

- `legacy` (default) weights the loss by

  ```text
  W = 1 + alpha_sem * S + alpha_edge * E
  ```

  normalized to mean 1 per sample. `--alpha-sem` defaults to 1.0 and
  `--alpha-edge` to 0.5. With no mask or edge input, `W` is all ones, so the
  loss matches the baseline.
- `region` replaces `W` with an explicit ROI/background/edge loss per
  progressive step:

  ```text
  L = roi_weight  * masked_l1(recon, target, roi_map)
    + bg_weight   * masked_l1(recon, target, 1 - S)
    + edge_weight * masked_l1(recon, target, E)
    + lpips_weight * lpips(...)
  ```

  where `masked_l1` uses masked means (`sum(|x-y| * m) / (sum(m) * C + eps)`).
  `--roi-weight` (2.0), `--bg-weight` (0.5), and `--edge-weight` (0.5) set the
  term weights. `--boundary-band` (>0) boosts a max-pool dilated boundary ring
  around the ROI by `--boundary-weight`. `--roi-ramp linear` ramps the ROI
  weight from `--roi-ramp-start` at step 1 to `1.0` at step N. Empty masks
  fall back to the legacy term for that sample and the count is logged.
  `--lpips-scope` (`full`/`roi`) and `--lpips-steps` (`final`/`last3`/`all`)
  control the LPIPS term; `last3` avoids the final-only objective that caused
  the step-10 PSNR collapse.

`--use-conditioning` adds conditioning that reads `(S, E)` and does not change
the input channels of existing layers. `--conditioning-mode legacy` uses a
pooled `FiLM` at the four pixel-shuffle scales of `DecoderCell`/`DecoderCell2`
plus a `MaskAdd` branch in the encoder. `--conditioning-mode spatial` uses
`SpatialFiLM`, which pools the semantic map to the feature resolution (not
`(1,1)`) so the modulation varies across space. `--conditioning-init zero`
zero-initializes the new heads so the module is the identity at start.

The ablation ladder is:

- **A**: baseline copy, no mask or edge arguments, legacy loss, no LPIPS.
- **B**: region loss (`--loss-mode region`). The architecture is unchanged, so
  B is checkpoint-compatible with A.
- **C**: B plus spatial conditioning (`--use-conditioning`,
  `--conditioning-mode spatial`, `--conditioning-init zero`).

A uses the baseline loss from `codec/src`. The notebook never adds
`--use-wlpips` to A. For B and C, `USE_WLPIPS` controls the weighted LPIPS term.
The notebook sets `LPIPS_SCOPE=roi` and `LPIPS_STEPS=last3` for B/C, and
`BEST_METRIC=mean` so the best checkpoint is selected by mean PSNR over steps
instead of the anomalous final step.

Warm start: C starts from B, not from scratch. Point `--load-model-dir` at B's
`model/` directory and load with `--load-strict false`. Shared weights load and
new conditioning keys keep their zero init, so C starts identical to B. A
single-run resume (same `--model-dir`) still uses strict loading via
`--load-model-name`/`--load-iter`.

The semantic notebook reads these paths:

```text
data/kinetics/train|test/<clip>/frame_%04d.png
data/kinetics/train_mv|test_mv/<clip>/...
data/kinetics/train_masks|test_masks/<clip>/frame_%04d.png
data/kinetics/train_edges|test_edges/<clip>/frame_%04d.png
```

derf and UVG use `data/<dataset>/{frames,mv,masks,edges}/<seq>/`. Use `data/`,
not `data_old/` or the original authors' paths.

The new modules are `semantic_loss.py` (weighted L1/LPIPS and the region loss),
`conditioning.py` (`MaskAdd`, `FiLM`, and `SpatialFiLM`), `mask_codec.py`
(lossless mask/edge bytes), and `semantic_metrics.py` (VMAF, LPIPS, and the ROI
metrics `psnr_masked`, `bbox_from_mask`, `msssim_bbox`, `lpips_roi`; IoU/F1 are
still placeholders). `--use-wlpips` adds a VGG16 feature-space LPIPS term to
training. `--use-vmaf` and `--use-lpips` add eval-only metrics and are
best-effort. They return `None` or print nothing when the optional backend is
missing.

Evaluation reports ROI-masked PSNR, background PSNR, bbox MS-SSIM, ROI LPIPS,
and `total_bytes_per_frame` (mask + edge + visual code bytes) in addition to
the full-frame metrics. A run with all-zero masks (ablation A) skips the ROI
metrics and reports them as `None`. `--best-metric` (`last`/`mean`/`max`) picks
the checkpoint PSNR criterion; new B/C runs use `mean`.

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
2. Sets `HIERARCHIES`, `ABLATION` (`A`, `B`, or `C`), and the loss and metric
   flags.
3. Runs the mask/edge inspection cell. Run this cell before a long training.
4. Builds and checks the train and eval datasets for each hierarchy level.
5. Launches `codec/semantic/train.py` with `torchrun` and DDP, one cell per run.
6. Writes logs, TensorBoard events, checkpoints, and evaluation outputs under a
   new timestamped run directory
   `codec/semantic/experiments/hierarchy_<level>/<ABLATION>/<YYYYmmdd_HHMMSS>/`.
   Each `run_training` call gets its own timestamped folder, so re-runs do not
   overwrite earlier results. Set `RUN_TAG` to pin a fixed folder, or pass
   `EVAL_RUN_TAG`/`SUMMARY_RUN_TAG` to evaluate an earlier run.
7. Loads a saved checkpoint and runs standalone evaluation. The evaluation cell
   uses `EVAL_ABLATION` to select the ablation.

Set `QUICK_TEST = True` for a short check. The notebook passes
`--mp-context spawn` and `--eval-stride 8`, which evaluates every eighth test
sample. Run the standalone evaluation cell only after the selected hierarchy
trainings finish. An evaluation between trainings keeps a model and CUDA-cached
memory alive during the next training.

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
- `save_best()` writes `*_best.pth` when evaluation PSNR improves.
- `RESUME_FROM` continues a level from a saved checkpoint.
- `latest_iteration()` ignores `*_best.pth` files.
- The semantic notebook writes `train.log`, `hierarchy<level>_<net>_<iter>.pth`,
  TensorBoard events, packed codes, and probe images under a timestamped run
  directory `codec/semantic/experiments/hierarchy_<level>/<ABLATION>/<run_tag>/`.
- Each semantic ablation and each run has its own `model/`, `output/`,
  `tensorboard/`, and `train.log` subtree. A/B/C runs and repeat runs do not
  overwrite each other. The checkpoint prefix stays `hierarchy<level>`.
- In the semantic notebook, `--save-codes` writes packed code files under
  `output/<suffix>/codes/`. `--save-out-img` is accepted, but `evaluate.py` does
  not dump per-frame images. Evaluation always writes one probe frame per run
  under `output/probe/`.
- Semantic evaluation reports loss, MS-SSIM, PSNR, the four early-exit PSNRs,
  mask and edge bytes per frame, and optional VMAF and LPIPS.

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
- A 20k semantic run on two ~20 GiB GPUs takes about 3 hours at ~0.5 s/iteration.
  Keep `--eval-stride` above 1 to bound evaluation time.
- `--use-wlpips` adds a VGG16 forward pass and extra memory. Turn it off when
  memory is tight.

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

`Environment.get_video_chunk()` (`streamer/simulator.py:147-452`) has two modes:

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

A chunk can receive at most two upgrades. The action mask in `streamer/net.py:337-355`
enforces this limit.

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
- Both codec trees use OpenCV and PIL. Weighted LPIPS needs torchvision VGG16.
  VMAF is optional and needs the ffmpeg `libvmaf` filter or the `vmaf`
  command-line tool. All run in `semswift_env`.
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
  A and B share the same state-dict keys. C adds `mask_add` and `film*` keys, and
  strict resume fails across the A/B-to-C boundary. Warm-start C from B with
  `--load-model-dir` (B's `model/`) and `--load-strict false`; the new
  conditioning keys are zero-initialized via `--conditioning-init zero`, so C
  starts identical to B.
- Do not commit `codec/semantic/experiments/`, checkpoints, logs, TensorBoard
  events, packed `.codes` files, or probe images.
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
  - For the semantic codec, compile the changed modules and run the notebook with
    `QUICK_TEST = True` on level 0. Run the mask/edge inspection cell first. Check
    the loss lines, the mask and edge byte counts, the ROI scalars, and the best
    checkpoint. Run the lightweight unit tests with
    `python -m unittest tests.test_semantic -v` from `codec/semantic/`.
  - For the streamer, run a short `test_grad.py` call against one trace. Then check the
    logs and the reward.
- Use `python`, not `python3`, only when the configured `python_path` points there.
  The streamer's `grad.ini` controls the executable for subprocess tests.
