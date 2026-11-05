# AGENTS.md

## Project identity

This repository is the research artifact for **Swift: Adaptive Video Streaming with
Layered Neural Codecs** (NSDI 2022). It is not Apple's Swift language.

The repository contains three code development lines:

1. An offline layered neural video codec. The codec uses PyTorch and lives in `codec/src/`.
2. An online adaptive-bitrate (ABR) streaming simulator. The simulator uses a
   reinforcement-learning agent and TensorFlow 1.x APIs. It lives in `streamer/`.
3. A semantic codec under development. It extends the Swift codec with segmentation
   masks for a research paper. It lives in `codec/semantic/`.

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
| `codec/src/training_experiments.ipynb` | Standard notebook for the three hierarchy trainings. It launches DDP training and records TensorBoard events. |
| `codec/semantic/` | Semantic Swift codec under development for a research paper. It will use SAM3 segmentation masks. The folder is empty now. |
| `codec/icodec/` | Separate progressive I-frame codec. It derives from `pytorch-image-comp-rnn`. |
| `codec/laplacian/`, `codec/singleshot/` | Experimental codec variants and ablation codecs. They are not part of the main path. |
| `data/` | Codec training and evaluation data. The `scripts/` pipeline writes `data/kinetics/`, `data/derf/`, and `data/uvg/`. |
| `scripts/` | Dataset download and preparation pipeline. It downloads Kinetics-400 from the CVDF mirror, derf, and UVG. It extracts motion vectors, SAM3 masks, and Canny edges. |
| `environments/` | Conda and pip environment files for `semswift_env`, `sam3`, and `mv_env`. |
| `streamer/` | Grad-style ABR simulator, actor-critic agent, mahimahi traces, example model, and chunk-size tables. |
| `baselines/svc/`, `baselines/shvc/` | MPEG reference software for H.264 SVC and H.265 SHVC comparisons. The Python code does not call this software. |
| `assets/` | Paper and architecture figure. |
| `external/sam3/` | SAM 3 library (submodule). The semantic codec uses it to extract segmentation masks. Do not modify or vendor it. |

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

The semantic codec adds this mask path:

```text
video frames
  -> external/sam3 extracts segmentation masks
  -> codec/semantic trains and encodes with mask information
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
3. `03_sam3_masks.py` writes binary SAM3 ROI masks in the `sam3` env. SAM3 segments noun concepts, not actions. Kinetics labels are actions, so the script prompts `person` by default. `scripts/kinetics_prompts.yaml` overrides this per class. It needs Hugging Face access to `facebook/sam3`. It reuses one session per clip and propagates forward by default. It uses all visible GPUs by default and launches one worker per GPU. It checkpoints progress after each clip and resumes on the next run. For long runs, use `--restart-every` and `--offload-video`. Do not use `--compile` for long runs: its CUDA-graph memory grows and can OOM. Use `--gpus`, `--gpu`, or `--shard-index`/`--num-shards` to override GPU selection.
4. `04_canny_edges.py` writes binary Canny edge maps in the `sam3` env.

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

Missing motion-vector files produce zero-filled motion and a warning. Fix the
extraction before a long run.

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

The `codec/semantic/` folder is the development location for the semantic Swift codec.
The folder is empty now. The semantic codec will extend the Swift codec with semantic
information for a research paper.

The semantic codec will use segmentation masks. Extract the masks with SAM3 from
`external/sam3` for training. This work reserves the existing `data/train_masks`,
`data/eval_masks`, `data/train_edges`, and `data/eval_edges` folders. The base codec in
`codec/src/` ignores masks and edge maps.

Reuse the existing codec code where practical. Do not copy the training loop into
`codec/semantic/`. Import the base modules or add controlled extensions.

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

### Distributed training

`codec/src/train.py` uses `DistributedDataParallel` with the NCCL backend.
The notebook launches it with this command:

```bash
python -m torch.distributed.run --nproc_per_node <N> train.py ...
```

`CUDA_VISIBLE_DEVICES` selects the physical GPUs. Do not use `--gpus` for DDP.
`train.py` falls back to single-GPU mode when `LOCAL_RANK` is absent.

The DDP wrappers use `find_unused_parameters=True`. `unet.py` defines an unused
`up4` layer. Remove `up4` only after you migrate the checkpoints.

### Logs and checkpoints

- `run_training()` renames an old `train.log` to `train_<timestamp>.log`.
- The new `train.log` is empty at first. The notebook buffers it. TensorBoard is the
  real-time view.
- TensorBoard events accumulate in `experiments/hierarchy_<level>/tensorboard/`.
- `save_best()` writes `*_best.pth` when evaluation PSNR improves.
- `RESUME_FROM` continues a level from a saved checkpoint.
- `latest_iteration()` ignores `*_best.pth` files.

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
  notebook writes a 352x288 frame cache to `data/prepared_352x288/`.
- The streamer sets `CUDA_VISIBLE_DEVICES=''`. It is CPU-only.
- `baselines/svc` and `baselines/shvc` are separate MPEG reference codebases. They are
  not integrated with the Python pipeline.
- The base codec and the streamer do not use `external/sam3`. The semantic codec uses
  it to extract segmentation masks.

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
- Do not modify or vendor `external/sam3`. Use it as an external dependency for the
  semantic codec.
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
  - For the codec, run `python -m py_compile`. Then run a short DDP training with
    `--max-train-iters 20`, `--eval-iters 15`, `--num-workers 2`, and
    `--mp-context spawn`. Check the exit code, the loss lines, and the best checkpoint.
  - For the streamer, run a short `test_grad.py` call against one trace. Then check the
    logs and the reward.
- Use `python`, not `python3`, only when the configured `python_path` points there.
  The streamer's `grad.ini` controls the executable for subprocess tests.
