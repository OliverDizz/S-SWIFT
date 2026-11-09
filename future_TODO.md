# Side task: Semantic rate savings and ROI-oriented streaming

## Objective

Turn the semantic codec's mask/edge side information into a real end-to-end gain:
either **smaller total size at equal quality**, **better quality at equal total
size**, or **ROI-prioritized streaming**. Measure all results against ablation A
(baseline) and include the side-information bytes in the total bitrate.

## Background and current gap

- The semantic codec (`codec/semantic/`) adds two binary side channels: `S` (SAM3
  mask) and `E` (Canny edges) at full resolution (channels 13 and 14 of the loaded
  tensor).
- Training uses a normalized weight map
  `W = 1 + alpha_sem * S + alpha_edge * E` (mean 1 per sample). This only
  **reallocates** distortion toward the ROI and edges; it does not reduce the
  number of codes.
- The visual bitstream is fixed: 10 progressive iterations x `bits` x H/16 x W/16.
  The codec does not entropy-code the visual codes.
- `S` and `E` are stored losslessly (`mask_codec.py`: `np.packbits` +
  `np.savez_compressed`) and reported as **bytes per frame** in evaluation.
- Result: total size = visual codes + mask bytes + edge bytes. The current
  implementation gives an ROI quality gain but **adds** side-information bytes. It
  is not yet a strict rate win.
- The streamer (`streamer/`) is not wired to the mask. ROI-aware streaming is not
  implemented.

## Work items (pick one)

### 1. Mask-driven adaptive coding (recommended for "smaller file, same quality")

- Use `S` to vary the number of progressive iterations/codes per region: full
  budget on the ROI, fewer (or zero) iterations on the background.
- Objective: keep ROI quality equal to baseline while cutting total visual bits.
  The side information then becomes an investment instead of pure overhead.
- Files: `codec/semantic/train.py` (progressive loop),
  `codec/semantic/network.py` (code packing/decoding),
  `codec/semantic/evaluate.py` (bit counting),
  `codec/semantic/util.py` (`prepare_inputs`).
- Metric: ROI PSNR/LPIPS versus total bytes (visual + S + E), A vs B vs C, on
  Kinetics test.

### 2. Efficient side-information coding

- Replace lossless `np.savez_compressed` with a real coder: contour/run-length for
  masks, a learned or arithmetic-coded edge map, or a cheaper decoder-side mask
  reconstruction.
- Objective: cut the `mask_bitrate` + `edge_bitrate` bytes per frame with
  negligible ROI-quality loss.
- Files: `codec/semantic/mask_codec.py`, `codec/semantic/evaluate.py`.

### 3. ROI-oriented streaming

- Feed the mask into `streamer/` so the ABR agent prioritizes/upgrades ROI chunks,
  or weights the regret/upgrade reward by the ROI.
- Objective: better QoE (ROI quality, rebuffer time, smoothness) at the same or
  lower total bandwidth.
- Files: `streamer/config.py`, `streamer/net.py`, `streamer/simulator.py`,
  `streamer/train_grad.py`, `streamer/test_grad.py`. Update state, action, and
  reward in all four files together, per `AGENTS.md`.
- Note: no bridge currently turns `.codes` + masks into DASH/chunk-size tables.
  Write that bridge first.

## Success criteria

- Report **ROI quality versus background quality** separately. Do not report only
  whole-frame PSNR.
- Put **mask + edge bytes on the bitrate axis** in every RD/BD-rate comparison.
- Keep the ablation ladder as the control:
  - **A**: baseline loss (never wLPIPS).
  - **B**: weighted loss.
  - **C**: B plus conditioning.
- Preserve checkpoint compatibility rules in `AGENTS.md`. A and B share state-dict
  keys. C adds `mask_add` and `FiLM` keys, so train C from scratch.

## Constraints

- Keep `codec/src/` pristine. Re-apply baseline fixes to `codec/semantic/` manually.
- Do not modify or vendor `external/sam3`.
- Use `torchrun`/DDP with `--mp-context spawn` and
  `find_unused_parameters=True`. Keep DataLoader `persistent_workers=True`.
