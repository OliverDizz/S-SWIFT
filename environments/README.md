# Reproducing the environments

The repository uses three conda environments. Each has its own subfolder with
an `environment.yml` (conda packages) and, where pip is used, a
`requirements.txt` (pip packages).

Run every command from the repository root.

| Environment | Used for | Setup |
|---|---|---|
| `semswift_env` | Codec training and evaluation (`codec/src`) | `environments/semswift_env/` |
| `sam3` | SAM3 masks and Canny edges (`scripts/03`, `scripts/04`) | `environments/sam3/` |
| `mv_env` | Motion-vector tools (`scripts/02_motion_vectors.sh`) | `environments/mv_env/` |

Dataset download (`scripts/01_download_datasets.sh`) uses the system `curl`,
`wget`, `tar`, `ffmpeg`, and `ffprobe`, plus a standard-library Python helper,
so it needs none of these conda environments.

## 1. Fetch the SAM3 submodule

SAM3 is a git submodule. Fetch it before creating the `sam3` environment:

```bash
git submodule update --init --recursive
```

## 2. Create the environments

```bash
# Codec training / evaluation
conda env create -f environments/semswift_env/environment.yml
conda activate semswift_env
pip install -r environments/semswift_env/requirements.txt

# Data download, SAM3 masks, Canny edges
conda env create -f environments/sam3/environment.yml
conda activate sam3
pip install -r environments/sam3/requirements.txt

# Motion-vector toolchain (conda-only, legacy)
conda env create -f environments/mv_env/environment.yml
```

## Notes

- PyTorch is installed from the PyTorch wheel index (cu126 for the codec,
  cu128 for SAM3). The `--extra-index-url` lines in each `requirements.txt`
  handle this. Do not install a CUDA 13 build on the school hosts.
- The motion-vector binaries (`MotionVector`, `extract_mvs`) are rebuilt by
  `scripts/02_motion_vectors.sh` for the local `mv_env` paths. Pass
  `--rebuild` to force a rebuild on a new machine.
- `scripts/config.env` auto-detects the conda environment directory
  (`$CONDA_ENVS_DIR`, then `~/conda_envs`, `~/.conda/envs`, or
  `<conda base>/envs`). Set the `CONDA_ENVS_DIR` environment variable only if
  your conda environments live somewhere unusual.
