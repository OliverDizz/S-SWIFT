#!/usr/bin/env bash
#
# scripts/02_motion_vectors.sh
#
# Build the motion-vector tools (MotionVector, extract_mvs) if missing, then
# extract motion-vector images for every clip listed in the manifests.
#
# Each clip produces four images per target frame under <dataset>/mv/<clip>/:
#   frame_NNNN_{before,after}_flow_{x,y}_0001.jpg
#
# Usage:
#   bash scripts/02_motion_vectors.sh [--dataset kinetics|derf|uvg] [--limit N]
#                                     [--smoke]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

resolve_conda_envs_dir() {
    [[ -n "${CONDA_ENVS_DIR:-}" ]] && { printf '%s' "${CONDA_ENVS_DIR}"; return; }
    local dir conda_base
    for dir in "$HOME/conda_envs" "$HOME/.conda/envs"; do
        if [[ -d "${dir}/sam3" || -d "${dir}/mv_env" ]]; then
            printf '%s' "${dir}"; return
        fi
    done
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${conda_base}" && ( -d "${conda_base}/envs/sam3" || -d "${conda_base}/envs/mv_env" ) ]]; then
        printf '%s' "${conda_base}/envs"; return
    fi
    for dir in "$HOME/conda_envs" "$HOME/.conda/envs"; do
        [[ -d "${dir}" ]] && { printf '%s' "${dir}"; return; }
    done
    printf '%s' "$HOME/.conda/envs"
}
export CONDA_ENVS_DIR="$(resolve_conda_envs_dir)"

source "${SCRIPT_DIR}/config.env"

DATASET=""
LIMIT=0
SMOKE=0
REBUILD=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="${2:-}"; shift 2 ;;
        --limit) LIMIT="${2:-0}"; shift 2 ;;
        --smoke) SMOKE=1; shift ;;
        --rebuild) REBUILD=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
if [[ "${SMOKE}" -eq 1 && -z "${DATASET}" ]]; then
    LIMIT=1
fi

export PYTHON SYSTEM_FFPROBE MV_ENV_DIR EXTRACT_MVS_BIN MOTIONVECTOR_BIN GET_MV_SH

# Atomic "Clip N/Total" line shared by the parallel workers.
progress_tick() {
    local total="${PROGRESS_TOTAL:-0}"
    if command -v flock >/dev/null 2>&1 && [[ -n "${PROGRESS_STATE:-}" ]]; then
        flock "${PROGRESS_STATE}.lock" bash -c '
            state="$1"; total="$2"
            n=$(cat "$state" 2>/dev/null || echo 0); n=$((n + 1))
            printf "%s" "$n" > "$state"
            if [ -t 2 ]; then
                if [ "$total" -gt 0 ]; then
                    printf "\rClip %d/%d" "$n" "$total" >&2
                else
                    printf "\rClip %d" "$n" >&2
                fi
            fi
        ' _ "${PROGRESS_STATE}" "${total}"
    else
        local n=1
        if [[ -n "${PROGRESS_STATE:-}" ]]; then
            n=$(( $(cat "${PROGRESS_STATE}" 2>/dev/null || echo 0) + 1 ))
            printf '%s' "${n}" > "${PROGRESS_STATE}"
        fi
        if [[ -t 2 ]]; then
            if [[ "${total}" -gt 0 ]]; then
                printf '\rClip %d/%d' "${n}" "${total}" >&2
            else
                printf '\rClip %d' "${n}" >&2
            fi
        fi
    fi
}
export -f progress_tick

build_mv_tools() {
    if [[ "${REBUILD}" -eq 1 || ! -x "${MOTIONVECTOR_BIN}" ]]; then
        echo "Building MotionVector..."
        PKG_CONFIG_PATH="${MV_ENV_PKG_CONFIG_PATH}" \
        "${MV_ENV_GXX}" -O2 -D_GLIBCXX_USE_CXX11_ABI=0 \
            "${DRAW_FLOW_SRC}" -o "${MOTIONVECTOR_BIN}" \
            $("${MV_ENV_PKG_CONFIG}" --cflags --libs opencv)
    fi
    if [[ "${REBUILD}" -eq 1 || ! -x "${EXTRACT_MVS_BIN}" ]]; then
        echo "Building extract_mvs..."
        PKG_CONFIG_PATH="${MV_ENV_PKG_CONFIG_PATH}" \
        "${MV_ENV_GCC}" -O2 "${EXTRACT_MVS_SRC}" -o "${EXTRACT_MVS_BIN}" \
            $("${MV_ENV_PKG_CONFIG}" --cflags --libs libavcodec libavformat libavutil)
    fi
}

mv_worker() {
    local dataset="$1" split="$2" clip="$3" frames="$4" mv="$5"
    local first dims w h
    first="$(ls "${frames}"/frame_*.png 2>/dev/null | head -1)"
    if [[ -z "${first}" ]]; then
        printf '\nno frames: %s/%s\n' "${dataset}" "${clip}" >&2
        progress_tick
        return 0
    fi
    dims="$("${SYSTEM_FFPROBE}" -v error -select_streams v:0 \
        -show_entries stream=width,height -of csv=s=x:p=0 "${first}")"
    w="${dims%x*}"
    h="${dims#*x}"
    if ! PATH="${MV_ENV_DIR}/bin:${PATH}" \
        EXTRACT_MVS="${EXTRACT_MVS_BIN}" MOTIONVECTOR="${MOTIONVECTOR_BIN}" \
        bash "${GET_MV_SH}" "${frames}" "${mv}" "${w}" "${h}" png; then
        printf '\nFAIL mv: %s/%s/%s\n' "${dataset}" "${split}" "${clip}" >&2
    fi
    progress_tick
}
export -f mv_worker

run_dataset() {
    local dataset="$1"
    local limit_flag=()
    if [[ "${LIMIT}" -gt 0 ]]; then
        limit_flag=(--limit "${LIMIT}")
    fi
    local list_file total
    list_file="$(mktemp)"
    "${PYTHON}" "${SCRIPT_DIR}/config.py" \
        --dataset "${dataset}" "${limit_flag[@]}" > "${list_file}"
    total="$(wc -l < "${list_file}")"
    PROGRESS_STATE="$(mktemp)"
    printf '0' > "${PROGRESS_STATE}"
    export PROGRESS_TOTAL="${total}" PROGRESS_STATE
    xargs -r -P "${PARALLELISM}" -n 7 bash -c 'mv_worker "$@"' _ < "${list_file}"
    printf '\n' >&2
    rm -f "${list_file}" "${PROGRESS_STATE}" "${PROGRESS_STATE}.lock"
}

build_mv_tools

if [[ -n "${DATASET}" ]]; then
    run_dataset "${DATASET}"
else
    run_dataset kinetics
    run_dataset derf
    run_dataset uvg
fi

echo "02_motion_vectors.sh done."
