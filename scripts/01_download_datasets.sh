#!/usr/bin/env bash
#
# scripts/01_download_datasets.sh
#
# Download and prepare the raw datasets for the Swift codec:
#   - Kinetics-400: N_TRAIN + N_TEST clips, 100 frames/clip at 352x288, taken
#     from the official CVDF S3 mirror (tar.gz archives, no YouTube access).
#   - derf (Xiph) and UVG (FullHD) test sequences at native resolution.
#
# The script writes the split manifest and the per-sequence manifests under
# data/manifests/, then decodes frames under data/.
#
# Usage:
#   bash scripts/01_download_datasets.sh [--limit N] [--skip-kinetics]
#                                        [--skip-derf] [--skip-uvg] [--dry-run]

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

# ---- CLI flags -------------------------------------------------------------
LIMIT=0
SKIP_KINETICS=0
SKIP_DERF=0
SKIP_UVG=0
DRY_RUN=0
RESAMPLE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit) LIMIT="${2:-0}"; shift 2 ;;
        --skip-kinetics) SKIP_KINETICS=1; shift ;;
        --skip-derf) SKIP_DERF=1; shift ;;
        --skip-uvg) SKIP_UVG=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --resample) RESAMPLE=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# Export the values the inline Python driver needs.
export N_TRAIN N_TEST FRAMES_PER_CLIP KINETICS_W KINETICS_H SPLIT_SEED
export KINETICS_DIR RAW_DIR MANIFESTS_DIR PARALLELISM LIMIT RESAMPLE
export KINETICS_MANIFEST SCRIPT_DIR
export PYTHON SYSTEM_FFMPEG SYSTEM_FFPROBE

DOWNLOAD_LOG="${MANIFESTS_DIR}/download.log"

# derf: sequence key -> y4m filename on media.xiph.org.
declare -A DERF_FILES=(
    [blue_sky]="blue_sky_1080p25.y4m"
    [flower]="flower_cif.y4m"
    [foreman]="foreman_cif.y4m"
    [four_people]="FourPeople_1280x720_60.y4m"
    [soccer]="soccer_cif.y4m"
    [stefan]="stefan_sif.y4m"
    [tennis]="tennis_sif.y4m"
    [vidyo3]="vidyo3_720p_60fps.y4m"
    [waterfall]="waterfall_cif.y4m"
    [silent]="silent_cif.y4m"
)

# UVG: the extracted YUV prefix can differ from the archive name.
declare -A UVG_PREFIX=(
    [ReadySetGo]="ReadySteadyGo"
)

log_fail() {
    echo "$*" | tee -a "${DOWNLOAD_LOG}" >&2
}

setup_dirs() {
    mkdir -p "${RAW_DIR}/kinetics/train" "${RAW_DIR}/kinetics/test"
    mkdir -p "${RAW_DIR}/kinetics/targz/train" "${RAW_DIR}/kinetics/targz/val"
    mkdir -p "${RAW_DIR}/derf" "${RAW_DIR}/uvg"
    mkdir -p "${KINETICS_DIR}/train" "${KINETICS_DIR}/test"
    mkdir -p "${DERF_DIR}/frames" "${UVG_DIR}/frames"
    mkdir -p "${MANIFESTS_DIR}"
}

download_kinetics_csv() {
    curl -fL --retry 3 -sS -o "${MANIFESTS_DIR}/train.csv" "${K400_TRAIN_CSV}"
    curl -fL --retry 3 -sS -o "${MANIFESTS_DIR}/val.csv" "${K400_VAL_CSV}"
    echo "Downloaded Kinetics-400 annotations."
}

# download_kinetics_tars <source_split> <dest_split> <n_archives>
# source_split selects the path list (train or val); dest_split selects the
# extraction/frames directory (train or test).
download_kinetics_tars() {
    local source="$1" dest="$2" nparts="$3"
    local list_url list
    if [[ "${source}" == "train" ]]; then
        list_url="${K400_TRAIN_PATH_LIST}"
    else
        list_url="${K400_VAL_PATH_LIST}"
    fi
    list="${MANIFESTS_DIR}/k400_${source}_path.txt"
    if [[ ! -f "${list}" ]]; then
        curl -fL --retry 3 -sS -o "${list}" "${list_url}"
    fi

    local targz_dir="${RAW_DIR}/kinetics/targz/${source}"
    local extract_dir="${RAW_DIR}/kinetics/${dest}"
    mkdir -p "${targz_dir}" "${extract_dir}"

    local count=0 url fn tar done_marker extract_marker
    while IFS= read -r url; do
        [[ -z "${url}" ]] && continue
        if (( count >= nparts )); then
            break
        fi
        fn="${url##*/}"
        tar="${targz_dir}/${fn}"
        done_marker="${targz_dir}/${fn}.done"
        extract_marker="${extract_dir}/.done_${fn}"

        if [[ ! -f "${done_marker}" ]]; then
            echo "Downloading ${source}/${fn}"
            if ! wget -c -P "${targz_dir}" "${url}"; then
                log_fail "FAIL tar ${source}/${fn}"
                rm -f "${tar}"
                count=$((count + 1))
                continue
            fi
            touch "${done_marker}"
        fi

        if [[ ! -f "${extract_marker}" ]]; then
            echo "Extracting ${source}/${fn} -> ${extract_dir}"
            if tar zxf "${tar}" -C "${extract_dir}"; then
                touch "${extract_marker}"
            else
                log_fail "FAIL extract ${source}/${fn}"
            fi
        fi
        count=$((count + 1))
    done < "${list}"
}

run_kinetics() {
    if [[ "${SKIP_KINETICS}" -eq 1 ]]; then
        echo "Skipping Kinetics download."
        return 0
    fi
    download_kinetics_csv

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "Dry run: would download ${K400_TRAIN_PARTS} train and ${K400_VAL_PARTS} val archives."
        return 0
    fi

    download_kinetics_tars train train "${K400_TRAIN_PARTS}"
    download_kinetics_tars val test "${K400_VAL_PARTS}"

    # Validate the committed manifest (default) or resample a new split.
    "${PYTHON}" - <<'PY'
import csv
import os
import random
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

env = os.environ
n_frames = int(env["FRAMES_PER_CLIP"])
w, h = env["KINETICS_W"], env["KINETICS_H"]
seed = int(env["SPLIT_SEED"])
kin_dir = env["KINETICS_DIR"]
raw_dir = env["RAW_DIR"]
manifest_dir = env["MANIFESTS_DIR"]
manifest = env["KINETICS_MANIFEST"]
ffmpeg = env["SYSTEM_FFMPEG"]
limit = int(env.get("LIMIT", "0") or 0)
parallelism = int(env.get("PARALLELISM", "8") or 8)
n_train = int(env["N_TRAIN"])
n_test = int(env["N_TEST"])
resample = env.get("RESAMPLE", "0") == "1"
log_path = os.path.join(manifest_dir, "download.log")

sys.path.insert(0, env["SCRIPT_DIR"])
import config  # noqa: E402

raw_for = {"train": os.path.join(raw_dir, "kinetics", "train"),
           "test": os.path.join(raw_dir, "kinetics", "test")}

header = ["split", "clip_id", "youtube_id", "label", "time_start", "time_end",
          "n_frames"]


def read_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def build_label_map(rows):
    labels = {}
    for row in rows:
        try:
            start = int(float(row["time_start"]))
        except (KeyError, ValueError):
            continue
        labels[(row["youtube_id"], start)] = row["label"]
    return labels


def parse_archive_name(filename):
    # <youtube_id>_<start:06d>_<end:06d>.mp4; youtube_id may contain '_'.
    parts = filename[:-4].rsplit("_", 2)
    if len(parts) != 3:
        return None
    youtube_id, start, end = parts
    try:
        return youtube_id, int(start), int(end)
    except ValueError:
        return None


def clip_record(split, clip_id, youtube_id, label, start, end):
    return {"split": split, "clip_id": clip_id, "youtube_id": youtube_id,
            "label": label, "time_start": start, "time_end": end,
            "n_frames": n_frames,
            "src": os.path.join(raw_for[split], clip_id + ".mp4")}


def build_clips(split, labels):
    clips = []
    raw_split_dir = raw_for[split]
    if not os.path.isdir(raw_split_dir):
        return clips
    for filename in sorted(os.listdir(raw_split_dir)):
        if not filename.endswith(".mp4"):
            continue
        parsed = parse_archive_name(filename)
        if parsed is None:
            continue
        youtube_id, start, end = parsed
        label = labels.get((youtube_id, start))
        if label is None:
            continue
        clips.append(clip_record(split, filename[:-4], youtube_id, label,
                                 start, end))
    return clips


def process(clip):
    out_dir = os.path.join(kin_dir, clip["split"], clip["clip_id"])
    existing = (len([f for f in os.listdir(out_dir) if f.startswith("frame_")])
                if os.path.isdir(out_dir) else 0)
    if existing >= n_frames:
        return "ok", clip, existing
    if not os.path.isfile(clip["src"]):
        return "missing", clip, 0
    os.makedirs(out_dir, exist_ok=True)
    cmd = [ffmpeg, "-y", "-i", clip["src"],
           "-vf", f"scale={w}:{h}:flags=area",
           "-frames:v", str(n_frames), "-start_number", "1",
           os.path.join(out_dir, "frame_%04d.png")]
    result = subprocess.run(cmd, stderr=subprocess.DEVNULL)
    written = len([f for f in os.listdir(out_dir) if f.startswith("frame_")])
    if result.returncode != 0 or written < n_frames:
        return "fail", clip, written
    return "ok", clip, written


def decode_split(clips, target, log, bar):
    # Decode in batches and stop once `target` clips are kept. Backfill comes
    # from the remaining candidates, so short or corrupt clips do not shrink
    # the split.
    kept, failures = [], 0
    order = {id(c): i for i, c in enumerate(clips)}
    batch = max(parallelism * 8, 64)
    i = 0
    with ThreadPoolExecutor(max_workers=parallelism) as ex:
        while i < len(clips) and len(kept) < target:
            chunk = clips[i:i + batch]
            i += batch
            for fut in as_completed([ex.submit(process, c) for c in chunk]):
                bar.tick()
                status, clip, _ = fut.result()
                if status == "ok":
                    kept.append(clip)
                    bar.log(f"ok {clip['split']} {clip['clip_id']} "
                            f"{clip['label']}")
                else:
                    failures += 1
                    msg = (f"FAIL {status} {clip['split']} "
                           f"{clip['clip_id']} {clip['label']}")
                    bar.log(msg)
                    log.write(msg + "\n")
    kept.sort(key=lambda c: order[id(c)])
    return kept[:target], failures


os.makedirs(manifest_dir, exist_ok=True)
train_labels = build_label_map(read_rows(os.path.join(manifest_dir, "train.csv")))
test_labels = build_label_map(read_rows(os.path.join(manifest_dir, "val.csv")))
labels_for = {"train": train_labels, "test": test_labels}

if os.path.isfile(manifest) and not resample:
    # Validate: the committed manifest is the authoritative selection.
    rows = read_rows(manifest)
    if limit:
        rows = rows[:limit]
    expected = {(r["split"], r["clip_id"]): int(r["n_frames"]) for r in rows}
    clips = [clip_record(r["split"], r["clip_id"], r["youtube_id"],
                         r["label"], int(r["time_start"]), int(r["time_end"]))
             for r in rows]
    bar = config.Progress(len(clips))
    ok = missing = failed = label_bad = frame_bad = 0
    with open(log_path, "a") as log:
        with ThreadPoolExecutor(max_workers=parallelism) as ex:
            for idx, (status, clip, written) in enumerate(
                    ex.map(process, clips), 1):
                bar.tick(idx)
                if status == "missing":
                    missing += 1
                    msg = f"MISSING {clip['split']} {clip['clip_id']}"
                    bar.log(msg + " (raise K400_TRAIN_PARTS/K400_VAL_PARTS)")
                    log.write(msg + "\n")
                elif status == "fail":
                    failed += 1
                    log.write(f"FAIL decode {clip['split']} "
                              f"{clip['clip_id']}\n")
                else:
                    ok += 1
                    ann = labels_for[clip["split"]].get(
                        (clip["youtube_id"], clip["time_start"]))
                    if ann is not None and ann != clip["label"]:
                        label_bad += 1
                        bar.log(f"LABEL MISMATCH {clip['split']} "
                                f"{clip['clip_id']}: "
                                f"manifest={clip['label']!r} "
                                f"annotation={ann!r}")
                    want = expected.get((clip["split"], clip["clip_id"]),
                                        n_frames)
                    if written != want:
                        frame_bad += 1
                        bar.log(f"FRAME COUNT {clip['split']} "
                                f"{clip['clip_id']}: {written} != {want}")
    bar.close()
    print(f"Kinetics validation: {len(clips)} clips, {ok} ok, {missing} missing, "
          f"{failed} decode-failed, {label_bad} label mismatches, "
          f"{frame_bad} frame-count mismatches.")
else:
    # Resample: build a new split and write the manifest.
    train_clips = build_clips("train", train_labels)
    test_clips = build_clips("test", test_labels)
    rng = random.Random(seed)
    rng.shuffle(train_clips)
    rng.shuffle(test_clips)
    bar = config.Progress(len(train_clips) + len(test_clips))
    with open(log_path, "a") as log:
        if limit:
            with ThreadPoolExecutor(max_workers=parallelism) as ex:
                results = []
                for idx, result in enumerate(
                        ex.map(process, (train_clips + test_clips)[:limit]), 1):
                    bar.tick(idx)
                    results.append(result)
            final = [c for s, c, _ in results if s == "ok"]
            failures = sum(1 for s, _, _ in results if s != "ok")
        else:
            train_kept, f1 = decode_split(train_clips, n_train, log, bar)
            test_kept, f2 = decode_split(test_clips, n_test, log, bar)
            final = train_kept + test_kept
            failures = f1 + f2
    bar.close()
    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    with open(manifest, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for clip in sorted(final, key=lambda c: (c["split"], c["clip_id"])):
            writer.writerow({key: clip[key] for key in header})
    if not limit:
        keep = {(c["split"], c["clip_id"]) for c in final}
        for split in ("train", "test"):
            split_dir = os.path.join(kin_dir, split)
            if not os.path.isdir(split_dir):
                continue
            for name in os.listdir(split_dir):
                if (split, name) not in keep:
                    shutil.rmtree(os.path.join(split_dir, name),
                                  ignore_errors=True)
    print(f"Kinetics resampled: {len(final)} kept, {failures} failed. "
          f"Manifest: {manifest}")
PY
}

declare -A EXP_W=()
declare -A EXP_H=()
declare -A EXP_N=()

load_manifest_expected() {
    local file="$1" s f ww hh nn
    EXP_W=(); EXP_H=(); EXP_N=()
    [[ -f "${file}" ]] || return 1
    while IFS=, read -r s f ww hh nn; do
        [[ -z "${s}" || "${s}" == "seq" ]] && continue
        EXP_W[$s]="${ww}"; EXP_H[$s]="${hh}"; EXP_N[$s]="${nn}"
    done < "${file}"
    return 0
}

download_derf() {
    if [[ "${SKIP_DERF}" -eq 1 ]]; then
        echo "Skipping derf download."
        return 0
    fi
    local validate=0 mismatches=0
    if [[ "${DRY_RUN}" -eq 0 && -f "${DERF_MANIFEST}" && "${RESAMPLE}" -eq 0 ]]; then
        validate=1
        load_manifest_expected "${DERF_MANIFEST}"
    elif [[ "${DRY_RUN}" -eq 0 ]]; then
        { echo "seq,filename,width,height,n_frames"; } > "${DERF_MANIFEST}"
    fi
    for seq in ${DERF_SEQS}; do
        local file="${DERF_FILES[$seq]}"
        local raw="${RAW_DIR}/derf/${file}"
        local out="${DERF_DIR}/frames/${seq}"
        mkdir -p "${out}"
        if [[ -z "$(ls "${out}"/frame_*.png 2>/dev/null)" ]]; then
            if [[ "${DRY_RUN}" -eq 0 ]]; then
                curl -fL --retry 3 -sS -o "${raw}" "${DERF_BASE}/${file}"
                "${SYSTEM_FFMPEG}" -y -i "${raw}" -start_number 1 \
                    "${out}/frame_%04d.png"
            fi
        fi
        if [[ "${DRY_RUN}" -eq 0 ]]; then
            local dims nframes ww hh
            dims="$("${SYSTEM_FFPROBE}" -v error -select_streams v:0 \
                -show_entries stream=width,height -of csv=s=x:p=0 \
                "${out}/frame_0001.png")"
            nframes="$(ls "${out}"/frame_*.png | wc -l)"
            ww="${dims%x*}"; hh="${dims#*x}"
            if [[ "${validate}" -eq 1 ]]; then
                if [[ "${EXP_W[$seq]:-}" != "${ww}" || "${EXP_H[$seq]:-}" != "${hh}" \
                      || "${EXP_N[$seq]:-}" != "${nframes}" ]]; then
                    echo "VALIDATE derf ${seq}: got ${ww}x${hh} ${nframes}f, expected ${EXP_W[$seq]:-?}x${EXP_H[$seq]:-?} ${EXP_N[$seq]:-?}f" >&2
                    mismatches=$((mismatches + 1))
                fi
            else
                printf '%s,%s,%s,%s,%s\n' \
                    "${seq}" "${file}" "${ww}" "${hh}" "${nframes}" \
                    >> "${DERF_MANIFEST}"
            fi
        fi
    done
    if [[ "${validate}" -eq 1 ]]; then
        echo "derf validation: ${mismatches} sequence mismatches."
    fi
}

download_uvg() {
    if [[ "${SKIP_UVG}" -eq 1 ]]; then
        echo "Skipping UVG download."
        return 0
    fi
    local validate=0 mismatches=0
    if [[ "${DRY_RUN}" -eq 0 && -f "${UVG_MANIFEST}" && "${RESAMPLE}" -eq 0 ]]; then
        validate=1
        load_manifest_expected "${UVG_MANIFEST}"
    elif [[ "${DRY_RUN}" -eq 0 ]]; then
        { echo "seq,filename,width,height,n_frames"; } > "${UVG_MANIFEST}"
    fi
    for seq in ${UVG_SEQS}; do
        local prefix="${UVG_PREFIX[$seq]:-$seq}"
        local name="${seq}_1920x1080_120fps_420_8bit_YUV_RAW"
        local raw7z="${RAW_DIR}/uvg/${name}.7z"
        local yuv="${RAW_DIR}/uvg/${prefix}_1920x1080_120fps_420_8bit_YUV.yuv"
        local out="${UVG_DIR}/frames/${seq}"
        mkdir -p "${out}"
        if [[ -z "$(ls "${out}"/frame_*.png 2>/dev/null)" && "${DRY_RUN}" -eq 0 ]]; then
            if [[ ! -f "${raw7z}" ]]; then
                curl -fL --retry 3 -sS -o "${raw7z}" "${UVG_BASE}/${name}.7z"
            fi
            if ! 7z x -y -o"${RAW_DIR}/uvg" "${raw7z}" >/dev/null; then
                log_fail "FAIL 7z ${seq}"; mismatches=$((mismatches + 1)); continue
            fi
            if [[ ! -f "${yuv}" ]]; then
                yuv="$(ls "${RAW_DIR}/uvg"/*.yuv 2>/dev/null | grep -i "${prefix}" | head -1)"
            fi
            if [[ -z "${yuv}" || ! -f "${yuv}" ]]; then
                log_fail "FAIL yuv ${seq}"; mismatches=$((mismatches + 1)); continue
            fi
            if ! "${SYSTEM_FFMPEG}" -y -f rawvideo -pix_fmt yuv420p \
                    -s 1920x1080 -r 120 -i "${yuv}" -start_number 1 \
                    "${out}/frame_%04d.png"; then
                log_fail "FAIL decode ${seq}"; mismatches=$((mismatches + 1)); continue
            fi
        fi
        if [[ "${DRY_RUN}" -eq 0 ]]; then
            local nframes
            nframes="$(ls "${out}"/frame_*.png | wc -l)"
            if [[ "${validate}" -eq 1 ]]; then
                if [[ "${EXP_N[$seq]:-}" != "${nframes}" ]]; then
                    echo "VALIDATE uvg ${seq}: got 1920x1080 ${nframes}f, expected ${EXP_N[$seq]:-?}f" >&2
                    mismatches=$((mismatches + 1))
                fi
            else
                printf '%s,%s,1920,1080,%s\n' \
                    "${seq}" "${yuv##*/}" "${nframes}" >> "${UVG_MANIFEST}"
            fi
        fi
    done
    if [[ "${validate}" -eq 1 ]]; then
        echo "uvg validation: ${mismatches} sequence mismatches."
    fi
}

validate() {
    echo "---- validation ----"
    if [[ "${SKIP_KINETICS}" -ne 1 && -f "${KINETICS_MANIFEST}" ]]; then
        local ntrain ntest
        ntrain="$(awk -F, '$1=="train"{c++}END{print c+0}' "${KINETICS_MANIFEST}")"
        ntest="$(awk -F, '$1=="test"{c++}END{print c+0}' "${KINETICS_MANIFEST}")"
        echo "Kinetics manifest: ${ntrain} train / ${ntest} test clips."
    fi
    if [[ "${SKIP_DERF}" -ne 1 ]]; then
        for seq in ${DERF_SEQS}; do
            local n
            n="$(ls "${DERF_DIR}/frames/${seq}"/frame_*.png 2>/dev/null | wc -l)"
            echo "derf ${seq}: ${n} frames"
        done
    fi
    if [[ "${SKIP_UVG}" -ne 1 ]]; then
        for seq in ${UVG_SEQS}; do
            local n
            n="$(ls "${UVG_DIR}/frames/${seq}"/frame_*.png 2>/dev/null | wc -l)"
            echo "uvg ${seq}: ${n} frames"
        done
    fi
    echo "Download log: ${DOWNLOAD_LOG}"
}

setup_dirs
run_kinetics
download_derf
download_uvg
validate

echo "01_download_datasets.sh done."
