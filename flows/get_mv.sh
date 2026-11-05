#!/usr/bin/env bash
#
# flows/get_mv.sh
#
# Extract motion-vector images for one clip of 1-indexed frames
# (frame_%04d.<type>). Called by scripts/02_motion_vectors.sh.
#
# Usage:
#   EXTRACT_MVS=/path/to/extract_mvs MOTIONVECTOR=/path/to/MotionVector \
#     bash get_mv.sh <indir> <outdir> <width> <height> [type]
#
# The extraction writes four images per target frame:
#   <outdir>/<frame>_{before,after}_flow_{x,y}_0001.jpg

set -euo pipefail

indir="$1"
outdir="$2"
width="$3"
height="$4"
type="${5:-png}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRACT_MVS="${EXTRACT_MVS:-${SCRIPT_DIR}/MV_extract/extract_mvs}"
MOTIONVECTOR="${MOTIONVECTOR:-${SCRIPT_DIR}/MV_extract/MV-code-release/MotionVector}"

mkdir -p "$outdir"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

for f in "${indir}"/*."${type}"; do
	[[ -f "$f" ]] || continue
	video="${f##*/}"
	name="${video%.*}"
	idx="${name##*_}"
	[[ "$idx" =~ ^[0-9]+$ ]] || continue
	idx=$((10#$idx))
	stem="${name%_*}"

	(( idx % 12 == 1 )) || continue

	last_idx=$((idx + 12))
	last_frame="${indir}/${stem}_$(printf '%04d' "$last_idx").${type}"
	[[ -f "$last_frame" ]] || continue

	for i in $(seq 1 11); do
		cur_idx=$((idx + i))
		case "$i" in
			1|4|7|10) prev_idx=$((cur_idx - 1)); next_idx=$((cur_idx + 2)) ;;
			2|5|8|11) prev_idx=$((cur_idx - 2)); next_idx=$((cur_idx + 1)) ;;
			3|9)      prev_idx=$((cur_idx - 3)); next_idx=$((cur_idx + 3)) ;;
			6)        prev_idx=$((cur_idx - 6)); next_idx=$((cur_idx + 6)) ;;
		esac

		prev_frame="${indir}/${stem}_$(printf '%04d' "$prev_idx").${type}"
		cur_frame="${indir}/${stem}_$(printf '%04d' "$cur_idx").${type}"
		next_frame="${indir}/${stem}_$(printf '%04d' "$next_idx").${type}"

		[[ -f "$prev_frame" && -f "$cur_frame" ]] || continue

		cur_name="${cur_frame##*/}"
		cur_name="${cur_name%.*}"

		cp "$prev_frame" "${tmp}/1.${type}"
		cp "$cur_frame"  "${tmp}/2.${type}"
		rm -f "${tmp}/3.${type}"

		ffmpeg -y -i "${tmp}/%01d.${type}" -c:v libx264 -g 2 -bf 0 \
			-b_strategy 0 -sc_threshold 0 "${tmp}/seq.mp4" \
			>/dev/null 2>&1 < /dev/null
		"$EXTRACT_MVS" "${tmp}/seq.mp4" > "${tmp}/seq.mvs0" 2>/dev/null
		"$MOTIONVECTOR" "${tmp}/seq.mvs0" \
			"${outdir}/${cur_name}_before_flow_x" \
			"${outdir}/${cur_name}_before_flow_y" "$width" "$height" \
			>/dev/null 2>&1

		cp "$next_frame" "${tmp}/1.${type}"
		ffmpeg -y -i "${tmp}/%01d.${type}" -c:v libx264 -g 2 -bf 0 \
			-b_strategy 0 -sc_threshold 0 "${tmp}/seq.mp4" \
			>/dev/null 2>&1 < /dev/null
		"$EXTRACT_MVS" "${tmp}/seq.mp4" > "${tmp}/seq.mvs0" 2>/dev/null
		"$MOTIONVECTOR" "${tmp}/seq.mvs0" \
			"${outdir}/${cur_name}_after_flow_x" \
			"${outdir}/${cur_name}_after_flow_y" "$width" "$height" \
			>/dev/null 2>&1
	done
done
