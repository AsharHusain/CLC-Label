#!/bin/bash
# SLURM job for the new_video batch. Edit the paths/resources marked EDIT, then:
#
#   cd <repo>/images
#   # 1) once, on a login/compute node -- index the videos:
#   python pipeline/build_manifest.py /path/to/new_video data/manifest_new.csv
#   # 2) extract keypoints, 8 shards in parallel:
#   sbatch --array=0-7 pipeline/run_new_videos.sbatch
#   # 3) after ALL array tasks finish -- boundaries, ELAN drafts, CSV:
#   bash pipeline/finish_new_videos.sh
#
#SBATCH --job-name=mudra_kp
#SBATCH --cpus-per-task=4          # EDIT
#SBATCH --mem=8G                   # EDIT
#SBATCH --time=24:00:00            # EDIT
#SBATCH --output=logs/kp_%A_%a.out

set -euo pipefail
cd "$(dirname "$0")/.."            # -> images/
mkdir -p logs keypoints
# EDIT: activate an env with Python<=3.12, mediapipe==0.10.14, opencv-python-headless, numpy, pympi-ling
# source ~/venvs/mudra/bin/activate

python pipeline/extract_all.py --manifest data/manifest_new.csv --out-dir keypoints \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "${SLURM_ARRAY_TASK_COUNT:-1}"
