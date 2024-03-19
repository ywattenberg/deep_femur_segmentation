#!/usr/bin/env bash
#SBATCH --mail-type=NONE
#SBATCH --job-name=ModelTraining
#SBATCH --time=08:00:00
#SBATCH --output=/home/%u/deep_femur_segmentation/logs/%j.out
#SBATCH --error=/home/%u/deep_femur_segmentation/logs/%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=V100:1
#SBATCH --constraint=GPUMEM32GB

# Exit on errors
set -o errexit

module load gpu python/3.11 cuda/12.3.0


# Set a directory for temporary files unique to the job with automatic removal at job termination
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
    echo 'Failed to create temp directory' >&2
    exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Change the current directory to the location where you want to store temporary files, exit if changing didn't succeed.
# Adapt this to your personal preference
cd "${TMPDIR}" || exit 1

# Send some noteworthy information to the output log
echo "Running on node:      $(hostname)"
echo "In directory:         $(pwd)"
echo "Starting on:          $(date)"
echo "SLURM_JOB_ID:         ${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST:   ${SLURM_JOB_NODELIST}"
echo "SLURM_NTASKS:         ${SLURM_NTASKS}"
echo "SLURM_CPUS_PER_TASK:  ${SLURM_CPUS_PER_TASK}"
echo "GPU:                  ${CUDA_VISIBLE_DEVICES}"

rsync -ah --stats /data/$USER/numpy $TMPDIR



/home/$USER/deep_femur_segmentation/.venv/bin/python3 /home/$USER/deep_femur_segmentation/scripts/cluster_seg_train.py --config /home/$USER/deep_femur_segmentation/config/segmentation_config.yaml --tmp_dir $TMPDIR

exit 0
