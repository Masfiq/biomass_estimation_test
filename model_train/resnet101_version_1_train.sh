#!/bin/bash
#SBATCH --job-name=biomass_attn
#SBATCH --partition=kestrel-gpu
#SBATCH --qos=gpu_medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=71:50:00
#SBATCH --output=out_and_err/resnet101_version_1_%j.out
#SBATCH --error=out_and_err/resnet101_version_1_%j.err



export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


SCRATCH_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp/${USER}/slurm-${SLURM_JOB_ID}}}"
mkdir -p "${SCRATCH_DIR}"
cp "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/1991_2020/koppen_geiger_0p00833333.tif" "${SCRATCH_DIR}/koppen.tif"
export KOPPEN_TIF="${SCRATCH_DIR}/koppen.tif"

nvidia-smi
srun --cpus-per-task=8 python -u resnet101_version_1_train.py