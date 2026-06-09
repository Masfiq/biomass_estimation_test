#!/bin/bash
#SBATCH --job-name=biomass_attn
#SBATCH --partition=peregrine-gpu
#SBATCH --qos=gpu_medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100_3g.40gb:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=71:30:00
#SBATCH --output=out_and_err/resnet18_version2_%j.out
#SBATCH --error=out_and_err/resnet18_version2_%j.err


# avoid CPU oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# copy Köppen to local scratch (you already use KOPPEN_TIF env var :contentReference[oaicite:8]{index=8})
SCRATCH_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp/${USER}/slurm-${SLURM_JOB_ID}}}"
mkdir -p "${SCRATCH_DIR}"
cp "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/1991_2020/koppen_geiger_0p00833333.tif" "${SCRATCH_DIR}/koppen.tif"
export KOPPEN_TIF="${SCRATCH_DIR}/koppen.tif"

nvidia-smi
srun python -u resnet18_version_3_train.py