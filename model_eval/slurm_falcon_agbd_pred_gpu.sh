#!/bin/bash
#SBATCH --job-name=biomass_infer
#SBATCH --partition=peregrine-gpu
#SBATCH --qos=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:nvidia_a100_3g.40gb:1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err


# Keep BLAS threads from fighting dataloader workers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- choose a writable scratch dir even if SLURM_TMPDIR is not set ---
SCRATCH_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp/${USER}/slurm-${SLURM_JOB_ID}}}"
mkdir -p "${SCRATCH_DIR}"
echo "Using SCRATCH_DIR=${SCRATCH_DIR}"

# Copy Köppen to local scratch
KOPPEN_SRC="/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/1991_2020/koppen_geiger_0p00833333.tif"
cp "${KOPPEN_SRC}" "${SCRATCH_DIR}/koppen.tif"
export KOPPEN_TIF="${SCRATCH_DIR}/koppen.tif"
ls -lh "${KOPPEN_TIF}"

nvidia-smi

# Run inference script
srun python -u resnet18_version_3_agbd_pred.py