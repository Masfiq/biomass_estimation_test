#!/bin/bash
#SBATCH --job-name=biomass_attn
#SBATCH --partition=peregrine-gpu
#SBATCH --qos=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100_3g.40gb:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=23:30:00
#SBATCH --output=out_and_err/eval_resnet18_version_4_%j.out
#SBATCH --error=out_and_err/eval_resnet18_version_4_%j.err


############################ edit variable here
PYTHON_FILE="eval_resnet18_version_4.py"

CSV_PATH="/s/chopin/e/proj/hyperspec/masfiq/csv_files/gedi_California_california_north_10_2021_whole_year_metadata.csv"

CKPT_PATH="/s/chopin/e/proj/hyperspec/masfiq/models/resnet18_fusion_geohash_month_koppen_withAttentionLayer_California_North_10_2021.pth"



VAL_FRAC=0.2
SEED=42
BATCH_SIZE=256

OUT_PATH="/s/chopin/e/proj/hyperspec/masfiq/models/json/resnet18_fusion_geohash_month_koppen_withAttentionLayer_California_North_10_2021.json"

###################################

# Avoid CPU oversubscription

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1



# Koppen file to local scratch
SCRATCH_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp/${USER}/slurm-${SLURM_JOB_ID}}}"
mkdir -p "${SCRATCH_DIR}"

cp "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/1991_2020/koppen_geiger_0p00833333.tif" "${SCRATCH_DIR}/koppen.tif"

export KOPPEN_TIF="${SCRATCH_DIR}/koppen.tif"

# main command 

nvidia-smi

srun python -u "${PYTHON_FILE}" --csv "${CSV_PATH}" --ckpt "${CKPT_PATH}" --val-frac "${VAL_FRAC}" --seed "${SEED}" --batch-size "${BATCH_SIZE}" --out "${OUT_PATH}"

