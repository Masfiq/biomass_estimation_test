#!/bin/bash
#SBATCH --job-name=eval_tinyUnet_v15
#SBATCH --account=standard
#SBATCH --partition=peregrine-gpu
#SBATCH --qos=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=23:30:00
#SBATCH --output=out_and_err/eval_tinyUnet_version_15_%j.out
#SBATCH --error=out_and_err/eval_tinyUnet_version_15_%j.err

# cpus-per-task=6 (was 10 in earlier evals): the eval's DataLoader uses
# min(8, SLURM_CPUS_PER_TASK) workers, and 6 keeps more of the per-user CPU cap free
# for training jobs. Eval is a single pass over the validation split, so this costs
# little.


############################ edit variable here
PYTHON_FILE="eval_tinyUnet_version_15.py"

CSV_PATH="/s/chopin/e/proj/hyperspec/masfiq/csv_files/gedi_california_north_10_2021_AprilToAugust_version_8.csv"

# version_15: log10/3 SAR + NLCD patch fractions + winsorised TRAINING labels.
# The _best checkpoint is the epoch with the lowest validation Huber.
CKPT_PATH="/s/chopin/e/proj/hyperspec/masfiq/models/tinyUnet_version_15_fusion_geohash_month_koppen_withAttentionLayer_SARlog_DEM_sqrt_NLCDfrac_delta2p5_winsor995_version_8_California_North_10_2021_best.pth"

# Must match tinyUnet_version_15_train.py exactly or the val split will not be the
# same 20% the model never saw.
VAL_FRAC=0.2
SEED=42
BATCH_SIZE=256

# Also writes tinyUnet_version_15_eval_version_8_professor_landcover.csv next to this,
# with the land-cover sensitivity breakdown (NLCD 81, 82, 41, 42, 43, 41+42+43, 81+82).
OUT_PATH="/s/chopin/e/proj/hyperspec/masfiq/models/json/tinyUnet_version_15_eval_version_8.json"

###################################

# The conda env ships a newer libstdc++ than /lib64 (libgdal needs GLIBCXX_3.4.30).
export LD_LIBRARY_PATH="/s/chopin/e/proj/hyperspec/masfiq/projenv/lib:${LD_LIBRARY_PATH}"

# Cap per-process GDAL block cache (MB) across the DataLoader workers.
export GDAL_CACHEMAX=256
export VSI_CACHE=FALSE

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRATCH_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp/${USER}/slurm-${SLURM_JOB_ID}}}"
mkdir -p "${SCRATCH_DIR}"
cp "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/1991_2020/koppen_geiger_0p00833333.tif" "${SCRATCH_DIR}/koppen.tif"
export KOPPEN_TIF="${SCRATCH_DIR}/koppen.tif"

nvidia-smi

srun python -u "${PYTHON_FILE}" --csv "${CSV_PATH}" --ckpt "${CKPT_PATH}" --val-frac "${VAL_FRAC}" --seed "${SEED}" --batch-size "${BATCH_SIZE}" --out "${OUT_PATH}"
