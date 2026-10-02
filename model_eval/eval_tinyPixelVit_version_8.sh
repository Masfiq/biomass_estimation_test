#!/bin/bash
#SBATCH --job-name=eval_tinyPixelVit_v8
#SBATCH --account=standard
#SBATCH --partition=peregrine-gpu
#SBATCH --qos=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=23:30:00
#SBATCH --output=out_and_err/eval_tinyPixelVit_version_8_%j.out
#SBATCH --error=out_and_err/eval_tinyPixelVit_version_8_%j.err

# cpus-per-task=6 (was 10 in earlier evals): the eval's DataLoader uses
# min(8, SLURM_CPUS_PER_TASK) workers, and 6 keeps more of the per-user CPU cap free
# for training jobs. Eval is a single pass over the validation split, so this costs
# little.


############################ edit variable here
PYTHON_FILE="eval_tinyPixelVit_version_8.py"

CSV_PATH="/s/chopin/e/proj/hyperspec/masfiq/csv_files/gedi_california_north_10_2021_AprilToAugust_version_10.csv"

# tinyPixelViT version_8: log10/3 SAR + NLCD patch fractions, version_10 clean patches, no winsor.
# The _best checkpoint is the epoch with the lowest validation Huber.
CKPT_PATH="/s/chopin/e/proj/hyperspec/masfiq/models/tinyPixelViT_version_8_fusion_geohash_month_koppen_SARlog_DEM_sqrt_NLCDfrac_delta2p5_version_10_California_North_10_2021_best.pth"

# Must match tinyPixelVit_version_8_train.py exactly or the val split will not be the
# same 20% the model never saw.
VAL_FRAC=0.2
SEED=42
BATCH_SIZE=256

# Also writes tinyPixelViT_version_8_eval_version_10_professor_landcover.csv next to this,
# with the land-cover sensitivity breakdown (NLCD 81, 82, 41, 42, 43, 41+42+43, 81+82).
OUT_PATH="/s/chopin/e/proj/hyperspec/masfiq/models/json/tinyPixelViT_version_8_eval_version_10.json"

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

# Per-JOB scratch folder. The old form fell back to plain $TMPDIR (= /tmp), so every
# job on the node shared /tmp/koppen.tif; a second job's cp overwrote it mid-read and
# GDAL logged "IReadBlock failed ... koppen.tif" (seen in 29 old logs).
SCRATCH_DIR="${TMPDIR:-/tmp}/${USER}_slurm_${SLURM_JOB_ID}"
mkdir -p "${SCRATCH_DIR}"
cp "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/1991_2020/koppen_geiger_0p00833333.tif" "${SCRATCH_DIR}/koppen.tif"
export KOPPEN_TIF="${SCRATCH_DIR}/koppen.tif"

nvidia-smi

srun python -u "${PYTHON_FILE}" --csv "${CSV_PATH}" --ckpt "${CKPT_PATH}" --val-frac "${VAL_FRAC}" --seed "${SEED}" --batch-size "${BATCH_SIZE}" --out "${OUT_PATH}"
