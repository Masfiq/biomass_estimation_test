#!/bin/bash
#SBATCH --job-name=biomass_attn_unet_v13
#SBATCH --account=standard
#SBATCH --partition=peregrine-gpu
#SBATCH --qos=gpu_medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100_3g.40gb:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=128G
#SBATCH --time=71:50:00
#SBATCH --output=out_and_err/tinyUnet_version_13_train_%j.out
#SBATCH --error=out_and_err/tinyUnet_version_13_train_%j.err



export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# The conda env ships a newer libstdc++ than /lib64; libgdal needs GLIBCXX_3.4.30
# and the system one only has 3.4.29. Without this, `import rasterio` fails
# depending on which shell the job was submitted from.
export LD_LIBRARY_PATH="/s/chopin/e/proj/hyperspec/masfiq/projenv/lib:${LD_LIBRARY_PATH}"

# Cap per-process GDAL block cache (MB). Without this GDAL sizes its cache at
# ~5% of the node's PHYSICAL RAM per process, which across the DataLoader
# workers blows past the cgroup limit and gets OOM-killed. The Copernicus DEM VRT
# (inherited from version_8) is a second raster source read per sample, and version_13
# patch tifs carry 28 bands instead of 11, so this cap matters more than ever.
export GDAL_CACHEMAX=256
export VSI_CACHE=FALSE
export DL_WORKERS=5


SCRATCH_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp/${USER}/slurm-${SLURM_JOB_ID}}}"
mkdir -p "${SCRATCH_DIR}"
cp "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/1991_2020/koppen_geiger_0p00833333.tif" "${SCRATCH_DIR}/koppen.tif"
export KOPPEN_TIF="${SCRATCH_DIR}/koppen.tif"

nvidia-smi
srun python -u tinyUnet_version_13_train.py
