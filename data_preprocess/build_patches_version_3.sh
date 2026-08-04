#!/bin/bash
#SBATCH --job-name=gedi_patches
#SBATCH --partition=peregrine-cpu
#SBATCH --qos=cpu_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=16G
#SBATCH --time=239:50:00
#SBATCH --output=out_and_err/build_patches_version_3_%j.out
#SBATCH --error=out_and_err/build_patches_version_3_%j.err


# IMPORTANT for multiprocessing: keep BLAS/MKL single-threaded per process
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Limit GDAL tile cache per process — must be set before Python/rasterio starts
export GDAL_CACHEMAX=256
export GDAL_NUM_THREADS=1

srun python build_patches_version_3.py