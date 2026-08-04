#!/bin/bash
#SBATCH --job-name=gedi_patches_v5
#SBATCH --partition=peregrine-cpu
#SBATCH --qos=cpu_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem-per-cpu=16G
#SBATCH --time=239:50:00
#SBATCH --output=out_and_err/build_patches_version_5_%j.out
#SBATCH --error=out_and_err/build_patches_version_5_%j.err


# IMPORTANT for multiprocessing: keep BLAS/MKL single-threaded per process
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export GDAL_CACHEMAX=256
export GDAL_NUM_THREADS=1

srun python build_patches_version_5.py
