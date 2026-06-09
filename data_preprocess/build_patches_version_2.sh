#!/bin/bash
#SBATCH --job-name=gedi_patches
#SBATCH --partition=peregrine-cpu
#SBATCH --qos=cpu_medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=8G
#SBATCH --time=39:30:00
#SBATCH --output=out_and_err/build_patches_version_2_%j.out
#SBATCH --error=out_and_err/build_patches_version_2_%j.err


# IMPORTANT for multiprocessing: keep BLAS/MKL single-threaded per process
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

srun python build_patches_version_2.py