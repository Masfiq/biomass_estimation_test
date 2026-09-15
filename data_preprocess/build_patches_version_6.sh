#!/bin/bash
#SBATCH --job-name=gedi_patches_v6
#SBATCH --partition=peregrine-cpu
#SBATCH --qos=cpu_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem-per-cpu=20G
#SBATCH --time=239:50:00
#SBATCH --output=out_and_err/build_patches_version_6_%j.out
#SBATCH --error=out_and_err/build_patches_version_6_%j.err


# The conda env ships a newer libstdc++ than /lib64; libgdal needs GLIBCXX_3.4.30
# and the system one only has 3.4.29. Without this, `import rasterio` fails
# depending on which shell/node the job lands on.
export LD_LIBRARY_PATH="/s/chopin/e/proj/hyperspec/masfiq/projenv/lib:${LD_LIBRARY_PATH}"

# IMPORTANT for multiprocessing: keep BLAS/MKL single-threaded per process
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export GDAL_CACHEMAX=256
export GDAL_NUM_THREADS=1

srun python build_patches_version_6.py
