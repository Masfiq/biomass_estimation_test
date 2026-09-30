#!/bin/bash
#SBATCH --job-name=gedi_patches_v10
#SBATCH --account=standard
#SBATCH --partition=peregrine-cpu
#SBATCH --qos=cpu_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem-per-cpu=20G
#SBATCH --time=239:50:00
#SBATCH --output=out_and_err/build_patches_version_10_%j.out
#SBATCH --error=out_and_err/build_patches_version_10_%j.err

# --account=standard is required: the default account is blank on this cluster and
# submission fails with "Invalid account or account/partition combination specified".



# The conda env ships a newer libstdc++ than /lib64; libgdal needs GLIBCXX_3.4.30 and
# the system one only has 3.4.29. Without this, `import rasterio` fails depending on
# which node the job lands on.
export LD_LIBRARY_PATH="/s/chopin/e/proj/hyperspec/masfiq/projenv/lib:${LD_LIBRARY_PATH}"

# One thread per process. With 17 worker processes, letting BLAS/MKL each spawn their
# own thread pool would oversubscribe the allocation several times over.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Cap the per-process GDAL block cache. Left alone, GDAL sizes it at ~5% of the node's
# PHYSICAL RAM per process, which across 17 workers blows past the cgroup limit.
# version_10 opens MORE rasters per shot than version_8 -- it keeps trying candidate
# granules until it finds one with a clean 5x5 window -- so this cap matters more here.
export GDAL_CACHEMAX=256
export GDAL_NUM_THREADS=1
export VSI_CACHE=FALSE

# Walltime note: version_8 needed more than one 239h window and was resumed via the
# existing-patch scan in build_gedi_hls_patches_multiprocess. version_10 does strictly
# more I/O per shot, so expect at least one resume. Relaunching the same script is safe
# and idempotent: it rescans PATCHES_ROOT and skips shots that already have a patch.

srun python -u build_patches_version_10.py
