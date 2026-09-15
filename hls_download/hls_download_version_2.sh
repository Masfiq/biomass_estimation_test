#!/bin/bash
#SBATCH --job-name="HLS-MP"
#SBATCH --account=standard
#SBATCH --partition=peregrine-cpu
#SBATCH --qos=cpu_medium        
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18        
#SBATCH --mem-per-cpu=4G
#SBATCH --time=71:50:00
#SBATCH --output=out_and_err/hls_download_version_2_%j.out
#SBATCH --error=out_and_err/hls_download_version_2_%j.err

# Avoid oversubscription: each process should not spawn more threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# The conda env ships a newer libstdc++ than /lib64; libgdal needs GLIBCXX_3.4.30
# and the system one only has 3.4.29. Without this, `from osgeo import gdal` fails
# with "GLIBCXX_3.4.30 not found" / "No module named '_gdal'" depending on which
# node the job lands on. Same fix already present in the train/eval/build .sh files.
export LD_LIBRARY_PATH="/s/chopin/e/proj/hyperspec/masfiq/projenv/lib:${LD_LIBRARY_PATH}"

# Make python unbuffered so you can tail -f the output live
srun python -u hls_download_version_2.py