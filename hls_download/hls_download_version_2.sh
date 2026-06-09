#!/bin/bash
#SBATCH --job-name="HLS-MP"
#SBATCH --partition=peregrine-cpu
#SBATCH --qos=cpu_medium        # 24 hours :contentReference[oaicite:9]{index=9}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18        # <= 20 cores limit on cpu_short :contentReference[oaicite:10]{index=10}
#SBATCH --mem-per-cpu=4G
#SBATCH --time=71:50:00
#SBATCH --output=out_and_err/hls_download_version_2_%j.out
#SBATCH --error=out_and_err/hls_download_version_2_%j.err

# Avoid oversubscription: each process should not spawn more threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Make python unbuffered so you can tail -f the output live
srun python -u hls_download_version_2.py