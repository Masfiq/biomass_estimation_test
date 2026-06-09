#!/bin/bash
#SBATCH --job-name=hls_download
#SBATCH --partition=peregrine-cpu
#SBATCH --qos=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=hls_%j.out
#SBATCH --error=hls_%j.err



# Prevent hidden thread oversubscription inside libraries
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

srun python hls_download.py