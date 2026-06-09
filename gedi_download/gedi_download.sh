#!/bin/bash
#SBATCH --job-name="gedi_l4a_download"
#SBATCH --partition=peregrine-cpu
#SBATCH --qos=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=out_and_err/gedi_l4a_%j.out
#SBATCH --error=out_and_err/gedi_l4a_%j.err


srun python gedi_download.py
