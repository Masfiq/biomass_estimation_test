#!/bin/bash
#SBATCH --job-name=biomass_train
#SBATCH --partition=peregrine-gpu
#SBATCH --qos=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Falcon limits on peregrine-gpu: up to 2 GPUs and 20 CPUs per job :contentReference[oaicite:2]{index=2}
# Falcon note: peregrine CPU:GPU ratio is 6:1 :contentReference[oaicite:3]{index=3}
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=23:00:00

# Pick ONE of these (GPU type must be specified) :contentReference[oaicite:4]{index=4}
#SBATCH --gres=gpu:nvidia_a100_3g.40gb:1
##SBATCH --gres=gpu:a100-sxm4-80gb:1

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err



# Good defaults: avoid CPU thread oversubscription while GPU trains
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cp /s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/1991_2020/koppen_geiger_0p00833333.tif "{$SLURM_TMPDIR}/koppen.tif"
export KOPPEN_TIF=$SLURM_TMPDIR/koppen.tif

# Run training
#srun python -u resnet18_version_2_train.py


srun python -u resnet18_version_3_train.py