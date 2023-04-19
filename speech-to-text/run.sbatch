#!/bin/bash
#SBATCH --job-name=sunbird
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --mem=120Gb
#SBATCH --time=3-18:00
#SBATCH --gres=gpu:1
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

module load anaconda/3
conda activate speech-env
export HF_DATASETS_CACHE="/home/mila/a/akeraben/scratch/akera/cache"
python whisper_lg_finetune.py
