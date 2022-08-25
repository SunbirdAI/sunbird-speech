#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --gpus-per-node=rtx8000:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --partition=long
#SBATCH --output=/home/mila/a/akeraben/scratch/akera/sunbird/sunbird-speech/output/job_output.txt
#SBATCH --error=/home/mila/a/akeraben/scratch/akera/sunbird/sunbird-speech/output/job_error.txt

module load anaconda/3
conda activate speech-env
python -m torch.distributed.launch --nproc_per_node=4 finetune.py finetuning_common_voice.json