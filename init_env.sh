salloc --gres=gpu:1 -c 4 --mem=120G -t 2:00:00
module add anaconda/3
conda activate speech-env
