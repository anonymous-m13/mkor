#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:0

module load anaconda3
source activate torch

MODEL=$1
LR=$2
EPOCHS=$3
WEIGHT_DECAY=$4
OPTIMIZER=$5
DATASET=$6

if [ -z "$MODEL" ]
then
    MODEL="resnet50"
fi

if [ -z "$LR" ]
then
    LR="1e-3"
fi

if [ -z "$EPOCHS" ]
then
    EPOCHS="50"
fi

if [ -z "$WEIGHT_DECAY" ]
then
    WEIGHT_DECAY="1e-2"
fi

if [ -z "$OPTIMIZER" ]
then
    OPTIMIZER="sgd"
fi

if [ -z "$DATASET" ]
then
    DATASET=""
else
    DATASET="--dataset $DATASET"
fi


cd ../
echo python pruning.py --model $MODEL $DATASET --lr $LR --method dense --epochs $EPOCHS --weight_decay $WEIGHT_DECAY --plot --time --plot_time --optimizer $OPTIMIZER --inv_freq 75
python pruning.py --model $MODEL $DATASET --lr $LR --method dense --epochs $EPOCHS --weight_decay $WEIGHT_DECAY --plot --time --plot_time --optimizer $OPTIMIZER --inv_freq 75
#python pruning.py --model resnet50 --method dense --lr 1e-2 --weight_decay 0 --plot --time --plot_time --optimizer kfac_approx_average

source deactivate