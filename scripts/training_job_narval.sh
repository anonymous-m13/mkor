#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --nodes 1
#SBATCH --mem=20000M
#SBATCH -t 1:00:00

#module load python/3.10.2 gcc/9.3.0 arrow/7 cuda/11 scipy-stack
#source /home/mozaffar/scratch/pytorch/bin/activate

export OMP_NUM_THREADS=8

. ./parse_inputs.sh $@

cd ../


# Distributed system configuration
if [[ -z "${NODEFILE}" ]]; then
    if [[ -n "${SLURM_NODELIST}" ]]; then
        NODEFILE=/tmp/imagenet_slurm_nodelist
        scontrol show hostnames $SLURM_NODELIST > $NODEFILE
    elif [[ -n "${COBALT_NODEFILE}" ]]; then
        NODEFILE=$COBALT_NODEFILE
    fi
fi
if [[ -z "${NODEFILE}" ]]; then
    MAIN_RANK=$HOSTNAME
    RANKS=$HOSTNAME
    NNODES=1
else
    MAIN_RANK=$(head -n 1 $NODEFILE)
    RANKS=$(tr '\n' ' ' < $NODEFILE)
    NNODES=$(< $NODEFILE wc -l)
fi

CURRENT_DIR=$(pwd)

CMD="pruning.py $TRAINING_FLAGS"

RANK=0
for NODE in $RANKS; do
    LAUNCHER="torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --node_rank=$RANK --master_addr=$MAIN_RANK --master_port=1234"
    FULL_CMD="$LAUNCHER $CMD --url tcp://$MAIN_RANK:1234"
    if [[ $NODE == $MAIN_RANK ]]; then
        echo $FULL_CMD
	    eval $FULL_CMD &
    else
        echo "Launching rank $RANK on remote node $NODE"
	      ssh $NODE "cd $CURRENT_DIR; module load python/3.10.2 gcc/9.3.0 arrow/7 cuda/11 scipy-stack; source /home/mozaffar/scratch/pytorch/bin/activate; export OMP_NUM_THREADS=8; cd $PWD; $FULL_CMD" &
    fi
    RANK=$((RANK + 1))
done


wait