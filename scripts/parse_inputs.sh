NONE="NONE"

MODEL=$1
EPOCHS=$2
OPTIMIZER=$3
DATASET=$4
NNODES=$5
NPROC_PER_NODE=$6
LR=$7
WEIGHT_DECAY=$8
BATCH_SIZE=$9
INV_FREQ=${10}
GRAD_ACCUM_ITER=${11}
LR_SCHEDULER=${12}
LR_SCHEDULER_WAIT=${13}
NUM_ITERS=${14}
CHECKPOINT_DIR=${15}
DATASET_PATH=${16}
CHECKPOINT_SAVING_INTERVAL_EPOCHS=${17}
CHECKPOINT_SAVING_INTERVAL_ITERS=${18}
SCHEDULE_BY_TEST=${19}
WARMUP_EPOCHS=${20}


if [ $MODEL = $NONE ]
then
    MODEL=""
else
    MODEL="--model $MODEL"
fi

if [ "$LR" = $NONE ]
then
    LR=""
else
    LR="--lr $LR"
fi

if [ "$EPOCHS" = $NONE ]
then
    EPOCHS=""
else
    EPOCHS="--epochs $EPOCHS"
fi

if [ "$WEIGHT_DECAY" = $NONE ]
then
    WEIGHT_DECAY=""
else
    WEIGHT_DECAY="--weight_decay $WEIGHT_DECAY"
fi

if [ "$OPTIMIZER" = $NONE ]
then
    OPTIMIZER=""
else
    OPTIMIZER="--optimizer $OPTIMIZER"
fi

if [ "$DATASET" = $NONE ]
then
    export DATASET=""
else
    export DATASET="--dataset $DATASET"
fi

if [ "$BATCH_SIZE" = $NONE ]
then
    BATCH_SIZE=""
else
    BATCH_SIZE="--batch_size $BATCH_SIZE"
fi

if [ "$INV_FREQ" = $NONE ]
then
    INV_FREQ=""
else
    INV_FREQ="--inv_freq $INV_FREQ"
fi

if [ "$GRAD_ACCUM_ITER" = $NONE ]
then
    GRAD_ACCUM_ITER=""
else
    GRAD_ACCUM_ITER="--grad_accum_iter $GRAD_ACCUM_ITER"
fi

if [ "$LR_SCHEDULER" = $NONE ]
then
    LR_SCHEDULER=""
else
    LR_SCHEDULER="--lr_scheduler $LR_SCHEDULER"
fi

if [ "$LR_SCHEDULER_WAIT" = $NONE ]
then
    LR_SCHEDULER_WAIT=""
else
    LR_SCHEDULER_WAIT="--lr_scheduler_wait $LR_SCHEDULER_WAIT"
fi

if [ "$NUM_ITERS" = $NONE ]
then
    NUM_ITERS=""
else
    NUM_ITERS="--num_iters $NUM_ITERS"
fi

if [ "$CHECKPOINT_DIR" = $NONE ]
then
    CHECKPOINT_DIR=""
else
    CHECKPOINT_DIR="--training_checkpoint_dir $CHECKPOINT_DIR"
fi

if [ "$DATASET_PATH" = $NONE ]
then
    DATASET_PATH=""
else
    DATASET_PATH="--dataset_path $DATASET_PATH"
fi

if [ "$CHECKPOINT_SAVING_INTERVAL_EPOCHS" = $NONE ]
then
    CHECKPOINT_SAVING_INTERVAL_EPOCHS=""
else
    CHECKPOINT_SAVING_INTERVAL_EPOCHS="--checkpoint_saving_interval_epochs $CHECKPOINT_SAVING_INTERVAL_EPOCHS"
fi

if [ "$CHECKPOINT_SAVING_INTERVAL_ITERS" = $NONE ]
then
    CHECKPOINT_SAVING_INTERVAL_ITERS=""
else
    CHECKPOINT_SAVING_INTERVAL_ITERS="--checkpoint_saving_interval_iters $CHECKPOINT_SAVING_INTERVAL_ITERS"
fi

if [ "$SCHEDULE_BY_TEST" = $NONE ]
then
    SCHEDULE_BY_TEST=""
else
    SCHEDULE_BY_TEST="--schedule_by_test"
fi

 if [ "$WARMUP_EPOCHS" = $NONE ]
 then
     WARMUP_EPOCHS=""
 else
     WARMUP_EPOCHS="--warmup-epochs $WARMUP_EPOCHS"
 fi

export TRAINING_FLAGS="$MODEL $EPOCHS $OPTIMIZER $DATASET $LR $WEIGHT_DECAY $BATCH_SIZE $INV_FREQ $GRAD_ACCUM_ITER $LR_SCHEDULER $LR_SCHEDULER_WAIT $NUM_ITERS $CHECKPOINT_DIR $DATASET_PATH $CHECKPOINT_SAVING_INTERVAL_EPOCHS $CHECKPOINT_SAVING_INTERVAL_ITERS $SCHEDULE_BY_TEST $WARMUP_EPOCHS --time --include_warmup"