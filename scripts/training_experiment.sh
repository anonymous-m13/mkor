# Set any variable you don't need to $NONE

NONE="NONE"

EPOCHS=55
NUM_ITERS=$NONE #200000 #Set to NONE if you want to use the EPOCHS variable
BATCH_SIZE=43
NUM_NODES=4
GRAD_ACCUM_ITER=1
LR_SCHEDULER=knee_point
LR_SCHEDULER_WAIT=1
SERVER=tacc
JOB_FILE_NAME=training_job_$SERVER.sh
TIME=40:00:00
CHECKPOINT_DIR=$NONE
DATASET_PATH="./data"
CHECKPOINT_SAVING_INTERVAL_EPOCHS=$NONE
CHECKPOINT_SAVING_INTERVAL_ITERS=$NONE #1000 #This overrides the checkpoint_saving_interval_epochs
SCHEDULE_BY_TEST=$NONE #"--schedule_by_test" #Set to $NONE if you want to use the training accuracy for the scheduler
WARMUP_EPOCHS=0


if [ $SERVER = "cedar" ]
then
    NUM_PROC_PER_NODE=4
    sed -i 's/#SBATCH --gpus-per-node=.*/#SBATCH --gpus-per-node=v100l:'$NUM_PROC_PER_NODE'/g' $JOB_FILE_NAME
    sed -i 's/#SBATCH --nodes.*/#SBATCH --nodes '$NUM_NODES'/g' $JOB_FILE_NAME
    sed -i 's/#SBATCH -t.*/#SBATCH -t '$TIME'/g' $JOB_FILE_NAME
fi

if [ $SERVER = "narval" ]
then
    NUM_PROC_PER_NODE=4
    sed -i 's/#SBATCH --gpus-per-node=.*/#SBATCH --gpus-per-node='$NUM_PROC_PER_NODE'/g' $JOB_FILE_NAME
    sed -i 's/#SBATCH --nodes.*/#SBATCH --nodes '$NUM_NODES'/g' $JOB_FILE_NAME
    sed -i 's/#SBATCH -t.*/#SBATCH -t '$TIME'/g' $JOB_FILE_NAME
fi

if [ $SERVER = "tacc" ]
then
    NUM_PROC_PER_NODE=3
    PARTITION=gpu-a100
    sed -i 's/#SBATCH --gpus-per-node=.*/#SBATCH --gpus-per-node='$NUM_PROC_PER_NODE'/g' $JOB_FILE_NAME
    sed -i 's/#SBATCH -N.*/#SBATCH -N '$NUM_NODES'/g' $JOB_FILE_NAME
    sed -i 's/#SBATCH -n.*/#SBATCH -n '$NUM_NODES'/g' $JOB_FILE_NAME
    sed -i 's/#SBATCH -t.*/#SBATCH -t '$TIME'/g' $JOB_FILE_NAME
    sed -i 's/#SBATCH -p.*/#SBATCH -p '$PARTITION'/g' $JOB_FILE_NAME
fi

if [ $SERVER = "mist" ]
then
    NUM_PROC_PER_NODE=4
    sed -i 's/#SBATCH --gpus-per-node=.*/#SBATCH --gpus-per-node='$NUM_PROC_PER_NODE'/g' $JOB_FILE_NAME
    sed -i 's/#SBATCH -N.*/#SBATCH -N '$NUM_NODES'/g' $JOB_FILE_NAME
    sed -i 's/#SBATCH -n.*/#SBATCH -n '$NUM_NODES'/g' $JOB_FILE_NAME
    sed -i 's/#SBATCH -t.*/#SBATCH -t '$TIME'/g' $JOB_FILE_NAME
fi

#SQUAD: SGD: 1e-3, R1KFAC: 1e-3, HyLo-KIS: 1e-3


for INV_FREQ in 10
do
    for DATASET in imagenet
    do
        for MODEL in resnet50
        do
            for LR in 1e-1
            do
                for WEIGHT_DECAY in 2.5e-5 #2.5e-4 2.5e-3
                do
                    for OPTIMIZER in mkor
                    do
                        sbatch $JOB_FILE_NAME $MODEL $EPOCHS $OPTIMIZER $DATASET $NUM_NODES $NUM_PROC_PER_NODE $LR $WEIGHT_DECAY $BATCH_SIZE $INV_FREQ $GRAD_ACCUM_ITER $LR_SCHEDULER $LR_SCHEDULER_WAIT $NUM_ITERS $CHECKPOINT_DIR $DATASET_PATH $CHECKPOINT_SAVING_INTERVAL_EPOCHS $CHECKPOINT_SAVING_INTERVAL_ITERS $SCHEDULE_BY_TEST $WARMUP_EPOCHS
                    done
                done
            done
        done
    done
done