EPOCHS=5
BATCH_SIZE=64
GRAD_ACCUM_ITER=1
LR_SCHEDULER=knee_point
LR_SCHEDULER_WAIT=1
SERVER=tacc
JOB_FILE_NAME=training_job_$SERVER.sh
TIME=0:30:00


MODEL=resnet164
DATASET=cifar100
LR=1e-5
WEIGHT_DECAY=0
INV_FREQ=10
OPTIMIZER=mkor






for NUM_NODES in 1 3 9 16
  do
    if [ $SERVER == "cedar" ]
  then
      NUM_PROC_PER_NODE=4
      sed -i 's/#SBATCH --gpus-per-node=.*/#SBATCH --gpus-per-node=v100l:'$NUM_PROC_PER_NODE'/g' $JOB_FILE_NAME
      sed -i 's/#SBATCH --nodes.*/#SBATCH --nodes '$NUM_NODES'/g' $JOB_FILE_NAME
      sed -i 's/#SBATCH -t.*/#SBATCH -t '$TIME'/g' $JOB_FILE_NAME
  fi

  if [ $SERVER == "narval" ]
  then
      NUM_PROC_PER_NODE=4
      sed -i 's/#SBATCH --gpus-per-node=.*/#SBATCH --gpus-per-node='$NUM_PROC_PER_NODE'/g' $JOB_FILE_NAME
      sed -i 's/#SBATCH --nodes.*/#SBATCH --nodes '$NUM_NODES'/g' $JOB_FILE_NAME
      sed -i 's/#SBATCH -t.*/#SBATCH -t '$TIME'/g' $JOB_FILE_NAME
  fi

  if [ $SERVER == "tacc" ]
  then
      if [ $NUM_NODES == 0 ]
      then
          NUM_NODES=1
          NUM_PROC_PER_NODE=1
      else
          NUM_PROC_PER_NODE=3
      fi
      PARTITION=gpu-a100
      sed -i 's/#SBATCH --gpus-per-node=.*/#SBATCH --gpus-per-node='$NUM_PROC_PER_NODE'/g' $JOB_FILE_NAME
      sed -i 's/#SBATCH -N.*/#SBATCH -N '$NUM_NODES'/g' $JOB_FILE_NAME
      sed -i 's/#SBATCH -t.*/#SBATCH -t '$TIME'/g' $JOB_FILE_NAME
      sed -i 's/#SBATCH -p.*/#SBATCH -p '$PARTITION'/g' $JOB_FILE_NAME
  fi

  if [ $SERVER == "mist" ]
  then
      NUM_PROC_PER_NODE=4
      sed -i 's/#SBATCH --gpus-per-node=.*/#SBATCH --gpus-per-node='$NUM_PROC_PER_NODE'/g' $JOB_FILE_NAME
      sed -i 's/#SBATCH -N.*/#SBATCH -N '$NUM_NODES'/g' $JOB_FILE_NAME
      sed -i 's/#SBATCH -t.*/#SBATCH -t '$TIME'/g' $JOB_FILE_NAME
  fi

  sbatch $JOB_FILE_NAME $MODEL $EPOCHS $OPTIMIZER $DATASET $NUM_NODES $NUM_PROC_PER_NODE $LR $WEIGHT_DECAY $BATCH_SIZE $INV_FREQ $GRAD_ACCUM_ITER $LR_SCHEDULER $LR_SCHEDULER_WAIT
done