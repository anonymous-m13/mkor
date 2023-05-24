EPOCHS=25
#NUM_ITERS=10405 #Comment out this line if you want to use the EPOCHS variable
NPROC_PER_NODE=1
BATCH_SIZE=256
GRAD_ACCUM_ITER=1
LR_SCHEDULER=knee_point
LR_SCHEDULER_WAIT=10
#SCHEDULE_BY_TEST="--schedule_by_test"
CHECKPOINT_SAVING_INTERVAL_EPOCHS="--checkpoint_saving_interval_epochs 20"
#CHECKPOINT_SAVING_INTERVAL_ITERS="--checkpoint_saving_interval_iters 50" #This overrides the checkpoint_saving_interval_epochs
#TRAINING_CHECKPOINT_DIR="--training_checkpoint_dir ./checkpoints/bert_base_bert_training_acc56.94_test_acc58.75.pth"
WARMUP_EPOCHS="--warmup-epochs 0"

SERVER=local
#CHECKPOINT_DIR="--checkpoint_dir ./checkpoints/dense/bert_base_question_answering/squad/bert_base_question_answering_squad_training_acc67.24_test_acc69.25.pth"
#CHECKPOINT_DIR="--checkpoint_dir /media/hdd/phase1.pt"


cd ../
#SQUAD:             SGD: 1e-3, MKOR: 1e-3, HyLo-KIS: 1e-3, KAISA:
#IMDB:              SGD: 1e-3, MKOR: 1e-3, HyLo-KIS: 1e-3, KAISA: 1e-3
#AlexNet-CIFAR10:   SGD: 1e-1, MKOR: 1e+1, HyLo-KIS: 1e-1, KAISA: 1e+1
#T2T-ViT24-CIFAR100:SGD: 1e-1, MKOR: 1e-0, HyLo-KIS: 1e-1
#T2T-ViT14-CIFAR100:SGD: 1e-1, MKOR: 1e-0, HyLo-KIS: 1e-1
#BERT:              SGD: 5e-3, MKOR: 5e-2, HyLo-KIS: 5e-2, KAISA: 1e-1

if [ $SERVER == tacc ]
then
  module load tacc-singularity

  export OMP_NUM_THREADS=8

  singularity shell --nv --bind $PWD:/home /scratch/09070/tg883700/torch.sif
fi

if [ -z $NUM_ITERS ]
then
    NUM_ITERS=""
else
    NUM_ITERS="--num_iters $NUM_ITERS"
fi

for INV_FREQ in 100 1000
do
    for DATASET in cifar100 #wikicorpus_phase2 #squad #bert_phase1
    do
        for MODEL in autoencoder #bert_large_uncased_nvidia #bert_large_uncased_question_answering_nvidia #bert_large_uncased_nvidia
        do
            for LR in 1e-1
            do
                for WEIGHT_DECAY in 0 #1e-4 #1e-3 2e-4 5e-4 7e-4 #1e-3 1e-4 1e-2 1e-1
                do
                    for OPTIMIZER in mkor
                    do
                        echo torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=1 pruning.py --model $MODEL --dataset $DATASET --lr $LR --method dense --epochs $EPOCHS --time --plot --plot_time --weight_decay $WEIGHT_DECAY --optimizer $OPTIMIZER --inv_freq $INV_FREQ --batch_size $BATCH_SIZE --grad_accum_iter $GRAD_ACCUM_ITER --lr_scheduler $LR_SCHEDULER --lr_scheduler_wait $LR_SCHEDULER_WAIT $CHECKPOINT_DIR $NUM_ITERS $SCHEDULE_BY_TEST $CHECKPOINT_SAVING_INTERVAL_EPOCHS $CHECKPOINT_SAVING_INTERVAL_ITERS $TRAINING_CHECKPOINT_DIR $WARMUP_EPOCHS --disable_mixed_precision
                        torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=1 pruning.py --model $MODEL --dataset $DATASET --lr $LR --method dense --epochs $EPOCHS --time --plot --plot_time --weight_decay $WEIGHT_DECAY --optimizer $OPTIMIZER --inv_freq $INV_FREQ --batch_size $BATCH_SIZE --grad_accum_iter $GRAD_ACCUM_ITER --lr_scheduler $LR_SCHEDULER --lr_scheduler_wait $LR_SCHEDULER_WAIT --num_lr_steps 3 $CHECKPOINT_DIR $NUM_ITERS $SCHEDULE_BY_TEST $CHECKPOINT_SAVING_INTERVAL_EPOCHS $CHECKPOINT_SAVING_INTERVAL_ITERS $TRAINING_CHECKPOINT_DIR $WARMUP_EPOCHS --disable_mixed_precision
                    done
                done
            done
        done
    done
done

#echo torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=1 pruning.py --model $MODEL --dataset $DATASET --lr $LR --method dense --epochs $EPOCHS --time --plot --plot_time --weight_decay 0 --optimizer $OPTIMIZER --inv_freq $INV_FREQ --batch_size $BATCH_SIZE --grad_accum_iter $GRAD_ACCUM_ITER --plot_all
#torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=1 pruning.py --model $MODEL --dataset $DATASET --lr $LR --method dense --epochs $EPOCHS --time --plot --plot_time --weight_decay 0 --optimizer $OPTIMIZER --inv_freq $INV_FREQ --batch_size $BATCH_SIZE --grad_accum_iter $GRAD_ACCUM_ITER --plot_all


#LR     1e1  1e0    1e-1  1-2
#SGD    N/A  N/A    108   145N/A
#MKOR   94   79     78     76
#HyLo   N/A  123N/A 98     150N/A
#KAISA  112  100    90     89N/A