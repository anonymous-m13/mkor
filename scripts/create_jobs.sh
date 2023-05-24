EPOCHS=50
for OPTIMIZER in sgd hylo hylo_kis kfac kfac_approx_average 
do
        for MODEL in resnet50
        do
                for LR in 1e-2 1e-3 1e-4
                do
                        for WEIGHT_DECAY in 1e-3 1e-4
                        do
                            sbatch pruning_job.sh $MODEL $LR $EPOCHS $WEIGHT_DECAY $OPTIMIZER
                        done
                done
        done
done