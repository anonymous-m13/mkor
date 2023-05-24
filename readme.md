# MKOR: Momentum-enabled Kronecker-factor-based Optimizer with Rank-1 Updates

## Quickstart

### Loading WikiCorpus Dataset in TACC
Use the following command on the TACC servers to load the WikiCorpus dataset.

```angular2html
cp -r /scratch/projects/longhorn3/scratch/00946/zzhang/data/bert/bert_masked_wikicorpus_en ./data/wikicorpus
cp -r /scratch/projects/longhorn3/scratch/00946/zzhang/data/bert/google_pretrained_weights ./data/google_pretrained_weights
cp -r /scratch/projects/longhorn3/scratch/00946/zzhang/data/bert/squad ./data/squad
```


### Loading BERT Checkpoints for Downstream Tasks
Use the following command on the TACC servers to download the pretrained NVIDIA BERT checkpoint for downstream tasks.

```angular2html
cp /scratch/00946/zzhang/data/bert/lamb-ckpt/ckpt_8601.pt ./data/bert_large_uncased_nvidia.pt
```

### Submitting Jobs
For TACC servers, please download the singularity from <a href="https://drive.google.com/uc?id=10rYax2nuemPPqsYZkq_7ymBsxn9fzIBy&export=download"> here</a> and extract in in the `$SCRATCH` directory.

For training models on the servers, you have to modify the `./scripts/training_experiment.sh` and set the proper variables.

`EPOCHS`: The number of epochs for training process. `[Type: int]`

`NUM_ITERS`: The number of iterations for training process - if set, will override `EPOCHS`. `[Type: int]`

`BATCH_SIZE`: The mini-batch size for training process. `[Type: int]`

`NUM_NODES`: Number of nodes to request for running the job. `[Type: int]`

`GRAD_ACCUM_ITER`: The number of iterations for gradient accumulation. `[Type: int, Default: 1]`

`LR_SCHEDULER`: The learning rate scheduler. `[Default: 'knee_point', Options: 'cosine', 'linear', 'multistep', 'knee_point']`

`LR_SCHEDULER_WAIT`: The number of epochs to wait before reducing the learning rate in the 'knee_point' scheduler. `[Type: int, Default: 1]`

`SERVER`: The server name. `[Options: 'tacc', 'mist', 'narval', 'cedar']`

`TIME`: The requested time from the server. `[Format: HH:MM:SS]`

`CHECKPOINT_DIR`: The directory for saving the checkpoints, if not set, model will be randomly initialized. `[Type: str, Default: ""]`

For submitting multiple jobs at the same time, you can use the variables in the for loop in the same file.


`INV_FREQ`: The factor inversion frequency. Use 10 for MKOR, 100 for HyLo and KAISA, and any number for "SGD".

`DATASET`: The name of the dataset.

`MODEL`: The name of the model.

Default MODEL-DATASET Pairs:
    
* `MODEL=bert_large_uncased_nvidia, DATASET=wikicorpus_phase1` -> BERT-Large Pretraining Phase 1
* `MODEL=bert_large_uncased_nvidia, DATASET=wikicorpus_phase2` -> BERT-Large Pretraining Phase 2
* `MODEL=bert_large_uncased_classification_nvidia, DATASET=imdb`
* `MODEL=bert_large_uncased_question_answering_nvidia, DATASET=squad`
* `MODEL=t2t_vit_24, DATASET=imagenet`
* `MODEL=resnet_164, DATASET=cifar100`
* `MODEL=vit_base_patch16, DATASET=imagenet`
* `MODEL=vit_large_patch16, DATASET=imagenet`
* `MODEL=vit_huge_patch14, DATASET=imagenet`

`LR`: The learning rate.

`WEIGHT_DECAY`: The weight decay.

`OPTIMIZER`: The optimizer. `[Options: 'mkor', 'sgd', 'kaisa', 'hylo_kis']`

After setting all the parameters, for submitting the jobs on the server use the following commands:

```angular2html
cd scripts
sh training_experiment.sh
```

### Local Runs
For running jobs locally or in an interactive session, you can set the variables in the `./scripts/local_run.sh` file. 
Most of the variables in this file are similar to the ones in the `./scripts/training_experiment.sh` file. 
The differences are as follows:

`NPROC_PER_NODE`: The number of nodes used for current run.

`SERVER`: This variable should be set to `local`.

For running a job locally, you can use the following command:

```angular2html
#Only for TACC servers:
module load tacc-singularity
export OMP_NUM_THREADS=8
singularity shell --nv --bind $PWD:/home /scratch/09070/tg883700/torch.sif

cd scripts
sh local_run.sh
```

## ImageNet Download and Preprocessing
For downloading the ImageNet dataset, you can use the following command:

```angular2html
cd scripts
sh imagenet.sh
```

If you have imagenet already downloaded and proprocessed in a directory other than `./data/imagenet`, you can use the following command to create a symbolic link to that directory:

```angular2html
ln -s <PATH_TO_IMAGENET> ./data/imagenet
```

## Visualizatoin
For visualizing the training/test accuracy vs time/#epochs for different runs in a figure, you can use the following command, which generates a PDF
file in the `./figures/dense/MODEL/DATASET` directory:

```angular2html
python pruning.py --method dense --plot_all --model MODEL --dataset DATASET
```

## Scalability
For running the scalability experiments, first you have to set the variables in the `./scripts/strong_scalability.sh` file.
The variables are similar to the ones in the `./scripts/training_experiment.sh` file. The only difference is that you have
to define the number of nodes for the experiment in the `NUM_NODES` variable.

For running the experiment, you can use the following command:

```angular2html
cd scripts
sh strong_scalability.sh
```

For visualizing the results, you can use the following command, which generates a PDF file in the `./figures/scalability` directory:

```angular2html
#Only for TACC servers:
module load tacc-singularity
export OMP_NUM_THREADS=8
singularity shell --nv --bind $PWD:/home #SCRATCH/torch.sif

python pruning.py --method dense --plot_scalability --model MODEL --dataset DATASET
```