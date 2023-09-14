# MKOR: Momentum-Enabled Kronecker-Factor-Based Optimizer Using Rank-1 Updates

We have tested MKOR on BERT-Large-Uncased Pretraining and ResNet-50 and have achieved up to 2.57x speedup in comparison to state-of-the-art baselines.

## BERT-Large-Uncased Pretraining


## ResNet-50 Training

We are using the exact implementation of [this](https://github.com/gpauloski/kfac-pytorch/tree/v0.3.1) repo, and have integrated MKOR as an external optimizer to it.
You can find the code in `resnet` folder. For getting the desired results, please use the hyperparameters mentioned in the paper. If any hyperparameter isn't mentioned in the paper, please use the default value from the original repo.

