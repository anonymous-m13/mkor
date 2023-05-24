import data_loader.data_loader as data_loader
import torch
from utils.model_utils import get_model
import argparse
import os
from utils.plot import plot_all_optimizers, compare_optimizers, plot_eigenvalues, plot_scalability
from utils.timing import Timer
from utils.comm import get_comm_backend
from training.trainer import Trainer

num_classes_dict = {
    'cifar10': 10,
    'cifar100': 100,
    'imagenet': 1000,
    'imdb': 2,
    'mnli': 3,
    'mnli-mm': 3,
    'stsb': 2,
    'cola': 2,
    'mrpc': 2,
    'qnli': 2,
    'qqp': 2,
    'rte': 2,
    'sst2': 2,
    'stsb': 2,
    'wnli': 2
}

img_size_dict = {
    'mnist': 28,
    'cifar10': 32,
    'cifar100': 32,
    'imagenet': 224
}


def get_depth(model_name):
    if model_name.startswith('resnet'):
        return int(model_name[6:])
    else:
        return 1


def get_dataset(model_name):
    if model_name == "resnet20":
        return "cifar10"
    elif model_name == "resnet50":
        return "cifar100"
    elif model_name == "alexnet":
        return "cifar100"
    elif model_name.startswith("lra"):
        return model_name[4:]
    elif model_name == 't2t_vit_24':
        return "cifar100"
    elif model_name.startswith("t2t_vit"):
        return "cifar100"
    else:
        return "cifar100"


def get_model_name(model_name):
    if model_name.startswith("resnet"):
        return "resnet"
    elif model_name.startswith("densenet"):
        return "densenet"
    elif model_name.startswith("lra"):
        return "lra"
    else:
        return model_name


parser = argparse.ArgumentParser()

# Training Arguments
parser.add_argument('--model_name', type=str, default='lra_listops', help='Model Name')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--num_iters', type=int, default=None, help='Number of iterations for training - overrides epochs')
parser.add_argument('--method', type=str, default='accelerate', help='Prune, Fine Tune, Accelerate, or Dense Training')
parser.add_argument('--dataset', type=str, default='default', help='Dataset name')
parser.add_argument('--backend', type=str, default='nccl')
parser.add_argument('--url', type=str, default='env://')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seq_len', type=int, default=64)
parser.add_argument('--lr_scheduler', type=str, default='knee_point')
parser.add_argument('--test_set_only', action="store_true")
parser.add_argument('--lr_scheduler_wait', type=int, default=10,
                    help='The number of epochs with no improvement to wait before reducing the learning rate')
parser.add_argument('--checkpoint_dir', type=str, default=None, help='Checkpoint directory')
parser.add_argument('--training_checkpoint_dir', type=str, default=None, help='Training checkpoint directory')
parser.add_argument('--dataset_path', type=str, default="./data", help='The path to the dataset')
parser.add_argument('--stochastic_depth', type=float, default=0.1, help='Stochastic depth')
parser.add_argument('--schedule_by_test', action="store_true", help="Use test accuracy for scheduler")
parser.add_argument('--disable_mixed_precision', action="store_true", help='Whether or not use mixed precision')

# Optimizer Arguments
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
parser.add_argument('--inv_freq', type=int, default=100, help='Frequency of Fisher Information Matrix Inversion')
parser.add_argument('--num_lr_steps', type=int, default=3, help='Number of learning rate steps')
parser.add_argument('--warmup_proportion', type=float, default=0.01, help='Proportion of warmup steps')
parser.add_argument('--optimizer', type=str, default="sgd", help='Optimizer name')
parser.add_argument('--grad_accum_iter', type=int, default=1)

# Plotting Arguments
parser.add_argument('--plot', action="store_true")
parser.add_argument('--plot_time', action="store_true")
parser.add_argument('--plot_all', action="store_true")
parser.add_argument('--plot_eigenvalues', action="store_true")
parser.add_argument('--time', action="store_true")
parser.add_argument('--compare_optimizers', action="store_true")
parser.add_argument('--plot_samples_per_epoch', type=int, default=5, help='Number of samples per epoch for plotting')
parser.add_argument('--include_warmup', action="store_true")
parser.add_argument('--plot_scalability', action="store_true")
parser.add_argument('--checkpoint_saving_interval_epochs', type=int, default=1,
                    help='Number of epochs between saving checkpoints')
parser.add_argument('--checkpoint_saving_interval_iters', type=int, default=None,
                    help='Number of iterations between saving checkpoints - '
                         'if set overwrites checkpoint_saving_interval_epochs')
parser.add_argument('--global_batch_size', type=int, default=None, help='Global batch size for distributed training')
parser.add_argument('--target_accuracy', type=float, default=None, help='Target accuracy for training')

# Learning Rate Scheduler Arguments
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')


# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.0)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')



args = parser.parse_args()

model_name = args.model_name
depth = get_depth(model_name)
dataset = get_dataset(model_name) if args.dataset == "default" else args.dataset
num_classes = num_classes_dict[dataset] if dataset in num_classes_dict else 0
img_size = img_size_dict[dataset] if dataset in img_size_dict else 0
growth_rate = 12
compression_rate = 2
widen_factor = 1
drop_rate = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = args.epochs
criterion = torch.nn.CrossEntropyLoss() if model_name != "linear" else torch.nn.MSELoss()
batch_size = args.batch_size
timer = Timer(args.time)
sigmoid_multiplier = 10


def print_info(string, rank):
    if rank == 0:
        print(string)


def init():
    if not torch.cuda.is_available():
        print('Error: CUDA is not available.')
        raise RuntimeError

    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    world_rank = int(os.environ['RANK'])

    print(f'Starting process {world_rank} of {world_size} on {local_rank}.')
    torch.distributed.init_process_group(backend=args.backend, init_method=args.url, world_size=world_size,
                                         rank=world_rank)
    print(f'Process {world_rank} of {world_size} on {local_rank} started.')
    backend = get_comm_backend()
    torch.cuda.set_device(local_rank)
    torch.cuda.manual_seed(args.seed)
    train_sampler, train_loader, test_sampler, test_loader, vocab, tokenizer = data_loader.get_dataloader(args,
                                                                                                          world_size,
                                                                                                          world_rank)

    model = get_model(get_model_name(model_name),
                      depth=depth,
                      num_classes=num_classes,
                      growthRate=growth_rate,
                      compressionRate=compression_rate,
                      widen_factor=widen_factor,
                      dropRate=drop_rate,
                      task=dataset,
                      img_size=img_size,
                      vocab=vocab,
                      drop_path_rate=args.stochastic_depth).to(device)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    if args.checkpoint_dir is not None:
        def fix_bert_embeddings(checkpoint):
            if "bert.embeddings.word_embeddings.weight" in checkpoint:
                checkpoint["bert.embeddings.word_embeddings.weight"] = checkpoint["bert.embeddings.word_embeddings.weight"][:30522, :]
                checkpoint["cls.predictions.decoder.weight"] = checkpoint["cls.predictions.decoder.weight"][
                                                           :30522, :]
                checkpoint["cls.predictions.bias"] = checkpoint["cls.predictions.bias"][:30522]

        state_dict = torch.load(args.checkpoint_dir)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        # fix_bert_embeddings(state_dict)
        final_state_dict = {}
        for key, val in state_dict.items():
            if key.startswith('module.'):
                final_state_dict[key[7:]] = state_dict[key]
                key = key[7:]
            else:
                final_state_dict[key] = state_dict[key]
        model.load_state_dict(final_state_dict)


    # model_checkpoints = {'alexnet': './checkpoints/dense/sgd_cifar100_alexnet50_best.t7', 'resnet20': "./checkpoints/dense/sgd_cifar10_resnet20_best.t7", "resnet50": './checkpoints/dense/kfac_cifar100_resnet50_best.t7', 'lra': './checkpoints/dense/lra_listops_listops_training_acc36.22_test_acc37.17.pth'}

    # if model_name in model_checkpoints:
    #     state_dict = torch.load(model_checkpoints[model_name])
    #     model.load_state_dict(state_dict['net'] if 'net' in state_dict else state_dict)

    verbose = world_rank == 0
    args.plot = args.plot and verbose
    args.plot_time = args.plot_time and verbose
    args.plot_all = args.plot_all and verbose
    args.time = args.time and verbose

    if verbose: print("Number of Parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    torch.distributed.barrier()

    trainer = Trainer(args, model, backend, verbose, train_loader, train_sampler, test_loader, test_sampler,
                      device, timer, tokenizer)
    return trainer


if args.plot_eigenvalues:
    try:
        world_rank = int(os.environ['RANK'])
    except:
        world_rank = 0
    if world_rank == 0:
        plot_eigenvalues()
elif args.plot_scalability:
    try:
        world_rank = int(os.environ['RANK'])
    except:
        world_rank = 0
    if world_rank == 0:
        plot_scalability(args)
elif args.plot_all:
    try:
        world_rank = int(os.environ['RANK'])
    except:
        world_rank = 0
    if world_rank == 0:
        plot_all_optimizers(args.method, args.model_name, dataset, target_accuracy=args.target_accuracy)  # , min_train_acc=95)
elif args.compare_optimizers:
    try:
        world_rank = int(os.environ['RANK'])
    except:
        world_rank = 0
    if world_rank == 0:
        compare_optimizers(args.method, args.model_name, dataset)
else:
    trainer = init()
    trainer.train()

    # if model_name.startswith("lra"):
    #     if args.method == "full":
    #         method_list = ["dense", "prune"]
    #     else:
    #         method_list = [args.method]
    #     for method in method_list:
    #         train_lra(model, train_loader, test_loader, lr=args.lr, weight_decay=args.weight_decay, epochs=1, method=method, sigmoid_multiplier=20, num_steps=10000, eval_freq=400, plot=args.plot)#, checkpoint_name="lra_listops_sparsity94.68_training_acc37.41_test_acc36.83.pth")#"lra_listops_listops_training_acc36.22_test_acc37.17.pth")
    #     # train_lra(model, train_loader, test_loader, lr=args.lr, weight_decay=0, epochs=1, method=args.method, sigmoid_multiplier=20, num_steps=10000, eval_freq=400, checkpoint_name="lra_sparsity98.52_training_acc38.31_test_acc37.33.pth") #"lra_text_text_training_acc56.62_test_acc56.65.pth")
    #     # train_lra(model, train_loader, test_loader, lr=args.lr, weight_decay=args.weight_decay, epochs=1, method="prune", sigmoid_multiplier=20, num_steps=10000)#, checkpoint_name="lra_sparsity0.0_training_acc0.32_test_acc0.37.pth")
    # else:
    #     train(model, train_loader, train_sampler, test_loader, test_sampler, world_size, world_rank, backend, plot=args.plot, method=args.method, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs, sigmoid_multiplier=20, automatic_weight_decay=True, checkpoint_name="resnet50_sparsity72.96_training_acc62.21_test_acc58.01.pth")#, num_iters=200)

    # train(model, train_loader, test_loader, plot=True, method="prune", lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs, sigmoid_multiplier=20)
    # model.prune(method="magnitude", sigmoid_multiplier=1000, sparsity_ratio=80)
    # print(compute_sparsity_ratio(model, sigmoid_multiplier, 0.9))
    # model.load_state_dict(torch.load(f'./checkpoints/pruning/resnet50_sparsity71.61_training_acc73.53_test_acc59.27.pth'))
    # model.prune()
    # if model_name == "alexnet":
    #     checkpoint_dir = "./checkpoints/pruning/alexnet_sparsity88.75_training_acc42.1_test_acc39.27.pth"
    # if model_name == "resnet20":
    #     checkpoint_dir = "./checkpoints/pruning/resnet20_sparsity87.04_training_acc87.85_test_acc87.03.pth"
    # if model_name == "resnet50":
    #     checkpoint_dir = "./checkpoints/pruning/resnet50_sparsity72.96_training_acc62.21_test_acc58.01.pth"
    # model.load_state_dict(torch.load(checkpoint_dir))
    # thresholds = {"resnet20": 0.306640625, "resnet50": 0.404296875, "alexnet": 0.724609375}
    # compute_sparsity_ratio(model, sigmoid_multiplier, model_name, plot_histogram=True, threshold=thresholds[model_name], plot_layer_ratio=True)
    # train(model, train_loader, test_loader, plot=False, method="fine_tune", lr=1e-4, weight_decay=0, epochs=args.epochs, sigmoid_multiplier=20)

    # test(model, test_loader)
    # for checkpoint_name in os.listdir("./checkpoints/"):
    #     if checkpoint_name.startswith("best_" + model_name):
    #         print(checkpoint_name)
    #         model.load_state_dict(torch.load(f"./checkpoints/{checkpoint_name}"))
    # test(model, test_loader)
    # model.fix_masks()
    # test(model, test_loader)
