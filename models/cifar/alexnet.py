'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch.nn as nn
import pruning_layers.linear as pruning_linear
import pruning_layers.conv2d as pruning_conv2d


__all__ = ['alexnet']


class AlexNet(nn.Module):

    def __init__(self, num_classes=10, **kwargs):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 1 * 1, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, num_classes),
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def prune(self, sigmoid_multiplier=10.0, method="learnable", sparsity_ratio=None):
        self.pruned_layers = []
        # self.features[0] = pruning_conv2d.PruningConv2d(self.features[0], sigmoid_multiplier, method=method, sparsity_ratio=sparsity_ratio)
        # self.pruned_layers.append(self.features[0])
        # self.features[3] = pruning_conv2d.PruningConv2d(self.features[3], sigmoid_multiplier, method=method, sparsity_ratio=sparsity_ratio)
        # self.pruned_layers.append(self.features[3])
        # self.features[6] = pruning_conv2d.PruningConv2d(self.features[6], sigmoid_multiplier, method=method, sparsity_ratio=sparsity_ratio)
        # self.pruned_layers.append(self.features[6])
        # self.features[8] = pruning_conv2d.PruningConv2d(self.features[8], sigmoid_multiplier, method=method, sparsity_ratio=sparsity_ratio)
        # self.pruned_layers.append(self.features[8])
        # self.features[10] = pruning_conv2d.PruningConv2d(self.features[10], sigmoid_multiplier, method=method, sparsity_ratio=sparsity_ratio)
        # self.pruned_layers.append(self.features[10])
        self.classifier = pruning_linear.PruningLinear(self.classifier, sigmoid_multiplier, method=method, sparsity_ratio=sparsity_ratio)
        self.pruned_layers.append(self.classifier)

    def fix_masks(self, threshold=1e-2):
        for layer in self.pruned_layers:
            layer.fix_masks(threshold)


def alexnet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    return model
