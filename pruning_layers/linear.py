import torch
import utils.pruning as pruning
import types


class PruningLinear(torch.nn.Module):
    def __init__(self, module, sigmoid_multiplier=10.0, method="learnable", sparsity_ratio=None, sparsity_pattern="uniform"):
        super(PruningLinear, self).__init__()
        self.module = module
        self.name = "Pruning" + module.__class__.__name__ 
        if method == "learnable": 
            self.mask_weights = torch.nn.Parameter(torch.ones_like(module.weight) * 2)
            self.mask_bias = torch.nn.Parameter(torch.ones_like(module.bias) * 2) if module.bias is not None else None
            if sparsity_pattern == "uniform":
                self.mask_centers_weight = torch.ones_like(module.weight)
                self.mask_centers_bias = torch.ones_like(module.bias) if module.bias is not None else None
            elif sparsity_pattern == "block_diagonal":
                self.mask_centers_weight = 1.8 * torch.ones_like(module.weight) - 1.3 * pruning.block_diagonal_mask(module.weight.shape, 4).to(module.weight.device)
                self.mask_centers_bias = torch.ones_like(module.bias) if module.bias is not None else None
        elif method == "magnitude":
            if sparsity_ratio is None:
                raise ValueError("sparsity_ratio must be specified for magnitude pruning")
            self.mask_weights = torch.nn.Parameter(pruning.prune_layer(module.weight.data.clone().detach(), sparsity_ratio))
            self.mask_weights.requires_grad = False
            self.mask_bias = torch.nn.Parameter(pruning.prune_layer(module.bias.data.clone().detach(), sparsity_ratio)) if module.bias is not None else None
            if self.mask_bias is not None:
                self.mask_bias.requires_grad = False
        self.module.weight.requires_grad = False
        if self.module.bias is not None:
            self.module.bias.requires_grad = False
        self.sigmoid_multiplier = sigmoid_multiplier
        self.freeze_masks = False

    def __call__(self, x):
        if self.freeze_masks:
            return torch.nn.functional.linear(x.view(x.size(0), -1), self.module.weight * self.frozen_weight_mask, self.module.bias * self.frozen_bias_mask if self.module.bias is not None else None)
        else:
            return torch.nn.functional.linear(x.view(x.size(0), -1), self.module.weight * torch.sigmoid((self.mask_weights - self.mask_centers_weight) * self.sigmoid_multiplier), self.module.bias * torch.sigmoid((self.mask_bias - self.mask_centers_bias) * self.sigmoid_multiplier) if self.module.bias is not None else None)

    def __repr__(self):
        return "Pruning " + self.module.__repr__()

    def __str__(self):
        return "Pruning " + self.module.__str__()

    def to(self, device):
        self.module.device = device
        return self

    # def parameters(self):
    #     self.module.parameters()

    def eval(self):
        self.module.eval()

    def fix_masks(self, threshold=1e-2):
        self.freeze_masks = True
        self.frozen_weight_mask = (torch.sigmoid((self.mask_weights - self.mask_centers_weight) * self.sigmoid_multiplier) > threshold) * torch.sigmoid((self.mask_weights - self.mask_centers_weight) * self.sigmoid_multiplier)
        self.mask_weights.requires_grad = False
        self.module.weight.requires_grad = True
        if self.module.bias is not None:
            self.frozen_bias_mask = (torch.sigmoid((self.mask_bias - self.mask_centers_bias) * self.sigmoid_multiplier) > threshold) * torch.sigmoid((self.mask_bias - self.mask_centers_bias) * self.sigmoid_multiplier)
            self.mask_bias.requires_grad = False
            self.module.bias.requires_grad = True

    def normalized_masks(self):
        return torch.sigmoid((self.mask_weights - self.mask_centers_weight) * self.sigmoid_multiplier), torch.sigmoid((self.mask_bias - self.mask_centers_bias) * self.sigmoid_multiplier) if self.module.bias is not None else None




def prune_linear(module, sigmoid_multiplier=10.0, method="learnable", sparsity_ratio=None, sparsity_pattern="uniform"):
    if method == "learnable": 
        module.mask_weights = torch.nn.Parameter(torch.ones_like(module.weight) * 2)
        module.mask_bias = torch.nn.Parameter(torch.ones_like(module.bias) * 2) if module.bias is not None else None
        if sparsity_pattern == "uniform":
            module.mask_centers_weight = torch.ones_like(module.weight)
            module.mask_centers_bias = torch.ones_like(module.bias) if module.bias is not None else None
        elif sparsity_pattern == "block_diagonal":
            module.mask_centers_weight = 1.8 * torch.ones_like(module.weight) - 1.3 * pruning.block_diagonal_mask(module.weight.shape, 4).to(module.weight.device)
            module.mask_centers_bias = torch.ones_like(module.bias) if module.bias is not None else None
    elif method == "magnitude":
        if sparsity_ratio is None:
            raise ValueError("sparsity_ratio must be specified for magnitude pruning")
        module.mask_weights = torch.nn.Parameter(pruning.prune_layer(module.weight.data.clone().detach(), sparsity_ratio))
        module.mask_weights.requires_grad = False
        module.mask_bias = torch.nn.Parameter(pruning.prune_layer(module.bias.data.clone().detach(), sparsity_ratio)) if module.bias is not None else None
        if module.mask_bias is not None:
            module.mask_bias.requires_grad = False
    module.weight.requires_grad = False
    if module.bias is not None:
        module.bias.requires_grad = False
    module.sigmoid_multiplier = sigmoid_multiplier
    module.freeze_masks = False

    def forward(self, x):
        if self.freeze_masks:
            return torch.nn.functional.linear(x, self.weight * self.frozen_weight_mask, self.bias * self.frozen_bias_mask if self.bias is not None else None)
        else:
            return torch.nn.functional.linear(x, self.weight * torch.sigmoid((self.mask_weights - self.mask_centers_weight) * self.sigmoid_multiplier), self.bias * torch.sigmoid((self.mask_bias - self.mask_centers_bias) * self.sigmoid_multiplier) if self.bias is not None else None)
    
    def fix_masks(self, threshold=1e-2):
        self.freeze_masks = True
        self.frozen_weight_mask = (torch.sigmoid((self.mask_weights - self.mask_centers_weight) * self.sigmoid_multiplier) > threshold) * torch.sigmoid((self.mask_weights - self.mask_centers_weight) * self.sigmoid_multiplier)
        self.mask_weights.requires_grad = False
        self.weight.requires_grad = True
        if self.bias is not None:
            self.frozen_bias_mask = (torch.sigmoid((self.mask_bias - self.mask_centers_bias) * self.sigmoid_multiplier) > threshold) * torch.sigmoid((self.mask_bias - self.mask_centers_bias) * self.sigmoid_multiplier)
            self.mask_bias.requires_grad = False
            self.bias.requires_grad = True
    
    module.forward = types.MethodType(forward, module)
    module.fix_masks = types.MethodType(fix_masks, module)


def accelerate_matmul(module):
    module.dense_weight = (module.weight.data * module.frozen_weight_mask).to(torch.float32) if (module.frozen_weight_mask.sum() > 0) else None
    module.sparse_weight = module.dense_weight.to_sparse_csr() if module.dense_weight is not None else None
    if module.dense_weight is None:
        module.dense_weight = torch.zeros_like(module.weight)
    module.start_sparse = torch.cuda.Event(enable_timing=True)
    module.end_sparse = torch.cuda.Event(enable_timing=True)
    module.start_dense = torch.cuda.Event(enable_timing=True)
    module.end_dense = torch.cuda.Event(enable_timing=True)

    def forward(self, x):
        module.start_dense.record()
        if x.shape[0] == 1 and len(x.shape) == 3:
            out = torch.unsqueeze(self.dense_weight.matmul(x.squeeze(0).transpose(-1, -2)).transpose(-1, -2) + self.bias.t(), 0) if self.sparse_weight is not None else self.bias.repeat(1, x.shape[0], 1)
        else:
            out = self.dense_weight.matmul(x.transpose(-1, -2)).transpose(-1, -2) + self.bias.t() if self.sparse_weight is not None else self.bias.repeat(x.shape[0], 1)
        module.end_dense.record()
        torch.cuda.synchronize()
        module.start_sparse.record()
        if x.shape[0] == 1 and len(x.shape) == 3:
            out = torch.unsqueeze(self.sparse_weight.matmul(x.squeeze(0).transpose(-1, -2)).transpose(-1, -2) + self.bias.t(), 0) if self.sparse_weight is not None else self.bias.repeat(1, x.shape[0], 1)
        else:
            out = self.sparse_weight.matmul(x.transpose(-1, -2)).transpose(-1, -2) + self.bias.t() if self.sparse_weight is not None else self.bias.repeat(x.shape[0], 1)
        module.end_sparse.record()
        torch.cuda.synchronize()
        print("Matrix Size: ", x.shape, self.dense_weight.shape, "Sparsity Ratio", torch.sum(self.dense_weight != 0) / self.dense_weight.numel(), " Dense Time: ", module.start_dense.elapsed_time(module.end_dense), " Sparse Time: ", module.start_sparse.elapsed_time(module.end_sparse))

        return out
    
    module.forward = types.MethodType(forward, module)