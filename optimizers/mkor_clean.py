import math

import torch
import torch.optim as optim
import numpy as np

from .utils.kfac_utils import (ComputeCovA, ComputeCovG)
from utils.timing import Timer
from .utils.factors import ComputeI, ComputeG
import torch.distributed as dist


class MKOROptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.001,
                 stat_decay=0.95,
                 weight_decay=0,
                 momentum=0.9,
                 inv_freq=10,
                 measure_time=False,
                 half_precision=True,
                 grad_accum_steps=1,
                 sgd_layers=[],
                 optimizer='sgd',
                 grad_scale=1.0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        # TODO (CW): MKOR optimizer now only support model as input
        super(MKOROptimizer, self).__init__(model.parameters(), defaults)
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.AHandler = ComputeI()
        self.GHandler = ComputeG()

        self.verbose = dist.get_rank() == 0

        self.known_modules = {'Linear', 'Conv2d'}

        self.modules = []

        self.model = model
        self.device = self.model.device

        self._prepare_model(sgd_layers=sgd_layers)

        self.steps = 0

        self.grad_outputs = [0] * len(self.modules)

        self.a_reset_factor = [True] * len(self.modules)
        self.g_reset_factor = [True] * len(self.modules)

        self.m_aa, self.m_gg = [0] * len(self.modules), [0] * len(self.modules)
        self.Q_a, self.Q_g = [0] * len(self.modules), [0] * len(self.modules)
        self.d_a, self.d_g = [0] * len(self.modules), [0] * len(self.modules)
        self.AA_inv, self.GG_inv = [0] * len(self.modules), [0] * len(self.modules)
        self.AA_Us, self.AA_Vt = [0] * len(self.modules), [0] * len(self.modules)
        self.GG_Us, self.GG_Vt = [0] * len(self.modules), [0] * len(self.modules)
        self.AA_sparse_factor, self.GG_sparse_factor = [0] * len(self.modules), [0] * len(self.modules)
        self.AA, self.GG = [0] * len(self.modules), [0] * len(self.modules)
        self.stat_decay = stat_decay

        self.inv_freq = inv_freq

        # Timing Variables
        self.timer = Timer(measure=measure_time)

        self.reset_factors_freq = 10

        self.inputs = [0] * len(self.modules)
        self.input_shapes = [0] * len(self.modules)
        self.inputs_reduced = False
        self.grads = [0] * len(self.modules)
        self.grad_shapes = [0] * len(self.modules)

        self.data_type = torch.float16 if half_precision else torch.float32

        self.manual_reset_factors = False

        if self.manual_reset_factors:
            self.reset_weight = 0.8
        else:
            self.reset_weight = 0.1

        self.error_average_list = []
        self.error_svd_list = []

        self.rank = 1
        self.sparse = False
        self.sparse_threshold = 5e-3
        self.sparse_AA, self.sparse_GG = [0] * len(self.modules), [0] * len(self.modules)

        self.dummy_timer_start = torch.cuda.Event(enable_timing=True)
        self.dummy_timer_end = torch.cuda.Event(enable_timing=True)
        self.grad_accum_steps = grad_accum_steps
        self.accumulated_steps = [0] * len(self.modules)

        if self.sparse:
            self.compute_sparse_preconditioning_costs()

        self.clipping_value = 100.0

        self.sgd_layers = sgd_layers

        self.warmup_steps = 0

        self.set_optimizer(optimizer)

        self.grad_scale = grad_scale

    def set_optimizer(self, optimizer):
        if type(optimizer) == str:
            if optimizer == 'sgd':
                self.optimizer = optim.SGD(self.param_groups, lr=self.defaults['lr'],
                                           weight_decay=self.defaults['weight_decay'],
                                           momentum=self.defaults['momentum'])
            elif optimizer == 'adam':
                self.optimizer = optim.Adam(self.param_groups, lr=self.defaults['lr'],
                                            weight_decay=self.defaults['weight_decay'])
            else:
                raise ValueError("Invalid optimizer: {}".format(optimizer))
        else:
            self.optimizer = optimizer

        self.param_groups = self.optimizer.param_groups
    def inverse(self, prev_inv, rank_1):
        tmp1 = (prev_inv @ rank_1)
        tmp2 = (rank_1.t() @ prev_inv)
        return prev_inv - 1 / (1 + tmp2 @ rank_1) * tmp1 @ tmp2

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and (self.steps % self.inv_freq == 0 or self.steps < self.warmup_steps):
            a = torch.clamp(self.AHandler(input[0].data, module).to(torch.float32) / self.grad_accum_steps,
                            -self.clipping_value, self.clipping_value)
            if self.accumulated_steps[self.index(module)] % self.grad_accum_steps == 0:
                self.inputs[self.index(module)] = a.to(self.data_type)
            else:
                self.inputs[self.index(module)] += a.to(self.data_type)

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.steps % self.inv_freq == 0 or self.steps < self.warmup_steps:
            if not self.inputs_reduced and \
                    self.accumulated_steps[self.index(module)] % self.grad_accum_steps == self.grad_accum_steps - 1:
                self.inputs = self.low_rank_approx(self.inputs)
                self.reduced_inputs, self.input_handles, self.input_shapes = self.reduce_data(self.inputs)
                self.inputs_reduced = True
            g, _ = self.GHandler(grad_output[0].data, module)
            g = torch.clamp(g.to(torch.float32) / (self.grad_accum_steps * self.grad_scale), -self.clipping_value,
                            self.clipping_value)
            if self.accumulated_steps[self.index(module)] % self.grad_accum_steps == 0:
                self.grads[self.index(module)] = g.to(self.data_type)
            else:
                self.grads[self.index(module)] += g.to(self.data_type)
            self.accumulated_steps[self.index(module)] += 1

    def _prepare_model(self, sgd_layers=[]):
        self.apply_sgd = []
        self.index_dict = {}
        count = 0
        if self.verbose:
            print("=> We keep following layers in MKOR. ")
        for module in self.model.modules():
            classname = module.__class__.__name__
            # print('=> We keep following layers in MKOR. <=')
            if classname in self.known_modules:
                self.modules.append(module)
                self.index_dict[module] = count
                self.apply_sgd.append(module in sgd_layers)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
                if self.verbose:
                    print('(%s): %s' % (count, module))
                count += 1

    def index(self, module):
        return self.index_dict[module]

    @staticmethod
    def _get_matrix_form_grad(m, classname):
        """
        :param m: the layer
        :param classname: the class name of the layer
        :return: a matrix form of the gradient. it should be a [output_dim, input_dim] matrix.
        """
        if classname == 'Conv2d':
            p_grad_mat = m.weight.grad.data.view(m.weight.grad.data.size(0), -1)  # n_filters * (in_c * kw * kh)
        else:
            p_grad_mat = m.weight.grad.data
        if m.bias is not None:
            p_grad_mat = torch.cat([p_grad_mat, m.bias.grad.data.view(-1, 1)], 1)
        return p_grad_mat
    def dense_precondition(self, module, grad_mat):
        return self.GG_inv[self.index(module)].to(torch.float32) @ grad_mat @ self.AA_inv[self.index(module)].to(
            torch.float32)

    def _get_natural_grad(self, m, p_grad_mat, identity=False):
        """
        :param m:  the layer
        :param p_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        # p_grad_mat is of output_dim * input_dim
        # inv((ss')) p_grad_mat inv(aa') = [ Q_g (1/R_g) Q_g^T ] @ p_grad_mat @ [Q_a (1/R_a) Q_a^T]
        # v1 = self.Q_g[self.index(m)].t() @ p_grad_mat @ self.Q_a[self.index(m)]
        # v2 = v1 / (self.d_g[self.index(m)].unsqueeze(1) * self.d_a[self.index(m)].unsqueeze(0) + damping)
        # v = self.Q_g[self.index(m)] @ v2 @ self.Q_a[self.index(m)].t()
        if identity:
            v = p_grad_mat
        else:
            v = self.dense_precondition(m, p_grad_mat)
        if m.bias is not None:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.data.size())
            v[1] = v[1].view(m.bias.grad.data.size())
        else:
            v = [v.view(m.weight.grad.data.size())]

        return v

    def _kl_clip_and_update_grad(self, updates):
        # Reset Norms
        norm_fixer = {}
        for m in self.modules:
            grad_norm = m.weight.grad.data.norm(2)
            update_norm = updates[self.index(m)][0].norm(2)
            if m.bias is not None:
                grad_norm += m.bias.grad.data.norm(2)
                update_norm += updates[self.index(m)][1].norm(2)
            norm_fixer[self.index(m)] = grad_norm / update_norm
            if torch.isnan(norm_fixer[self.index(m)]):  # Gradient is zero
                # if self.verbose:
                #     print("Gradient is zero for module: ", m)
                continue
            v = updates[self.index(m)]
            m.weight.grad.data.copy_(v[0])
            m.weight.grad.data.mul_(norm_fixer[self.index(m)])
            if m.bias is not None:
                m.bias.grad.data.copy_(v[1])
                m.bias.grad.data.mul_(norm_fixer[self.index(m)])

    def reduce_data(self, data_dict):
        if dist.get_world_size() == 1:
            return None, None, None
        data_list = []
        data_shapes = []
        for module in self.modules:
            if self.apply_sgd[self.index(module)]:
                continue
            data_list.append(data_dict[self.index(module)].reshape(1, -1))
            data_shapes.append(data_dict[self.index(module)].shape)
        reduced_data = torch.cat(data_list, dim=1)
        del data_list
        handles = []
        handles.append(self.allreduce(reduced_data, async_op=True, average=True))
        return reduced_data, handles, data_shapes

    def sync_data(self, synchronized_data, data_dict, data_shapes, handles, distributed_computation=False,
                  distribution_dict=None):
        if dist.get_world_size() == 1:
            return

        self.sync(handles)

        if distributed_computation:
            rank = dist.get_rank()
            offset = 0
            i = 0
            for module in self.modules:
                if self.apply_sgd[self.index(module)] or rank != distribution_dict[self.index(module)]:
                    continue
                data_shape = data_shapes[rank][i]
                data_numel = torch.prod(torch.tensor(data_shape)).item()
                data_dict[self.index(module)] = synchronized_data[0, offset:offset + data_numel].reshape(data_shape)
                offset += data_numel
        else:
            offset = 0
            i = 0
            for module in self.modules:
                if self.apply_sgd[self.index(module)]:
                    continue
                data_shape = data_shapes[i]
                data_numel = torch.prod(torch.tensor(data_shapes[i])).item()
                data_dict[self.index(module)] = synchronized_data[0, offset:offset + data_numel].reshape(data_shape)
                offset += data_numel
                i += 1

    def update_inv_factors(self, rank_1_dict, factor_dict, reset_factor_dict):
        rank = dist.get_rank()
        for module in self.modules:
            if self.apply_sgd[self.index(module)]:
                continue
            a = rank_1_dict[self.index(module)]
            v = a.t()
            if reset_factor_dict[self.index(module)]:
                self.reset_factors(module, factor_dict, a.size(1), reset_factor_dict)

            else:
                self.invert_factor(module, factor_dict, a, v)
                if self.rank != 1:
                    low_rank_Us_dict[self.index(module)], low_rank_Vt_dict[self.index(module)] = self.randomized_svd(
                        factor_dict[self.index(module)].to(torch.float32), self.rank)
                else:
                    if self.sparse:
                        self.sparsify_factor(module, factor_dict, sparsity_dict, sparse_factor_dict)

    def invert_factor(self, module, factor_dict, a, v):
        factor_dict[self.index(module)] = self.inverse(factor_dict[self.index(module)] / self.stat_decay,
                                                       v * math.sqrt(1 - self.stat_decay))
        self.set_reset_factor_flags(module, factor_dict)

    def reset_factors(self, module, factor_dict, dim, reset_factor_dict):
        if module not in factor_dict:
            factor_dict[self.index(module)] = torch.eye(dim, device=self.device, dtype=self.data_type)
        else:
            factor_dict[self.index(module)] = factor_dict[self.index(module)] * (
                        1 - self.reset_weight) + self.reset_weight * torch.eye(
                dim, device=self.device, dtype=self.data_type)
        reset_factor_dict[self.index(module)] = False

    def set_reset_factor_flags(self, module, factor_dict):
        if self.manual_reset_factors:
            self.a_reset_factor[self.index(module)] = self.steps % (self.inv_freq * self.reset_factors_freq) == 0
            self.g_reset_factor[self.index(module)] = self.steps % (self.inv_freq * self.reset_factors_freq) == 0
        else:
            if torch.max(torch.abs(factor_dict[self.index(module)].flatten())) > 2:
                self.a_reset_factor[self.index(module)] = True
                self.g_reset_factor[self.index(module)] = True

    def reduce_and_update_factors(self):
        self.grads = self.low_rank_approx(self.grads)

        self.reduced_grads, self.grad_handles, self.grad_shapes = self.reduce_data(self.grads)
        self.sync_data(self.reduced_inputs, self.inputs, self.input_shapes, self.input_handles)
        self.inputs_reduced = False
        self.update_inv_factors(self.inputs, self.AA_inv, self.a_reset_factor)

        self.sync_data(self.reduced_grads, self.grads, self.grad_shapes, self.grad_handles)

        self.update_inv_factors(self.grads, self.GG_inv, self.g_reset_factor)

    def step(self, closure=None):
        # FIXME(CW): temporal fix for compatibility with Official LR scheduler.
        group = self.param_groups[0]
        lr = group['lr']
        updates = {}
        if self.steps % self.inv_freq == 0 or self.steps < self.warmup_steps:
            self.timer("reduce_and_update_factors", self.reduce_and_update_factors)
            # self.compute_min_eigenvals()
        for m in self.modules:
            classname = m.__class__.__name__
            p_grad_mat = self.timer("precondition", self._get_matrix_form_grad, m=m, classname=classname)
            v = self.timer("precondition", self._get_natural_grad, m=m, p_grad_mat=p_grad_mat,
                           identity=self.apply_sgd[self.index(m)])
            updates[self.index(m)] = v

        self.timer("update_weights", self._kl_clip_and_update_grad, updates=updates,)

        self.timer("update_weights", self.optimizer.step, closure=closure)
        for param_group in self.param_groups:
            if not 'step' in param_group:
                param_group['step'] = 0
            else:
                param_group['step'] += 1
        self.steps += 1

    def update_grad_scale(self, scaler):
        self.grad_scale = scaler

    def low_rank_approx(self, data_dict):
        for module in self.modules:
            a = data_dict[self.index(module)]
            a = torch.mean(a, dim=0, keepdim=True)

            data_dict[self.index(module)] = a
        return data_dict

    def state_dict(self):
        return {
            "grad_outputs": self.grad_outputs,
            "a_reset_factor": self.a_reset_factor,
            "g_reset_factor": self.g_reset_factor,
            "steps": self.steps,
            "AA_inv": self.AA_inv,
            "GG_inv": self.GG_inv,
            "AA_Us": self.AA_Us,
            "AA_Vt": self.AA_Vt,
            "GG_Us": self.GG_Us,
            "GG_Vt": self.GG_Vt,
            "AA_sparse_factor": self.AA_sparse_factor,
            "GG_sparse_factor": self.GG_sparse_factor,
            "AA": self.AA,
            "GG": self.GG,
            "stat_decay": self.stat_decay,
            "inv_freq": self.inv_freq,
            "reset_factors_freq": self.reset_factors_freq,
            "inputs": self.inputs,
            "input_shapes": self.input_shapes,
            "grads": self.grads,
            "grad_shapes": self.grad_shapes,
            "data_type": self.data_type,
            "manual_reset_factors": self.manual_reset_factors,
            "reset_weight": self.reset_weight,
            "rank": self.rank,
            "sparse": self.sparse,
            "sparse_threshold": self.sparse_threshold,
            "sparse_AA": self.sparse_AA,
            "sparse_GG": self.sparse_GG,
            "grad_accum_steps": self.grad_accum_steps,
            "accumulated_steps": self.accumulated_steps,
            "clipping_value": self.clipping_value,
            "sgd_layers": self.sgd_layers,
            "warmup_steps": self.warmup_steps,
            "optimizer": self.optimizer.state_dict(),
            "grad_scale": self.grad_scale,
            "apply_sgd": self.apply_sgd,
        }

    def load_state_dict(self, state_dict):
        if len(state_dict) == 2:
            return
        for key in state_dict:
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict[key])
                self.param_groups = self.optimizer.param_groups
            else:
                if type(state_dict[key]) == list:
                    for i in range(len(state_dict[key])):
                        if type(state_dict[key][i]) == torch.Tensor:
                            state_dict[key][i] = state_dict[key][i].to(self.device)
                setattr(self, key, state_dict[key])
        print("Checkpoint Loaded")

    def broadcast(self, tensor, src, async_op=True):
        return dist.broadcast(tensor.contiguous(), src=src, async_op=async_op)

    def sync(self, handles):
        if isinstance(handles, list):
            if len(handles) == 0:
                return
            if isinstance(handles[0], tuple):
                for handle, tensor in handles:
                    self.wait(handle)
            else:  # async broadcast
                for handle in handles:
                    self.wait(handle)
        else:
            if isinstance(handles, tuple):
                handle, tensor = handles
                self.wait(handle)
                if isinstance(tensor, list):  # async allgather
                    pass
                else:  # async allreduce
                    tensor /= self.size()
            else:  # async broadcast
                self.wait(handles)

    def allreduce(self, tensor, async_op=True, average=False):
        # the result are averaged, not summed
        if average:
            operator = torch.distributed.ReduceOp.AVG
            average = False
        else:
            operator = torch.distributed.ReduceOp.SUM
        handle = dist.all_reduce(tensor, async_op=async_op, op=operator)

        if async_op == False:
            return
        return (handle, tensor)

    def reduce(self, tensor, dst, async_op=True, average=False):
        # the result are averaged, not summed
        if average:
            operator = torch.distributed.ReduceOp.AVG
            average = False
        else:
            operator = torch.distributed.ReduceOp.SUM
        handle = dist.reduce(tensor, dst=dst, async_op=async_op, op=operator)

        if async_op == False:
            return
        return (handle, tensor)

    def wait(self, handle):
        if handle:
            handle.wait()