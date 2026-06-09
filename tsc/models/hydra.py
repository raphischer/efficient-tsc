# based on original AALTD 2024 implementation by Angus Dempster, Daniel F Schmidt, Geoffrey I Webb
# Highly Scalable Time Series Classification for Very Large Datasets @ AALTD 2024 (ECML PKDD 2024)
# https://github.com/angus924/aaltd2024
# HYDRA: Competing Convolutional Kernels for Fast and Accurate Time Series Classification
# https://doi.org/10.1007/s10618-023-00939-3

# adaptions and improvements by raphischer

import math

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import mlflow

from models.ridge import RidgeClassifier


def estimate_hydra_subbatches(n_instances, n_channels, length, num_kernels_per_group, num_groups, max_memory_gb=24.0, overhead=3, dtype=torch.float32):
    """
    Estimate how many subbatches are needed to keep hydra memory usage below max_memory_gb.
    """
    # Get dtype size in bytes
    bytes_per_elem = torch.finfo(dtype).bits // 8
    total_bytes = n_instances * n_channels * length * bytes_per_elem * num_kernels_per_group * num_groups * overhead # expected tensor sizes + some overhead
    total_gb = total_bytes / (1024**3)
    subbatch = math.ceil(total_gb / max_memory_gb) # Compute required number of splits so each fits under limit
    if subbatch > 1:
        print(f'Hydrant will perform Hydra transformations on {subbatch} sub-batches per {n_instances} instances batch, to not extend memory capacity of {max_memory_gb} GB.')
    return subbatch # TODO improve by also checking whether pruned models might be able to handle larger batches?


class HydraMultivariateGPU(nn.Module):

    def __init__(self, config, custom_kernels=None, custom_imp_group_info=None):

        super().__init__()
        dev = config["device"]
        for short, cfg in zip(['k', 'g', 'l'], ['num_kernels_per_group', 'num_groups', 'kernel_length']):
            self.register_buffer(short, torch.tensor(config[cfg], dtype=torch.int64, device=dev))

        max_exponent = np.log2((config['length'] - 1) / (self.l.cpu() - 1))
        num_channels_per = np.clip(config['n_channels'] // 2, 2, config["max_num_channels"])

        self.register_buffer('dilations', 2 ** torch.arange(int(max_exponent) + 1, device=dev))
        self.register_buffer('num_dilations', torch.tensor(len(self.dilations), dtype=torch.int64, device=dev))
        self.register_buffer('divisor', torch.tensor(min(2, self.g), dtype=torch.int64, device=dev))
        self.register_buffer('h', self.g // self.divisor)
        self.register_buffer('paddings', torch.div((self.l - 1) * self.dilations, 2, rounding_mode="floor").int())
        self.register_buffer('I', torch.randint(0, config['n_channels'], (self.num_dilations, self.divisor, self.h, num_channels_per), device=dev))

        if custom_kernels is None: # place for storing info about important kernels - only used by PHYDRA
            W = torch.randn(self.num_dilations, self.divisor, self.k * self.h, 1, self.l, device=dev)
            W = W - W.mean(-1, keepdims=True)
            W = W / W.abs().sum(-1, keepdims=True)
            self.register_buffer("W", W)
            self.important_groups = None
        else:
            assert custom_imp_group_info is not None, "If providing custom kernels, must also provide important group info"
            self.register_buffer("W", custom_kernels.to(dev))
            self.important_groups = custom_imp_group_info

        self.register_buffer('num_features', torch.tensor(int(np.prod(self.W.shape) / config['kernel_length'] * 2), device=config['device'])) # counting min and max
        self.subbatches = estimate_hydra_subbatches(config['batch_size'], config['n_channels'], config['length'], config['num_kernels_per_group'], config['num_groups'])
        mlflow.log_param('hydra_subbatches', self.subbatches)
        self.number_of_trained_parameters = self.I.shape.numel() + self.W.shape.numel()
        
    def forward(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X.astype(np.float32, copy=False), device=self.W.device)
        if X.device != self.W.device:
            X = X.to(self.W.device)
        if self.subbatches == 1: # no sub-batching
            Z = self.transform_single(X.to(device=self.W.device))
        else: # write sub-batch results into prepared tensor on specified device
            i, Z = 0, torch.zeros((X.shape[0], self.num_features), device=self.W.device)
            for X__ in torch.chunk(X, self.subbatches, dim=0):
                Z[i:(i+X__.shape[0])] = self.transform_single(X__.to(device=self.W.device))
                i += X__.shape[0]
        
        # OTHER SUB-BATCHING IMPLEMENTATIONS (but empirically slower!)
        # write into list on CPU, and afterwards concat
        # ZH_ = [ self.hydra(X__.to(device=self.config['device']))[0] for X__ in torch.chunk(X_, self.hydra_subbatch, dim=0) ] # smaller batches for Hydra, to not crash memory
        # ZH_ = torch.cat(ZH_).to('cpu')
        # write into prepared tensor on CPU
        # ZH__, i = torch.zeros((ZQ.shape[0], self.hydra.num_features)), 0
        # for X__ in torch.chunk(X_, self.hydra_subbatch, dim=0):
        #     ZH__[i:(i+X__.shape[0])] = self.hydra(X__.to(device=self.config['device']))[0].to('cpu')
        #     i += X__.shape[0]

        return Z
    
    def transform_single(self, X):

        num_examples = X.shape[0]
        if self.divisor > 1 and (self.important_groups is None or self.important_groups['use_diff']):
            diff_X = torch.diff(X)

        Z, h = [], self.h

        for dilation_index in range(self.num_dilations):

            d = self.dilations[dilation_index].item()
            p = self.paddings[dilation_index].item()

            for diff_index in range(self.divisor):
                if self.important_groups and f"{dilation_index}_{diff_index}" not in self.important_groups: # str instead of int because of json saving/loading
                    continue # skip if there are no important groups for this diff_index

                input = X[:, self.I[dilation_index, diff_index]].sum(2) if diff_index == 0 else diff_X[:, self.I[dilation_index, diff_index]].sum(2)

                if self.important_groups is not None: # only select kernels and input for import groups (after pruning)
                    info = self.important_groups[f"{dilation_index}_{diff_index}"]
                    h = info['h']
                    kernels = self.W[info['start']:info['end']].view(info['shape'])
                    input = input[:,info['groups']]
                else:
                    kernels = self.W[dilation_index, diff_index]

                _Z = F.conv1d(input, kernels, dilation=d, padding=p, groups=h).view(num_examples, h, self.k, -1)

                max_values, max_indices = _Z.max(2)
                min_values, min_indices = _Z.min(2)
                
                count_max = torch.zeros(num_examples, h, self.k, device=X.device)
                count_min = torch.zeros(num_examples, h, self.k, device=X.device)

                count_max.scatter_add_(-1, max_indices, max_values)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))

                Z.append(count_max)
                Z.append(count_min)

        Z = torch.cat(Z, 1).view(num_examples, -1)    
        return Z.clamp(0).sqrt()

    def transform(self, X, indices=None): # unified API, indices only used in Hydrant transformation
        return self(X)

class Hydra(RidgeClassifier):

    def __init__(self, config):
        transform = HydraMultivariateGPU(config)
        super().__init__(transform=transform, device=config['device'], seed=config['seed'])
