# Angus Dempster, Chang Wei Tan, Lynn Miller
# Navid Mohammadi Foumani, Daniel F Schmidt, and Geoffrey I Webb
# Highly Scalable Time Series Classification for Very Large Datasets
# AALTD 2024 (ECML PKDD 2024)

# Angus Dempster, Daniel F Schmidt, Geoffrey I Webb
# HYDRA: Competing Convolutional Kernels for Fast and Accurate Time Series Classification
# https://doi.org/10.1007/s10618-023-00939-3

# adaptions and improvements by ANON2

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from models.ridge import RidgeClassifier


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
        
    def forward(self, X):

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

    def transform(self, X):
        return self(X)

    def fit_transform(self, X, Y):
        return self(X)

class Hydra(RidgeClassifier):

    def __init__(self, config):
        transform = HydraMultivariateGPU(config)
        super().__init__(transform=transform, device=config['device'], seed=config['seed'])
