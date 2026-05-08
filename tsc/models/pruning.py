from itertools import product
import os
import json

import mlflow
import numpy as np
import torch, torch.nn.functional as F
from tqdm import tqdm

from models.hydra import HydraMultivariateGPU
from models.ridge import RidgeClassifier
from models.quant import QuantClassifier, BatchedTransformRF

def prepare_group_info_io(important_groups, to_json=True, device='cpu'):
    for key, vals in important_groups.items():
        if key != 'use_diff':
            if to_json:
                important_groups[key]['groups'] = vals['groups'].cpu().tolist()
                important_groups[key]['shape'] = list(vals['shape'])
                for key2 in ['start', 'end', 'h']:
                    important_groups[key][key2] = important_groups[key][key2].item()
            else: # from json
                important_groups[key]['groups'] = torch.tensor(vals['groups']).to(device)
                important_groups[key]['shape'] = torch.Size(vals['shape'])
                for key2 in ['start', 'end', 'h']:
                    important_groups[key][key2] = torch.tensor(important_groups[key][key2], device=device)
    return important_groups

def init_initermediate(model, transform, seed, device, n_est, cri, max_feat):
    if model == 'ridge':
        return RidgeClassifier(transform=transform, device=device, seed=seed)
    else:
        assert model in ['xrf', 'rf']
        return BatchedTransformRF(classifier=model, transform=transform, num_estimators=n_est, criterion=cri, max_features=max_feat, max_depth=None, seed=seed) # intermediate classifier with unrestricted depth

class PrunedHydra:

    def __init__(self, config, **kwargs):
        self.config = config
        self.trnsf = HydraMultivariateGPU(config)
        self.clsf = RidgeClassifier(transform=self.trnsf, device=config["device"], seed=config['seed'], **kwargs)
        
    def fit(self, training_data, **kwargs):
        # train initial model and check feature importance
        interm_clsf = init_initermediate(self.config['prune_intermediate'], self.trnsf, self.config['seed'], self.config['device'], self.config['num_estimators'], self.config['criterion'], self.config['max_features'])
        interm_clsf.fit(training_data, **kwargs)
        imp = interm_clsf.ft_imp_coeffs()
        results = {
            'ft_imp_type': self.config["prune_intermediate"],
            'n_par_pre_prune': interm_clsf.count_params(),
            'n_ft_pre_prune': imp.shape[0]
        }
        del interm_clsf
        print(f'Pruning {self.config["prune_rate"]*100:.0f}% of {results["n_ft_pre_prune"]} Hydra features with {results["ft_imp_type"]}')
        
        # identify groups with highest mean importance
        kernel_imp = imp.view(self.trnsf.num_dilations, self.trnsf.divisor, self.trnsf.k, self.trnsf.h, 2) # reformat to access min and max response counts of kernels
        mean_kernel_imp = kernel_imp.mean(dim=-1) # average the min and max response count importance -> shape (D, divisor, k, h)
        imp_per_group = {}
        for div, group in product(range(self.trnsf.divisor), range(self.trnsf.h)):
            imp_per_group[(div, group)] = mean_kernel_imp[:, div, :, group].mean().item()
        sorted_imp = sorted(imp_per_group.items(), key=lambda item: item[1], reverse=True)
        important_groups = {}
        num_feat = 0
        for (div, group), _ in sorted_imp[:(int(len(sorted_imp) * (1-self.config["prune_rate"])))]:
            important_groups.setdefault(str(div), []).append(group) # str(div) instead of int because of json saving/loading
            num_feat += 1

        # Collect pruned kernels and info
        important_group_info, all_kernels, current_offset = {}, [], 0
        for dil in range(self.trnsf.num_dilations):
            for div, groups in important_groups.items():
                orig_kernels = self.trnsf.W[dil, int(div)]
                div_h = len(groups)
                keep_kernels = orig_kernels.view(self.trnsf.k, self.trnsf.h, 1, self.trnsf.l)[:, groups].view(self.trnsf.k * div_h, 1, self.trnsf.l)
                # Flatten and store
                keep_kernels_flat = keep_kernels.flatten()
                end_offset = current_offset + keep_kernels_flat.numel()
                
                important_group_info[f"{dil}_{div}"] = {
                    'start': torch.tensor(current_offset, device=self.config['device']),
                    'end': torch.tensor(end_offset, device=self.config['device']),
                    'h': torch.tensor(div_h, device=self.config['device']),
                    'groups': torch.tensor(groups, device=self.config['device']),
                    'shape': keep_kernels.shape
                }
                
                all_kernels.append(keep_kernels_flat)
                current_offset = end_offset

        # Concatenate all into single tensor
        important_group_info['use_diff'] = any([key.endswith('1') for key in important_group_info.keys()])
        self.trnsf = HydraMultivariateGPU(self.config, torch.cat(all_kernels), important_group_info)
        
        # create final ridge classifier
        self.clsf = RidgeClassifier(transform=self.trnsf, device=self.config['device'], seed=self.config['seed'], **kwargs)
        self.clsf.fit(training_data, **kwargs)
        results.update({
            'n_par_post_prune': self.count_params(),
            'n_ft_post_pruning': self.clsf.B.shape[0]
        })
        print('PRUNING RESULT:', results)
        for key, val in results.items():
            if isinstance(val, float):
                mlflow.log_metric(f"phydra_{key}", val)

    def save_to_disk(self, path):
        # after pruning, store the information in json and without torch datatypes
        imp_group_info = prepare_group_info_io(self.trnsf.important_groups)
        json.dump(imp_group_info, open(os.path.join(path, "imp_groups.json"), 'w'))
        self.trnsf.important_groups = prepare_group_info_io(imp_group_info, False, self.config["device"])
        fsize = self.clsf.save_to_disk(path) + os.path.getsize(os.path.join(path, "imp_groups.json"))
        return fsize
        
    def load_from_disk(self, path):
        # after pruning, load the information from json and store with torch datatypes
        if os.path.isfile(os.path.join(path, "imp_groups.json")):
            imp_group_info = json.load(open(os.path.join(path, "imp_groups.json"), 'r'))
            imp_group_info = prepare_group_info_io(imp_group_info, False, self.config["device"])
        # load weights
        state_dict = torch.load(os.path.join(path, "transform.pth"), map_location=self.config["device"])
        self.trnsf = HydraMultivariateGPU(self.config, state_dict['W'], imp_group_info)
        self.trnsf.load_state_dict(state_dict)
        self.clsf = RidgeClassifier(transform=self.trnsf, device=self.config['device'], seed=self.config['seed'])
        fsize = self.clsf.load_from_disk(path) + os.path.getsize(os.path.join(path, "imp_groups.json"))
        return fsize

    def count_params(self):
        return self.clsf.count_params()
    
    def _predict(self, test_data, **kwargs):
        return self.clsf._predict(test_data, **kwargs)


class PrunedQuant(QuantClassifier):

    def __init__(self, prune_rate=0.8, prune_intermediate='ridge', classifier='XRF', num_estimators=100, max_depth=20, max_features=0.1, criterion="entropy", seed=None, limit_mb=-1, **kwargs):

        super().__init__(classifier, num_estimators, max_depth, max_features, criterion, seed, limit_mb, **kwargs)
        self.prune_rate = prune_rate
        self.interm_model = prune_intermediate

    def fit(self, training_data, **kwargs):
        # fit intermediate model
        interm_clsf = init_initermediate(self.interm_model, self.transform, self.seed, device='cpu', n_est=self.num_estimators, cri=self.criterion, max_feat=self.max_features)
        interm_clsf.fit(training_data)
        imp = interm_clsf.ft_imp_coeffs()
        results = {
            'ft_imp_type': "Ridge" if isinstance(interm_clsf, RidgeClassifier) else "XRF",
            'n_par_pre_prune':  interm_clsf.count_params(),
            'n_ft_pre_prune': imp.shape[0],
        }
        del interm_clsf
        print('Identify important features')

        # identify mean feature importance per interval
        avg_imp = {}
        ft_offset = 0 # increases with each representation
        for t_idx, transf in self.transform.models.items():
            self.transform.models[t_idx].important_intervals = [] # placeholder, later to be filled with intervals
            for interval_idx in np.unique(transf.ft_map):
                interval_ft_idc = np.where(np.equal(transf.ft_map, interval_idx))[0]
                ft_start, ft_end = ft_offset + interval_ft_idc.min(), ft_offset + interval_ft_idc.max()
                avg_imp[(t_idx, interval_idx)] = np.mean(imp[ft_start:ft_end+1].numpy())
            ft_offset += len(transf.ft_map)
            
        # prune intervals with low importance
        sorted_imp = sorted(avg_imp.items(), key=lambda item: item[1], reverse=True)
        for (transf_idx, interv_idx), _ in sorted_imp[:(int(len(sorted_imp) * (1-self.prune_rate)))]:
            self.transform.models[transf_idx].important_intervals.append(interv_idx)

        print('Re-fit')

        # refit with pruned intervals
        num_batches = training_data._num_batches
        num_estimators_per_batch = self._set_num_estimators(num_batches)
        for i, (X, Y) in tqdm(enumerate(training_data), total=num_batches):
            self.classifier.n_estimators += num_estimators_per_batch[i]
            Z = self.transform.transform(torch.tensor(X.astype(np.float32)))
            self.classifier.fit(Z, Y)

        results.update({
            'n_par_post_prune': self.count_params(),
            'n_ft_post_prune': self.classifier.n_features_in_
        })
        for key, val in results.items():
            if isinstance(val, float):
                mlflow.log_metric(f"pquant_{key}", val)
        
        print('PRUNING RESULTS', results)
