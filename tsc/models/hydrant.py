# QFeat-HLogit-ET: Quant features + out-of-fold Hydra logits stacked into an ExtraTrees
# meta-learner. Pruning, when enabled, applies to Quant intervals only.
# Maniar, arXiv:2512.06666v1

import os

import numpy as np
import torch
import mlflow
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from models.hydra import Hydra
from models.pruning import PrunedHydra, RidgeClassifier, init_intermediate
from models.quant import BatchedTransformRF, QuantTransform


class QuantLogitHydraTransform(QuantTransform):

    def __init__(self, ts_channels, ts_length, n_labels):
        super().__init__(ts_channels, ts_length)
        self.num_features += n_labels # hydra logits will also be used as features
        self.hydra_logits = None # will be OOF logits for training and hydra_final transformation for inference

    def transform(self, X, indices=None):
        ZQ = super().transform(X)
        if isinstance(self.hydra_logits, np.ndarray):
            assert indices is not None, "indices must be provided for OOF Hydra logits during training"
            ZH = torch.tensor(self.hydra_logits[indices])
        else: # fully trained hydra instance
            ZH = self.hydra_logits._predict_single(X)
        return torch.cat([ZH, ZQ.to(ZH.device)], 1)


class Hydrant(BatchedTransformRF):

    def __init__(self, config):
        self.config       = config
        self.clsf_cls     = ExtraTreesClassifier if config['classifier'] == 'xrf' else RandomForestClassifier
        self.prune_rate   = config['prune_rate']
        self.n_folds      = config['n_folds']
        self.random_state = config['seed'] if config['seed'] >= 0 else None
        self.transform = QuantLogitHydraTransform(config['n_channels'], config['length'], config['n_labels'])
        super().__init__(classifier=config['classifier'], transform=self.transform, num_estimators=config['num_estimators'], max_depth=config['max_depth'], max_features=config['max_features'], criterion=config['criterion'], seed=config['seed'])

    def fit(self, training_data, **kwargs):
        num_classes = kwargs.get('num_classes', len(training_data.classes))
        oof_train_logits = self._oof_hydra_logits(training_data, num_classes)
        self.transform.hydra_logits = oof_train_logits

        prune_results = {}
        if self.prune_rate == 0: # just train Quant classifier on the internally transformed data batches (Quant + Logits)
            print('::: Training Hydrant classifier without pruning')
            super().fit(training_data, **kwargs)

        else: # first fit intermediate model and then prune unimportant intervals for inference efficiency
            print('::: Training Hydrant interim feature importance detector')
            interm_clsf = init_intermediate(self.config['prune_intermediate'], self.transform, self.config['seed'], self.config['device'], self.config['num_estimators'], self.config['criterion'], self.config['max_features'])
            interm_clsf.fit(training_data)
            imp = interm_clsf.ft_imp_coeffs().to('cpu')
            prune_results.update({
                'ft_imp_type': "ridge" if isinstance(interm_clsf, RidgeClassifier) else "xrf",
                'n_par_pre_prune':  interm_clsf.count_params(),
                'n_ft_pre_prune': imp.shape[0],
                'n_qft_pre_prune': imp.shape[0] - num_classes,
                'n_hft_pre_prune': num_classes,
            })
            del interm_clsf
            # assess average importance for each representation and interval
            avg_imp, ft_offset = {}, 0
            for t_idx, transf in enumerate(self.transform.models):
                self.transform.models[t_idx].important_intervals = [] # placeholder for later
                for interval_idx in np.unique(transf.ft_map):
                    interval_ft_idc = np.where(np.equal(transf.ft_map, interval_idx))[0]
                    ft_start, ft_end = ft_offset + interval_ft_idc.min(), ft_offset + interval_ft_idc.max()
                    avg_imp[(t_idx, interval_idx)] = np.mean(imp[ft_start:ft_end+1].numpy())
                ft_offset += len(transf.ft_map)
            # prune intervals with low importance
            sorted_imp = sorted(avg_imp.items(), key=lambda item: item[1], reverse=True)
            for (transf_idx, interv_idx), _ in sorted_imp[:(int(len(sorted_imp) * (1-self.prune_rate)))]:
                self.transform.models[transf_idx].important_intervals.append(interv_idx)
            # retrain final classifier on the new transformations
            print('::: Training final (pruned) Hydrant classifier')
            super().fit(training_data, **kwargs)

        # final-stage Hydra can also be pruned, logit-distribution shifts are expected to be neglectable (important kernels are kept)
        print('::: Training final (pruned) Hydra transformation')
        self.transform.hydra_logits = PrunedHydra(self.config) if self.prune_rate > 0 else Hydra(self.config)
        self.transform.hydra_logits.fit(training_data, num_classes=num_classes)
        self.transform.number_of_trained_parameters += self.transform.hydra_logits.transform.number_of_trained_parameters
        self.transform.num_features += self.transform.hydra_logits.transform.num_features

        if self.prune_rate > 0:
            prune_results.update({
                'n_par_post_prune': self.count_params(),
                'n_ft_post_prune': self.classifier.n_features_in_,
                'n_qft_post_prune': self.classifier.n_features_in_ - num_classes,
                'n_hft_post_prune': num_classes,
            })
            print('PRUNING HYDRANT RESULTS:', prune_results)
            for key, val in prune_results.items():
                if isinstance(val, float):
                    mlflow.log_metric(f"hydrant_{key}", val)

    def _oof_hydra_logits(self, training_data, num_classes):
        n_train = training_data.shape[0]
        oof_logits = np.zeros((n_train, num_classes), dtype=np.float32)
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for fold_train_idx, fold_val_idx in tqdm(skf.split(np.arange(n_train), training_data.Y),
                                                  total=self.n_folds, desc='::: OOF Hydra folds'):
            hydra = Hydra(self.config)
            hydra.fit(training_data[fold_train_idx], num_classes=num_classes)
            val_data = training_data[fold_val_idx].unbatch() # TODO necessary?
            oof_logits[fold_val_idx] = hydra._predict(val_data, num_classes=num_classes).cpu().numpy()

        return oof_logits

    def save_to_disk(self, path):
        quant_fsize = super().save_to_disk(path)
        hydra_fsize = self.transform.hydra_logits.save_to_disk(path)
        return quant_fsize + hydra_fsize

    def load_from_disk(self, path):
        quant_fsize = super().load_from_disk(path)
        # ensure correct device is set after loading, in case it was trained on a different one
        if hasattr(self.transform.hydra_logits, 'config'): # for pruned Hydra
            self.transform.hydra_logits.config['device'] = self.config['device']
        elif hasattr(self.transform.hydra_logits, 'device'): # for unpruned Hydra / standard Ridge
            self.transform.hydra_logits.device = self.config['device']
        hydra_fsize = self.transform.hydra_logits.load_from_disk(path)
        return quant_fsize + hydra_fsize

    def count_params(self):
        n_hydra = self.transform.hydra_logits.count_params()
        n_meta  = sum(est.tree_.node_count for est in self.classifier.estimators_)
        return n_hydra + n_meta
