# based on original AALTD 2024 implementation by Angus Dempster, Daniel F Schmidt, Geoffrey I Webb
# Highly Scalable Time Series Classification for Very Large Datasets @ AALTD 2024 (ECML PKDD 2024)
# https://github.com/angus924/aaltd2024
# HYDRA: Competing Convolutional Kernels for Fast and Accurate Time Series Classification
# https://doi.org/10.1007/s10618-023-00939-3

# adaptions and improvements by raphischer

import os
import joblib

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import torch, torch.nn.functional as F
from tqdm import tqdm

# representation_functions 

def identity(X):
    return X

def avg_pool_diff(X):
    return F.avg_pool1d(F.pad(X.diff(), (2, 2), "replicate"), 5, 1)

def diff2(X):
    return X.diff(n=2)

def fft_abs(X):
    return torch.fft.rfft(X).abs()

# generate intervals

def make_intervals(input_length, depth):

    exponent = \
    min(
        depth,
        int(np.log2(input_length)) + 1
    )

    intervals = []

    for n in 2 ** torch.arange(exponent):

        indices = torch.linspace(0, input_length, n + 1).long()

        intervals_n = torch.stack((indices[:-1], indices[1:]), 1)

        intervals.append(intervals_n)

        if n > 1 and intervals_n.diff().median() > 1:

            shift = int(np.ceil(input_length / n / 2))

            intervals.append((intervals_n[:-1] + shift))

    return torch.cat(intervals)

# quantile function

def f_quantile(X, div = 4):

    n = X.shape[-1]

    if n == 1:

        return X.view(X.shape[0], 1, X.shape[1] * X.shape[2])
    
    else:
        
        num_quantiles = 1 + (n - 1) // div

        if num_quantiles == 1:

            quantiles = X.quantile(torch.tensor([0.5]), dim = -1).permute(1, 2, 0)

            return quantiles.view(quantiles.shape[0], 1, quantiles.shape[1] * quantiles.shape[2])
        
        else:
            
            quantiles = X.quantile(torch.linspace(0, 1, num_quantiles), dim = -1).permute(1, 2, 0)
            quantiles[..., 1::2] = quantiles[..., 1::2] - X.mean(-1, keepdims = True)

            return quantiles.view(quantiles.shape[0], 1, quantiles.shape[1] * quantiles.shape[2])

# interval model (per representation)

class IntervalModel():

    def __init__(self, func, input_length, depth = 6, div = 4):

        assert div >= 1
        assert depth >= 1
        self.func = func
        self.div = div
        self.intervals = make_intervals(input_length=input_length, depth=depth)
        self.important_intervals = None # will be transformed to a list after pruning
        self.ft_map = []

    def transform(self, X):

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X.astype(np.float32, copy=False))
        X = self.func(X)
        features = []
        store_ft_map = self.ft_map == [] # pre-pruning, this stores a map of feature indices to intervals and representations

        if self.important_intervals is None: # before pruning
            for idx, (a, b) in enumerate(self.intervals):
                features.append(f_quantile(X[..., a:b], div=self.div).squeeze(1))
                if store_ft_map:
                    self.ft_map.extend([idx] * features[-1].shape[-1]) # store which features belong to which interval (idx)

        else: # after pruning
            for idx in self.important_intervals:
                features.append(f_quantile(X[..., self.intervals[idx][0]:self.intervals[idx][1]], div=self.div).squeeze(1))
        
        return torch.cat(features, -1)

# complete quant transformation

class QuantTransform:

    def __init__(self, ts_channels, ts_length, depth=6, div=4):

        assert depth >= 1
        assert div >= 1

        self.depth = depth
        self.div = div
        self.models = []
        # init the transformation (formerly required fit_transform call)
        dummy_data, dummy_ft = torch.zeros((10, ts_channels, ts_length)), []
        for func in [identity, avg_pool_diff, diff2, fft_abs]:
            Z = func(dummy_data)
            self.models.append( IntervalModel(func=func, input_length=Z.shape[-1], depth=self.depth, div=self.div))
            dummy_ft.append(self.models[-1].transform(dummy_data))
        self.num_features = torch.cat(dummy_ft, -1).shape[1]
        self.number_of_trained_parameters = 0

    def transform(self, X, indices=None): # unified API, indices only used in Hydrant transformation
        # calculate features across all representation models
        features = []
        for model in self.models:
            if model.important_intervals is None or len(model.important_intervals) > 0:
                features.append(model.transform(X))
        res = torch.cat(features, -1)
        return res

class BatchedTransformRF:

    def __init__(self, classifier, transform, num_estimators, criterion, max_features, max_depth, seed, limit_mb=-1):
        clsf_cls = ExtraTreesClassifier if classifier == 'xrf' else RandomForestClassifier
        self.transform = transform
        self.num_estimators = num_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.seed = seed
        self.limit_mb = limit_mb
        random_state = seed if seed >= 0 else None
        self.classifier = clsf_cls(criterion=criterion, max_features=max_features, max_depth=max_depth, random_state=random_state, n_estimators=0, n_jobs=-1, warm_start=True)

    def fit(self, training_data, **kwargs):
        if self.limit_mb > 0:
            training_data.set_batch_size(self.limit_mb)
        else:
            training_data._reset()

        num_batches = training_data._num_batches
        num_estimators_per_batch = self._set_num_estimators(num_batches)
        
        for i, (X, Y) in enumerate(tqdm(training_data, total=num_batches)):
            self.classifier.n_estimators += num_estimators_per_batch[i]
            indices = training_data._batches[training_data._batch_index-1] # needed by Hydrant
            Z = self.transform.transform(X, indices=indices)
            self.classifier.fit(Z.to('cpu'), Y)

    def _set_num_estimators(self, num_batches):

        num_estimators_per = max(1, int(self.num_estimators / num_batches))

        num_estimators_per_batch = np.ones(num_batches, dtype = np.int32) * num_estimators_per

        _total = num_estimators_per_batch.sum()
        _diff = self.num_estimators - _total
        if _diff > 0:
            num_estimators_per_batch[:_diff] += 1

        return num_estimators_per_batch
    
    def _predict(self, test_data, **kwargs):

        # new code with batching
        Y0, i = np.zeros((test_data.shape[0]), dtype=np.int64), 0
        for X, _ in tqdm(test_data, total=np.ceil(test_data.shape[0]/test_data.batch_size)):
            j = i + X.shape[0]
            Z = self.transform.transform(X)
            Y0[i:j] = self.classifier.predict(Z.to('cpu'))
            i = j
        return Y0

    def ft_imp_coeffs(self):
        return torch.tensor(self.classifier.feature_importances_)

    def save_to_disk(self, path):
        # Save sklearn classifier with joblib, and transform with torch if needed
        joblib.dump(self.classifier, os.path.join(path, "classifier.joblib"))
        # Save transform (Quant object) with torch or joblib
        joblib.dump(self.transform, os.path.join(path, "transform.joblib"))
        fsizes = [os.path.getsize(os.path.join(path, f"{f}.joblib")) for f in ['classifier', 'transform']]
        return sum(fsizes)

    def load_from_disk(self, path):
        self.classifier = joblib.load(os.path.join(path, "classifier.joblib"))
        try:
            self.transform = joblib.load(os.path.join(path, "transform.joblib"))
        except RuntimeError: # monkey patch the CPU torch loading (can crash when the transform was cuda-stored)
            import torch
            legacy = torch.load
            def patched_torch_load(*args, **kwargs):
                kwargs["map_location"] = torch.device("cpu")
                return legacy(*args, **kwargs)
            torch.load = patched_torch_load
            self.transform = joblib.load(os.path.join(path, "transform.joblib"))
            torch.load = legacy
        fsizes = [os.path.getsize(os.path.join(path, f"{f}.joblib")) for f in ['classifier', 'transform']]
        return sum(fsizes)

    def count_params(self):
        # Return number of parameters in the classifier and transform
        num_params = 0
        for estimator in self.classifier.estimators_:
            num_params += estimator.tree_.node_count
        # Note: transform parameters are not counted here as they are not learned parameters
        return num_params

class QuantClassifier(BatchedTransformRF):

    def __init__(self, ts_channels, ts_length, classifier='XRF', num_estimators=100, max_depth=20, max_features=0.1, criterion="entropy", seed=None, limit_mb=-1):
        self.transform = QuantTransform(ts_channels, ts_length)
        super().__init__(classifier, self.transform, num_estimators, criterion, max_features, max_depth, seed, limit_mb)
