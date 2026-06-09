# based on original AALTD 2024 implementation by Angus Dempster, Daniel F Schmidt, Geoffrey I Webb
# Highly Scalable Time Series Classification for Very Large Datasets @ AALTD 2024 (ECML PKDD 2024)
# https://github.com/angus924/aaltd2024

# adaptions and improvements by raphischer

import os
import time

import numpy as np
import torch, torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split

EPS = np.finfo(np.float32).eps

def binarize(Y, n):
    return -torch.ones(Y.shape[0], n).scatter(-1, torch.tensor(Y[:, None]).long(), -1)

def stratified_split(Y, validation_size, seed=None):
    rng = np.random.default_rng(seed) if seed >= 0 else np.random.default_rng()
    U, C = np.unique(Y, return_counts = True)
    _C = ((C / C.sum()) * validation_size).round().clip(1).astype(np.int64)
    VA = np.zeros(_C.sum(), dtype = np.int64)
    a = 0
    for i, y in enumerate(U):
        c = _C[i]
        b = a + c
        J = (Y == y).nonzero()[0]
        K = rng.choice(J, c, replace = False)
        VA[a:b] = K
        a = b
    return np.setdiff1d(np.arange(Y.shape[0]), VA), VA

class Scaler(nn.Module):

    def __init__(self, num_values, device, dtype, **kwargs):

        super().__init__()
        self.register_buffer("_mean", torch.zeros(num_values, dtype=dtype, device=device))
        self.register_buffer("_std", torch.zeros(num_values, dtype=dtype, device=device))
        self.register_buffer("_count", torch.tensor(0, dtype=torch.int64, device=device))
        self.register_buffer("_eps", torch.tensor(kwargs.get("eps", EPS * 10), device=device))
        self.register_buffer("_with_std", torch.tensor(kwargs.get("with_std", True), device=device))

    def partial_fit(self, X):

        batch_size = X.shape[0]
        new_count = self._count + batch_size

        batch_mean = X.mean(0)
        batch_std = X.std(0) if batch_size > 1 else 0

        self._mean = self._mean + ((batch_mean - self._mean) * (batch_size / new_count))
        self._std = self._std + ((batch_std - self._std) * (batch_size / new_count))

        self._count = new_count

    def fit(self, X):
        self._mean = X.mean(0)
        self._std = X.std(0)

    def scale(self, X):
        if self._with_std:
            return (X - self._mean) / (self._std + self._eps)
        else:
            return (X - self._mean)
        
    def fit_transfrom(self, X):
        self.fit(X)
        return self.scale(X)

class RidgeClassifier():

    def __init__(self, transform, device="cpu", seed=None, **kwargs):

        self.transform = transform
        self.device = device
        self.X_scaler = None
        self.Y_scaler = None
        self.lambdas = kwargs.get("lambdas", torch.logspace(-6, 6, 21))
        self.B = None
        self.B0 = None
        self.seed = seed

    def fit(self, training_data, **kwargs):
            
        n, p, k = training_data.shape[0], self.transform.num_features, kwargs.get("num_classes", len(training_data.classes))
        max_val_size = kwargs.get("max_val_size", 8_192)
        val_size = min(int(n * 0.2), max_val_size)

        if n < p: # low amounts of data, so first transform all batches, then fit ridge

            # calculate transformation features
            X0, Y0 = torch.zeros((n, p), device = self.device), torch.zeros((n, k), device = self.device)
            i = 0
            for X, Y in tqdm(training_data, total=np.ceil(n/training_data.batch_size), desc="Ridge single pass (n < p)"):
                j = i + X.shape[0]
                _idc = training_data._batches[training_data._batch_index-1] # retrieve indices for Hydrant
                if training_data._batch_index == len(training_data._batches): # last batch, potentially wrapping! shrink X, Y and _idcs
                    n_exp, bsize = X0[i:j].shape[0], X.shape[0]
                    X = X[(bsize-n_exp):bsize]
                    Y = Y[(bsize-n_exp):bsize]
                    _idc = _idc[(bsize-n_exp):bsize]
                X0[i:j] = self.transform.transform(X, indices=_idc).to(self.device)
                Y0[i:j] = binarize(Y, k)
                i = j

            # scale features
            if self.X_scaler is None:
                self.X_scaler = Scaler(num_values=p, device=self.device, dtype=X0.dtype)
                self.Y_scaler = Scaler(num_values=p, device=self.device, dtype=Y0.dtype, with_std=False)
            X0 = self.X_scaler.fit_transfrom(X0)
            Y0 = self.Y_scaler.fit_transfrom(Y0)
            self.B0 = self.Y_scaler._mean.to(self.device)
            
            # Ridge regression shortcut (via eigendecomposition) Tew et al. @NeurIPS 2023
            # print('FULL FIT', n, p, X0.shape)
            S2, U = torch.linalg.eigh((X0 @ X0.T))
            S2 = S2.clip(EPS)
            S = S2.sqrt()
            V = (X0.T @ U) * (1 / S)
            R = U * S
            R2 = R ** 2
            RTY = R.T @ Y0
            best_alpha_hat = None
            best_error = np.inf
            for lambda_ in self.lambdas * np.sqrt(n):
                alpha_hat = (1 / (S2[:, None] + lambda_)) * RTY
                Y_hat = R @ alpha_hat
                E = Y0 - Y_hat
                diag_H = (R2 / (S2 + lambda_)).sum(1)
                E_loocv = E / (1 - diag_H[:, None]).clip(EPS)
                err_lambda = (E_loocv ** 2).mean()
                if err_lambda < best_error:
                    best_error = err_lambda
                    best_alpha_hat = alpha_hat

            self.B = V @ best_alpha_hat

        else: # n >= p => memory-efficient fitting, estimating the LOOCV error with a validation set - see Dempster et al. @ AALTD 2024

            # split data
            TR, VA = stratified_split(training_data.Y, val_size, self.seed) # TODO why not use sklearn's stratified splitting logic?
            # TR_SK, VA_SK = train_test_split(np.arange(n), test_size=val_size, stratify=training_data.Y, random_state=self.seed) # sklearn's stratified splitting logic for sanity check
            
            TR, VA = np.sort(TR), np.sort(VA)
            training_data_1, validation_data = training_data[TR], training_data[VA]
            n1, n2 = training_data_1.shape[0], validation_data.shape[0]

            # First pass: compute mean using Welford's algorithm (numerically stable)
            mean_X = torch.zeros(p, device=self.device, dtype=torch.float64)
            mean_Y = torch.zeros(k, device=self.device, dtype=torch.float64)
            count = 0
            
            for X, Y in tqdm(training_data_1, total=np.ceil(n1/training_data_1.batch_size), desc="Ridge 1st pass (Welford's algorithm)"):
                _idc = training_data_1._batches[training_data_1._batch_index-1] # retrieve indices for Hydrant
                _X = self.transform.transform(X, indices=_idc).to(device=self.device, dtype=torch.float64)
                _Y = binarize(Y, k).to(device=self.device, dtype=torch.float64)
                batch_size = _X.shape[0]
                
                # Welford update for mean
                for i in range(batch_size):
                    count += 1
                    delta_X = _X[i] - mean_X
                    mean_X += delta_X / count
                    delta_Y = _Y[i] - mean_Y
                    mean_Y += delta_Y / count
            
            # Initialize scaler with computed mean
            if self.X_scaler is None:
                self.X_scaler = Scaler(num_values=p, device=self.device, dtype=torch.float32)
                self.Y_scaler = Scaler(num_values=k, device=self.device, dtype=torch.float32, with_std=False)
            
            self.X_scaler._mean = mean_X.float()
            self.Y_scaler._mean = mean_Y.float()
            self.B0 = self.Y_scaler._mean.to(self.device)
            
            # Second pass: compute gram matrix with stable scaling
            XTX = torch.zeros((p, p), device=self.device, dtype=torch.float64)
            XTY = torch.zeros((p, k), device=self.device, dtype=torch.float64)
            
            for X, Y in tqdm(training_data_1, total=np.ceil(n1/training_data_1.batch_size), desc="Ridge 2nd pass (gram matrix with scaling)"):
                _idc = training_data_1._batches[training_data_1._batch_index-1] # retrieve indices for Hydrant
                _X = self.transform.transform(X, indices=_idc).to(device=self.device, dtype=torch.float64)
                _Y = binarize(Y, k).to(device=self.device, dtype=torch.float64)
                
                # Scale using the stable mean computed in first pass
                _X_scaled = self.X_scaler.scale(_X)
                _Y_scaled = self.Y_scaler.scale(_Y)
                
                XTX += _X_scaled.T @ _X_scaled
                XTY += _X_scaled.T @ _Y_scaled

            # XTX and XTY are now already scaled, now calculate eigenvalue decomp
            # print('BATCHED FIT', n, p, XTX.shape)
            S2, V = torch.linalg.eigh(XTX.to(self.device))
            S2 = S2.clip(EPS)
            # calculate validation transformation features
            XV, YV = torch.zeros((n2, p), device=self.device), torch.zeros(n2, dtype=torch.int64, device=self.device)
            i = 0
            for i, (X, Y) in enumerate(tqdm(validation_data, desc="Ridge 3rd passt (LOOCV validation)")):
                j = i + X.shape[0]
                _idc = validation_data._batches[validation_data._batch_index-1] # retrieve indices for Hydrant
                X_ = self.transform.transform(X, indices=_idc).to(device=self.device, dtype=torch.float64)
                XV[i:j] = self.X_scaler.scale(X_)
                YV[i:j] = torch.tensor(Y, dtype=torch.int64, device=self.device)
                i = j

            # perform ridge regression with LOOCV on validation set
            best_error = np.inf
            for lambda_ in self.lambdas * np.sqrt(n1):
                _XTXi = (V * (1 / (S2 + lambda_))) @ V.T
                _B = (_XTXi @ XTY).float()
                err_lambda = (YV != ((XV @ _B) + self.B0).argmax(-1)).float().mean()
                if err_lambda < best_error:
                    best_error = err_lambda
                    self.B = _B.clone()

            # check reproducibility of fitting the ridge regression
            # for arr, fname in zip([XTX, XTY, S2, V, XV, YV, self.B], ['XTX', 'XTY', 'S2', 'V', 'XV', 'YV', 'B']):
                # torch.save(arr, f"{fname}.pt")
                # tmp = torch.load(f"{fname}.pt")
                # assert torch.allclose(arr, tmp, atol=1e-05)

            # delete temporary data sets
            validation_data.close()
            training_data_1.close()

    def ft_imp_coeffs(self):
        return torch.mean(torch.abs(self.B), dim=1)

    def save_to_disk(self, path):
        for attr_name in ['transform', 'B0', 'B', 'X_scaler', 'Y_scaler']:
            if hasattr(getattr(self, attr_name), 'state_dict'):
                torch.save(getattr(self, attr_name).state_dict(), os.path.join(path, f"{attr_name}.pth"))
                # for testing reproducibility, check the similarity with a previously stored model:
                # for key, tensor in getattr(self, attr_name).state_dict().items():
                #     tmp = torch.load(PATH mlruns/0/a14c5a64e91b47d0b96257693207f363/artifacts', f'{attr_name}.pth'))
                #     assert torch.allclose(tensor, tmp[key])
            else:
                torch.save(getattr(self, attr_name), os.path.join(path, f"{attr_name}.pt"))
                # for testing reproducibility, check the similarity with a previously stored model:
                # tmp = torch.load(os.path.join('PATH mlruns/0/a14c5a64e91b47d0b96257693207f363/artifacts', f'{attr_name}.pt'))
                # assert torch.allclose(getattr(self, attr_name), tmp)
        fsizes = [os.path.getsize(os.path.join(path, fname)) for fname in ['X_scaler.pth', 'Y_scaler.pth', 'transform.pth', 'B.pt', 'B0.pt']]
        return sum(fsizes)
        
    def load_from_disk(self, path):
        for attr_name in ['transform', 'B0', 'B', 'X_scaler', 'Y_scaler']:
            if attr_name == 'X_scaler' and self.X_scaler is None: # init scalers based on loaded info from transform and B / B0
                self.X_scaler = Scaler(num_values=self.transform.num_features, device=self.device, dtype=torch.float32)
                self.Y_scaler = Scaler(num_values=self.B.shape[1], device=self.device, dtype=torch.float32, with_std=False)
            if hasattr(getattr(self, attr_name), 'load_state_dict'):
                getattr(self, attr_name).load_state_dict(torch.load(os.path.join(path, f"{attr_name}.pth"), map_location=self.device))
            else:
                setattr(self, attr_name, torch.load(os.path.join(path, f"{attr_name}.pt"), map_location=self.device))
        self.X_scaler.to(self.device).eval()
        self.transform.to(self.device).eval()
        fsizes = [os.path.getsize(os.path.join(path, fname)) for fname in ['X_scaler.pth', 'transform.pth', 'B.pt', 'B0.pt']]
        return sum(fsizes)

    def count_params(self):
        p_transf = self.transform.number_of_trained_parameters
        p_scaler = sum([ s._mean.numel() + s._std.numel() + s._count.numel() + s._eps.numel() for s in [self.X_scaler, self.Y_scaler]])
        return p_transf + p_scaler + self.B.numel() + self.B0.numel()
    
    def _predict_single(self, X):
        _X = self.transform.transform(X).to(self.device)
        _X = self.X_scaler.scale(_X)
        res = _X @ self.B + self.B0
        return res

    def _predict(self, test_data, **kwargs):

        n, k = test_data.shape[0], kwargs.get("num_classes", len(test_data.classes))

        # new code with correct batching
        Y0 = torch.zeros((n, k), device=self.device)
        i = 0
        for X, _ in tqdm(test_data, total=np.ceil(n/test_data.batch_size)):
            j = i + X.shape[0]
            Y0[i:j,:] = self._predict_single(X)
            i = j
        
        return Y0
