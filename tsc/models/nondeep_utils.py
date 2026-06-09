# based on original AALTD 2024 implementation by Angus Dempster, Daniel F Schmidt, Geoffrey I Webb
# Highly Scalable Time Series Classification for Very Large Datasets @ AALTD 2024 (ECML PKDD 2024)
# https://github.com/angus924/aaltd2024

# adaptions and improvements by raphischer

import numpy as np
import torch

# == Dataset ===================================================================

class Dataset():

    def __init__(self, path_X, path_Y, batch_size = 256, shuffle = True, **kwargs):

        self.path_X = path_X
        self.path_Y = path_Y

        self.batch_size = batch_size

        self._shuffle = shuffle

        self._mmap_X = np.load(path_X, mmap_mode = "r")
        self._mmap_Y = np.load(path_Y, mmap_mode = "r")

        self._indices = kwargs.get("indices", torch.arange(self._mmap_X.shape[0]))

        self.is_open = True

        self._num_batches = self._mmap_X.shape[0] // batch_size + 1

        # self._reset()

    def __getitem__(self, key):

        return \
            Dataset(
                path_X     = self.path_X,
                path_Y     = self.path_Y,
                batch_size = self.batch_size,
                shuffle    = self._shuffle,
                indices    = self._indices[key],
            )
    
    def open(self):

        if not self.is_open:

            self._mmap_X = np.load(self.path_X, mmap_mode = "r")
            self._mmap_Y = np.load(self.path_Y, mmap_mode = "r")

            self.is_open = True

    def close(self):

        if self.is_open:

            self._mmap_X._mmap.close()
            self._mmap_Y._mmap.close()

            del self._mmap_X
            del self._mmap_Y
            
            self._mmap_X = None
            self._mmap_Y = None

            self.is_open = False

    @property
    def classes(self):

        self.open()

        return np.unique(self._mmap_Y[self._indices])

    @property
    def shape(self):

        self.open()
        
        return self._indices.shape[0], *self._mmap_X.shape[1:]
    
    def _reset(self):

        if self._shuffle:
            _batches = torch.randperm(self._indices.shape[0])
        else:
            _batches = torch.arange(self._indices.shape[0])

        self._batches = _batches.split(self.batch_size)

        self._num_batches = len(self._batches)
        self._batch_index = 0

    def __iter__(self):

        self._reset()

        return self

    def __next__(self):

        self.open()

        if self._batch_index < self._num_batches:

            X = self._mmap_X[self._indices[self._batches[self._batch_index]]]
            Y = self._mmap_Y[self._indices[self._batches[self._batch_index]]]

            if X.ndim < self._mmap_X.ndim:
                X = X.reshape(1, *X.shape)
                Y = np.atleast_1d(Y)

            self._batch_index += 1

            return X, Y
        
        else:
            
            raise StopIteration

    @property
    def Y(self):

        self.open()

        return self._mmap_Y[self._indices]
    
    @property
    def X(self):

        self.open()

        return self._mmap_X[self._indices]

class BatchDataset():

    def __init__(self, path_X, path_Y, batch_size=256, shuffle=True, seed=None, **kwargs):

        self.path_X = path_X
        self.path_Y = path_Y

        self.batch_size = batch_size

        self._shuffle = shuffle
        self.seed = seed if seed >= 0 else np.random.randint(1e12)

        self._mmap_X = np.load(path_X, mmap_mode = "r")
        self._mmap_Y = np.load(path_Y, mmap_mode = "r")

        self._indices = kwargs.get("indices", torch.arange(self._mmap_X.shape[0]))

        self.is_open = True

        self._num_batches = self._mmap_X.shape[0] // batch_size + 1

        # self._reset()

    def __getitem__(self, key):

        return \
            BatchDataset(
                path_X     = self.path_X,
                path_Y     = self.path_Y,
                batch_size = self.batch_size,
                shuffle    = self._shuffle,
                indices    = self._indices[key],
                seed       = self.seed
            )
    
    def open(self):

        if not self.is_open:

            self._mmap_X = np.load(self.path_X, mmap_mode = "r")
            self._mmap_Y = np.load(self.path_Y, mmap_mode = "r")

            self.is_open = True

    def close(self):

        if self.is_open:

            self._mmap_X._mmap.close()
            self._mmap_Y._mmap.close()

            del self._mmap_X
            del self._mmap_Y
            
            self._mmap_X = None
            self._mmap_Y = None

            self.is_open = False

    @property
    def classes(self):

        self.open()

        return np.unique(self._mmap_Y[self._indices])

    @property
    def shape(self):

        self.open()
        
        return self._indices.shape[0], *self._mmap_X.shape[1:]

    def _reset(self):

        rng = np.random.default_rng(self.seed) # enfore determistic behavior over multiple iterations / resets

        Y = torch.tensor(self.Y, dtype = torch.int64)
        self_U = Y.unique()
        self_Y_map = {y.item() : torch.atleast_1d((Y == y).nonzero().squeeze()) for y in self_U}
        num_classes = len(Y.unique())
        
        _missing = False
        _performed_replacement = False

        if self._shuffle:
            _batches = torch.randperm(self._indices.shape[0])
        else:
            _batches = torch.arange(self._indices.shape[0])
        
        # ======================================================================

        n = _batches.shape[0]
        
        max_size = self.batch_size
        
        div = n / max_size
        
        num_blocks = int(np.ceil(div))
        
        total = num_blocks * max_size
        
        surplus = total - n

        if num_blocks > 1:
            step = int(max_size - int(surplus / (num_blocks - 1)))
        else:
            step = max_size
        
        self._batches = []
                
        for i in range(num_blocks):
        
            a = i * step
            b = min(a + max_size, n)

            _batch = _batches[a:b]
            
            Y_batch = Y[_batch]
            U_batch, C_batch = Y_batch.unique(return_counts = True)
        
            Y_missing = np.setdiff1d(self_U, U_batch)

            if len(Y_missing) > 0:
                _missing = True

            self._batches.append(_batch)

        _num_batches = len(self._batches)

        if _missing:

            _supp = torch.zeros((_num_batches, num_classes), dtype = torch.int64)

            for i, y in enumerate(self_U):

                _supp[:, i] = torch.tensor(rng.choice(self_Y_map[y.item()], _num_batches), dtype = torch.int64)
            
            for i, _batch in enumerate(self._batches):

                I = rng.choice(_batch.shape[-1], min(num_classes, _batch.shape[-1]), replace = False)

                _batch[I] = _supp[i, :len(I)]

            _performed_replacement = True

        for _batch in self._batches:

            _batch = _batch.sort()[0]

        self._num_batches = len(self._batches)
        self._batch_index = 0

    def __iter__(self):

        self._reset()

        return self

    def __next__(self):

        self.open()

        if self._batch_index < self._num_batches:

            X = self._mmap_X[self._indices[self._batches[self._batch_index]]]
            Y = self._mmap_Y[self._indices[self._batches[self._batch_index]]]

            if X.ndim < self._mmap_X.ndim:
                X = X.reshape(1, *X.shape)
                Y = np.atleast_1d(Y)

            self._batch_index += 1

            return X, Y
        
        else:
            
            raise StopIteration

    @property
    def Y(self):

        self.open()

        return self._mmap_Y[self._indices]
    
    @property
    def X(self):

        self.open()

        return self._mmap_X[self._indices]

    def unbatch(self):

        return \
            Dataset(
                path_X     = self.path_X,
                path_Y     = self.path_Y,
                batch_size = self.batch_size,
                shuffle    = self._shuffle,
                indices    = self._indices,
            )
    
    def set_batch_size(self, limit_mb=100):

        mb_est = np.int64(np.prod(self.shape)) * 4 * 1e-6

        self.batch_size = min(self.shape[0], int(self.shape[0] / (mb_est / limit_mb)))

        self._reset()