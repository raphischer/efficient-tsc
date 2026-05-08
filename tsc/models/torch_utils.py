import os
from copy import deepcopy

import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset


def init_torch(config):
    if config['seed'] >= 0:
        torch.manual_seed(config['seed'])
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    if torch.cuda.is_available() and int(config['gpu']) != -1:
        config['device'] = torch.device('cuda')
        config['architecture'] = torch.cuda.get_device_name(config['gpu'])
        if config['seed'] >= 0:
            torch.cuda.manual_seed_all(config['seed'])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        config['device'] = torch.device('cpu')
    config['software'] = 'PyTorch' + ' ' + torch.__version__

class Torch_Dataset(Dataset):

    def __init__(self, data, label):
        super(Torch_Dataset, self).__init__()

        self.feature = data
        self.labels = label.astype(np.int32)
        # self.__padding__()

    def __padding__(self):
        origin_len = self.feature[0].shape[1]
        if origin_len % self.patch_size:
            padding_len = self.patch_size - (origin_len % self.patch_size)
            padding = np.zeros((len(self.feature), self.feature[0].shape[0], padding_len), dtype=np.float32)
            self.feature = np.concatenate([self.feature, padding], axis=-1)

    def __getitem__(self, ind):

        x = self.feature[ind]
        y = self.labels[ind]  # (num_labels,) array

        data = torch.tensor(x, dtype=torch.float32)
        label = torch.tensor(y, dtype=torch.int32)

        return data, label, ind

    def __len__(self):
        return len(self.labels)


def get_loss_module():
        return NoFussCrossEntropyLoss(reduction='none')  # outputs loss for each batch sample


def l2_reg_loss(model):
    """Returns the squared L2 norm of output layer of given model"""

    for name, param in model.named_parameters():
        if name == 'output_layer.weight':
            return torch.sum(torch.square(param))


class NoFussCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long(), weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)


def save_model(path, epoch, model, optimizer=None):
    # Ensure the directory exists before saving the model
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)

    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss

    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion, path):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"Best validation loss: {self.best_valid_loss}")
            print(f"Saving best model for epoch: {epoch}\n")
            save_model(path, epoch, model, optimizer)


class SaveBestACCModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, best_valid_acc=float('0')):
        self.best_valid_acc = best_valid_acc

    def __call__(self, current_valid_acc, epoch, model, optimizer, criterion, path):

        if current_valid_acc > self.best_valid_acc:
            self.best_valid_acc = current_valid_acc
            print(f"Best validation acc: {self.best_valid_acc}")
            print(f"Saving best model for epoch: {epoch}\n")
            save_model(path, epoch, model, optimizer)


def load_model(model, model_path, optimizer=None, resume=False, change_output=False,
               lr=None, lr_step=None, lr_factor=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict = deepcopy(checkpoint['state_dict'])
    if change_output:
        for key, val in checkpoint['state_dict'].items():
            if key.startswith('output_layer'):
                state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    print('Loaded model from {}. Epoch: {}'.format(model_path, checkpoint['epoch']))

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for i in range(len(lr_step)):
                if start_epoch >= lr_step[i]:
                    start_lr *= lr_factor[i]
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model, None, None
