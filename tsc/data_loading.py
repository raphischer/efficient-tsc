import argparse

import numpy as np
from sklearn import model_selection
from huggingface_hub import HfApi, hf_hub_download

from models.nondeep_utils import BatchDataset # for hydra and quant

api = HfApi()
DATASETS = sorted([ds.id.replace("monster-monash/", "") for ds in api.list_datasets(author="monster-monash")])

def load(config, subsample=None):
    ds = config["dataset"]
    repo_id = f"monster-monash/{ds}"

    # Download data
    data_path = hf_hub_download(repo_id=repo_id, filename=f"{ds}_X.npy", repo_type="dataset", cache_dir=config["cache_dir"])

    # Download labels
    label_filename = f"{ds}_Y.npy"
    try:
        label_path = hf_hub_download(repo_id=repo_id, filename=label_filename, repo_type="dataset")
    except:
        label_filename = f"{ds}_y.npy"
        label_path = hf_hub_download(repo_id=repo_id, filename=label_filename, repo_type="dataset")

    # Load test indices
    test_index_path = hf_hub_download(repo_id=repo_id, filename=f"test_indices_fold_{config['fold']}.txt", repo_type="dataset")
    test_index = np.loadtxt(test_index_path, dtype=int)

    if "Hydra" in config["model"] or 'Quant' in config["model"]: # load special memory-mapping dataset
        ds = BatchDataset(data_path, label_path, batch_size=config['batch_size'], shuffle=False, seed=config['seed'])
        train_index = np.setdiff1d(np.arange(ds.shape[0]), test_index)
        if subsample is not None:
            assert isinstance(subsample, float) and 0 < subsample < 1, "subsample should be a float between 0 and 1"
            # Stratify so every class is represented; RidgeClassifier uses labels as scatter indices.
            label_npy = np.load(label_path, mmap_mode="r")
            seed = config['seed'] if config['seed'] >= 0 else None
            def stratify(idx, floor):
                target = max(int(subsample * len(idx)), floor)
                splitter = model_selection.StratifiedShuffleSplit(1, train_size=target, random_state=seed)
                sub, _ = next(splitter.split(np.zeros(len(idx)), label_npy[idx]))
                return idx[sub]
            train_index, test_index = stratify(train_index, 100), stratify(test_index, 50)
        data = {'train_data': ds[train_index], 'test_data': ds[test_index].unbatch()}
        config['labels'] = data['train_data'].classes
        ds.close()
    else:
        data_npy = np.load(data_path, mmap_mode="r")  # (#Samples, #Channel, #Length)
        label_npy = np.load(label_path)
        if subsample is not None:
            assert isinstance(subsample, float) and 0 < subsample < 1, "subsample should be a float between 0 and 1"
            
            sel_s = config['rng'].choice(data_npy.shape[0], max(int(subsample * data_npy.shape[0]), 5), replace=False)
            sel_c = config['rng'].choice(data_npy.shape[1], max(int(subsample * data_npy.shape[1]), 2), replace=False)
            sel_l = config['rng'].choice(data_npy.shape[2], max(int(subsample * data_npy.shape[2]), 2), replace=False)
            data_npy = data_npy[sel_s][:,sel_c][:,:,sel_l]
            label_npy = label_npy[sel_s]
            test_index = np.concatenate([np.where(sel_s == idx)[0] for idx in test_index])
        data = split_data(data_npy, label_npy, test_index, None if config['seed'] < 0 else config['seed'])
        config['labels'] = np.unique(label_npy)
    # store some additional config information
    config['n_labels'] = config['labels'].size
    config['n_train_samples'] = data['train_data'].shape[0]
    config['n_samples'] = data['train_data'].shape[0] + data['test_data'].shape[0]
    config['n_channels'] = data['train_data'].shape[1]
    config['length'] = data['train_data'].shape[2]
    return data


def split_data(Data_npy, Label_npy, test_index, random_state=None):

    # Create a boolean array indicating the samples designated for the test set
    test_bool_index = np.zeros(len(Label_npy), dtype=bool)
    test_bool_index[test_index] = True

    Data = {'test_data': Data_npy[test_index], 'test_label': Label_npy[test_index],
            'All_train_data': Data_npy[~test_bool_index], 'All_train_label': Label_npy[~test_bool_index]}

    Data['train_data'], Data['train_label'], Data['val_data'], Data['val_label'] = non_subject_wise_split(Data['All_train_data'], Data['All_train_label'], random_state)
    return Data


def non_subject_wise_split(data, label, random_state=None):
    splitter = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=random_state)
    train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(label)), y=label))
    train_data = data[train_indices]
    train_label = label[train_indices]
    val_data = data[val_indices]
    val_label = label[val_indices]

    return train_data, train_label, val_data, val_label

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Load all MONSTER datasets")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--datadir", default="/data/d1/ts_archive/datasets")
    args = parser.parse_args().__dict__
    args["model"] = "FCN"

    success, errors = {}, []
    for ds_name in DATASETS:
        try:
            args["dataset"] = ds_name
            ds = load(args)
            success[ds_name] = ds["All_train_data"].shape
            print(f"{ds_name:<25} - data shape: {str(ds['All_train_data'].shape):<20} - n labels: {np.unique(ds['All_train_label']).size}")
        except Exception:
            print('Error with', ds_name)
            errors.append(ds_name)

    print("errors", errors)

    sizes = [(np.prod(np.array(list(vals))), key) for key, vals in success.items()]
    ds_sorted = [mod for _, mod in sorted(sizes)]
    print("DS in growing size:", '"' + '" "'.join(ds_sorted) + '"')
