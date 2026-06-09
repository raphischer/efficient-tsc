import argparse
import os
import sys
import platform
import re
import subprocess
import random
import traceback

import numpy as np
import pandas as pd
from lamarr_energy_tracker.ground_truth_tracking import GroundTruthTracker
from lamarr_energy_tracker.tracker import EnergyTracker
import mlflow
import pynvml

from models.models import init_train
from data_loading import load
from hyperparams import lookup_batch_size


def initialization(args):
    assert args.prune_rate >= 0 and args.prune_rate <= 1, 'Pruning rate must be between 0 and 1 (the higher the value, the more features are pruned for deployment)!'
    # seeding
    config = args.__dict__
    if args.seed > 0:
        config['rng'] = np.random.default_rng(config['seed'])
        random.seed(args.seed)
    else:
        config['rng'] = np.random.default_rng()
        random.seed(int(config['rng'].integers(1e9)))
        print('WARNING: Passed a negative seed, which disables fixed seeding. Algorithms might run more efficient however will obtain non-deterministic results!')
    
    if int(config['gpu']) < 0: # disable GPU for TF & Torch and store CPU name
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if platform.system() == "Windows":
            config['architecture'] = platform.processor()
        elif platform.system() == "Linux":
            command = "cat /proc/cpuinfo"
            all_info = subprocess.check_output(command, shell=True).strip().decode('ascii')
            for line in all_info.split("\n"):
                if "model name" in line:
                    config['architecture'] = re.sub( ".*model name.*:", "", line,1).strip()
        elif platform.system() == "Darwin":
            config['architecture'] = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).strip().decode('ascii')
    else: # use only first GPU and lookup name
        assert str(config['gpu']) in "0 1 2 3 4 5 6 7", "ERROR: GPU index must be between 0 and 7"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu'])
        pynvml.nvmlInit()
        config['architecture'] = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(int(config['gpu'])))
    # fix output path
    mlflow.start_run()
    config['output_dir'] = mlflow.get_artifact_uri().replace('file://', '')
    if not os.path.isdir(config['output_dir']):
        raise RuntimeError(f'ERROR: output_dir does not exist (should be at {config["output_dir"]})')
    # check batch size
    if config['batch_size'] == -1:
        config['batch_size'] = lookup_batch_size(config['dataset'], config['model'], config['architecture'])
    return config


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Perform inference with a model and few batches of data")
    # data configuration
    parser.add_argument("--dataset", default="WISDM", help="Name of the MONSTER dataset to use")
    parser.add_argument("--cache_dir", default=None, help="Directory to use for caching datasets")
    parser.add_argument("--fold", type=int, choices=[0, 1, 2, 3, 4], default=0, help="Pre-defined MONSTER dataset fold")
    # model and system
    parser.add_argument("--model", default="Hydra", choices=["Hydrant", "HydrantNaive", "Quant", "Hydra", "ConvTran", "FCN", "InceptionTime", "LSTMFCN", "MCDCNN", "MLP", "ResNet"], help='TSC model to run.')
    parser.add_argument('--batch_size', type=int, default=-1, help='Batch size for training and inference, pass -1 for using default')
    parser.add_argument('--gpu', type=int, default='-1', help='GPU index, pass -1 for CPU-only')
    parser.add_argument('--seed', default=1234, type=int, help='Randomization seed, pass -1 for non-deterministic but more efficient behavior')
    parser.add_argument("--use_pretrained", default='', help='Pass a csv summary that lists previous training results, from which the correct model will be loaded')
    parser.add_argument("--discard_model", type=bool, default=False, help="Activate to not store the trained model in the MLflow output directory")
    parser.add_argument('--subsample', type=float, default=None, help='Subsample train/test to this fraction (smoke testing)')

    # model hyperparameters
    parser.add_argument('--prune_rate', type=float, default=0, help='Pruning rate for HYDRA, QUANT and HYDRANT')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of OOF folds for HYDRANT')
    parser.add_argument("--prune_intermediate", type=str, choices=['ridge', 'xrf', 'rf'], default='ridge', help="Whether to use ridge regression or XRF as intermediate model for obtaining feature importance")
    # QUANT hyperparams
    parser.add_argument('--classifier', type=str, default="xrf", choices=['rf', 'xrf'], help='Classifier type for Quant: XRF or RF')
    parser.add_argument('--num_estimators', type=int, default=200, help='Number of estimators for Quant classifier')
    parser.add_argument('--max_depth', type=int, default=20, help='Maximum depth of trees for Quant classifier')
    parser.add_argument('--max_features', type=float, default=0.1, help='Maximum features for Quant classifier')
    parser.add_argument('--criterion', type=str, default="entropy", help='Criterion for Quant randomforest')
    # HYDRA hyperparams
    parser.add_argument('--num_kernels_per_group', type=int, default=8, help='Number of random kernels within each group')
    parser.add_argument('--num_groups', type=int, default=64, help='Number of groups for random kernels')
    parser.add_argument('--max_num_channels', type=int, default=8, help='Number of time series channels for each group')
    parser.add_argument('--kernel_length', type=int, default=9, help='Length of the random kernels')

    ############## deep learning hyperparams
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.01, help='Dropout regularization ratio')
    
    # initialization and data loading
    args = parser.parse_args()
    config = initialization(args)
    data = load(config, subsample=config['subsample']) # this adds some extra config entries
    for key, val in config.items(): # log config information
        if val is not None and (isinstance(val, str) or isinstance(val, int) or isinstance(val, float)):
            mlflow.log_param(key, val)
        else:
            print('WARNING: Not logging parameter %s, because it is not a string, int or float' % key)    

    try:
        # TRAINING
        model, train_func, init_evaluate = init_train(config, data)
        mlflow.log_param('software', config['software']) # is only set after the initialization
        gt_tracker = GroundTruthTracker(verbose=False, crash_if_unavailable=False) # track ground-truth energy (if available)
        tracker = EnergyTracker()
        gt_tracker.start()
        tracker.start()
        if config['use_pretrained']:
            if config['use_pretrained'].endswith('.csv'): # check for path where the original model was stored (given in mlflow experiments summary)
                if not os.path.isdir(config['use_pretrained']):
                    config['use_pretrained'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), config['use_pretrained'])
                results = pd.read_csv(config['use_pretrained'])
                crit = (results['status'] == 'FINISHED') & (results['params.model'] == config['model']) & (results['params.dataset'] == config['dataset']) & (results['params.fold'] == config['fold'])
                pcrit = config['model'] not in ['Hydra', 'Quant', 'Hydrant', 'HydrantNaive'] or results['params.prune_rate'] == config['prune_rate']
                c_res = results[crit & pcrit]
                assert (c_res.shape[0] > 0), f"ERROR: No pretrained results found for {config['model']} on {config['dataset']} with fold {config['fold']} - n results={c_res.shape[0]}"
                if c_res.shape[0] > 1:
                    print(f"WARNING: Multiple pretrained results found for {config['model']} on {config['dataset']} with fold {config['fold']}, using first model of {c_res.shape[0]} trained models")
                config['use_pretrained'] = c_res['params.output_dir'].values[0]
            assert os.path.isdir(config['use_pretrained'])
            print(f"Skipping training, loaded pretrained model from {config['use_pretrained']}")
        else:
            train_func()
        train_cc = tracker.stop(print_summary=False)
        train_gt = gt_tracker.stop()

        # INFERENCE
        evaluator = init_evaluate(config, data, model)
        tracker = EnergyTracker() # needs to be re-initialized, otherwise returned data doesn't align
        gt_tracker.start()
        tracker.start()
        results, _ = evaluator()
        evaluator() # run eval twice, for longer inference time / more stable results
        infer_cc = tracker.stop(print_summary=False)
        infer_gt = gt_tracker.stop()

        # assess parameter count
        try:
            if hasattr(model, 'count_params'): # Hydra & Quant & Variants
                results['parameters'] = model.count_params()
            elif hasattr(model, 'parameters'): # Torch models
                results['parameters'] = sum(p.numel() for p in model.parameters())
            elif hasattr(model, 'model_') and hasattr(model.model_, 'count_params'): # SKTIME
                results['parameters'] = model.model_.count_params()
            else:
                raise NotImplementedError(f"Could not count model parameters for {config['model']}")
        except Exception as e:
            print('ERROR when assessing parameters:\n', e)
            results['parameters'] = -1

        # store resource consumption data
        results['train_time_total'] = train_cc['duration']
        results['train_energy_total'] = train_cc['energy_consumed'] * 3.6e6 # kwh to ws
        results['infer_time_total'] = infer_cc['duration'] / 2 # two eval iterations
        results['infer_energy_total'] = infer_cc['energy_consumed'] * 3.6e6 / 2 # kwh to ws, two eval iterations
        results['gt_train_time_total'] = train_gt['duration']
        results['gt_train_energy_total'] = train_gt['energy_consumed'] * 3.6e6 # kwh to ws
        results['gt_infer_time_total'] = infer_gt['duration']
        results['gt_infer_energy_total'] = infer_gt['energy_consumed'] * 3.6e6 / 2 # kwh to ws, two eval iterations
        results['time_per_sample'] = results['infer_time_total'] / data["test_data"].shape[0]
        results['energy_per_sample'] = results['infer_energy_total'] / data["test_data"].shape[0]

        print(f"CC: {results['train_energy_total']:4.2f} Ws   GT: {results['gt_train_energy_total']:4.2f} Ws")

        # log results
        for key, val in results.items():
            if val is not None:
                mlflow.log_metric(key, val)

        print(f'{args.model} on {args.dataset} (fold {args.fold}) - params: {results["parameters"]/1000:4.1f}k - fit time: {results["train_time_total"]:.2f} s  -  inf time: {results["infer_time_total"]:.2f} s  -  acc: {results["accuracy"]*100:.2f} %  -  results at {os.path.dirname(config["output_dir"])}')
        
        mlflow.end_run()
        sys.exit(0)
    except Exception as e:
        # store traceback information as an error artifact
        tb_str = traceback.format_exc()
        error_file = "error_traceback.txt"
        with open(error_file, "w") as f:
            f.write(tb_str)
        mlflow.log_artifact(error_file)
        os.remove(error_file)
        mlflow.set_tag("error", str(e))
        mlflow.end_run('FAILED')
        print(e)
        sys.exit(1)
