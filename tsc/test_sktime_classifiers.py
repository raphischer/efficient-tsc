import argparse
import time

import numpy as np
from sktime.registry import all_estimators
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
ESTIMATORS = all_estimators(estimator_types="classifier", return_names=True)

from models.models import init_train
from data_loading_new import load

print('imports')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform inference with a model and few batches of data")
    parser.add_argument("--dataset", default="WISDM")
    parser.add_argument("--datadir", default="/data/d1/ts_archive/datasets")
    parser.add_argument("--output_path", default="/data/d1/ts_archive/results")
    parser.add_argument("--measure_power_secs", default=1)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument('--gpu', type=int, default='0', help='GPU index, -1 for CPU')
    parser.add_argument('--n_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()

    config = args.__dict__ # initialization(args)
    config['model'] = 'FCNClassifier' # None
    data = load(config, subsample=0.2)

    # print('inits')

    for mname, _ in ESTIMATORS:
        try:
            t0 = time.time()
            config["model"] = mname
            print(f"Testing model: {mname}")
            train_func, init_evaluate = init_train(config, data)
            model, _ = train_func()
            evaluator = init_evaluate(config, data, model)
            results, _ = evaluator()
            res_str = f'Model: {mname}, total_runtime: {time.time()-t0:.2}s, accuracy: {results["accuracy"]*100:4.2f}%'
        except Exception as e:
            res_str = f"Model: {mname} failed with error: {e.split('\n')[0]}"
        print(res_str)
        with open(f"sktime_results.txt", "a") as f:
            f.write(f"{res_str}\n")
