import argparse
import json
import os

import pandas as pd

BATCH_SIZE_FILE = os.path.join(os.path.dirname(__file__), 'batch_sizes.json')
BATCH_SIZE_STATS = os.path.join(os.path.dirname(__file__), 'batch_size_stats.json')

def lookup_batch_size(dataset, model, architecture, default=32):
    if 'ant' in model: # use much larger default batch sizes for Quant and Hydrant
        default = 16384 # 4096 8192 16384 32768 65536
    try:
        with open(BATCH_SIZE_STATS, 'r') as bf:
            batch_sizes = json.load(bf)
            default = batch_sizes[architecture][dataset][model]["ideal"]
            print(f"Found and using batch size {default} for {architecture} {dataset} {model} configuration")
    except (FileNotFoundError, KeyError) as e:
        print(f"No batch size stored for {architecture} {dataset} {model} configuration, so using default ({default})")
    return default

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Extract batch sizes from experiment logs")
    parser.add_argument("--file", default="ts_archive_experiment_0_2025-11-07_17-26-57.csv")
    args = parser.parse_args()

    if os.path.isfile(BATCH_SIZE_STATS):
        with open(BATCH_SIZE_STATS, 'r') as bf:
            batch_sizes = json.load(bf)
    else:
        batch_sizes = {}

    energy, time, i_energy, i_time, batch_size = "metrics.train_energy_per_epoch", "metrics.train_time_per_epoch", "metrics.time_per_sample", "metrics.energy_per_sample", "params.batch_size"

    data = pd.read_csv(args.file, index_col=False)
    data = data[data["status"] == "FINISHED"]
    for (arch, ds, mod), data in data.groupby(["params.architecture", "params.dataset", "params.model"]):
        if arch not in batch_sizes:
            batch_sizes[arch] = {} 
        if ds not in batch_sizes[arch]:
            batch_sizes[arch][ds] = {}
        data = data.groupby(batch_size)[[time, energy, i_energy, i_time, 'params.n_epochs']].mean().sort_values(time)
        optimal, worst, optimal_b, worst_b = data.iloc[0], data.iloc[-1], data.index[0], data.index[-1]
        batch_sizes[arch][ds][mod] = {"ideal": int(optimal_b)}
        for bs, row in data.sort_values(batch_size).iterrows():
            batch_sizes[arch][ds][mod][int(bs)] = {field: row[field] for field in [energy, time, i_energy, i_time]}
        print(f"{arch:<30} {ds:<30} {mod:<15} - {optimal[energy]/3600:6.2f} Wh (bs {optimal_b:<3}) - {worst[energy]/3600:6.2f} Wh (bs {worst_b:<3}) - {data.shape[0]} results total, trained for {data.iloc[0]['params.n_epochs']} epochs")

    with open(BATCH_SIZE_STATS, 'w') as bf:
        json.dump(batch_sizes, bf, indent=2)
