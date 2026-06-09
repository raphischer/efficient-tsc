import pandas as pd
import numpy as np

from run_analysis import prepare_df

old = prepare_df(pd.read_csv('results/tsc_hybrid_0_2026-05-19_16-31-21.csv', index_col=False), average_folds=False)

fields = ['model', 'dataset', 'environment', 'fold']
# for values, row in new.groupby(fields):
#     old_row = old
#     for field, value in zip(fields, values):
#         old_row = old_row[old_row[field] == value]
#     assert len(old_row) == 1, f"Expected exactly one matching row in old for {values}, but got {len(old_row)}"
#     old_row = old_row.iloc[0]
    
#     res = {}
#     for metric in ['bal_acc', 'gt_train_energy_total']:
#         new_value = row[metric].mean()
#         old_value = old_row[metric]
#         res[metric] = (new_value, (new_value - old_value) / old_value * 100)
#     print(f"{' '.join([f'{str(v)[:5]:<5}' for v in values])} - {' - '.join([f'{m}: {v:.5f} ({c:.2f}%)' for m, (v, c) in res.items()])}")

rel_diffs = []
for values, row in old.groupby(fields):
    cc = row['train_energy_total'].mean()
    gt = row['gt_train_energy_total'].mean() / 3.6e6
    rel_diff = (gt-cc)/gt*100
    print(f"{' '.join([f'{str(v)[:8]:<8}' for v in values])} - Train Energy: {cc:.5f}, GT Energy: {gt:.5f}, DIFF: {gt-cc:.5f}, %DIFF: {rel_diff:.5f}%)")
    rel_diffs.append(rel_diff)

print(f"Average relative difference: {np.mean(rel_diffs):.5f}% with std: {np.std(rel_diffs):.5f}%")