import os
import re

# data handling
from tqdm import tqdm
import numpy as np
import pandas as pd

# plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.colors import sample_colorscale, make_colorscale

# statistical testing
import operator
import math
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare

# scaling
from strep.util import load_meta, lookup_meta
from strep.index_scale import scale_and_rate
from strep.elex.app import Visualization
from strep.elex.graphs import GRAD
from strep.elex.util import hex_to_alpha
from strep.unit_formatter import CustomUnitReformater

def wilcoxon_holm(db, alpha=0.05): # adapted from https://github.com/hfawaz/cd-diagram
    column = [col for col in db.columns if col not in ['model', 'configuration']][0]
    # check for maximum number of tested configs and limit to those model evaluations
    df_counts = pd.DataFrame({'count': db.groupby(['model']).size()}).reset_index()
    max_nb_datasets = df_counts['count'].max()
    classifiers = list(df_counts.loc[df_counts['count'] == max_nb_datasets]['model'])
    assert len(set(classifiers)) == pd.unique(db['model']).size
    # test the null hypothesis using friedman before doing a post-hoc analysis
    friedman_p_value = friedmanchisquare(*(np.array(db.loc[db['model'] == c][column]) for c in classifiers))[1]
    if friedman_p_value >= alpha:
        raise RuntimeError('the null hypothesis over the entire classifiers cannot be rejected')
    # loop through classifiers to assess pairwise statistical significance via Wilcoxon signed rank test
    p_values = []
    for i, clsf_1 in tqdm(enumerate(classifiers[:-1]), total=len(classifiers)-1, desc='compare pairwise'):
        perf_1 = db.loc[db['model'] == clsf_1][column].values
        for clsf_2 in classifiers[(i+1):]:
            perf_2 = db.loc[db['model'] == clsf_2][column].values
            p_value = wilcoxon(perf_1, perf_2, zero_method='pratt')[1]
            p_values.append((clsf_1, clsf_2, p_value, False))
    k = len(p_values)
    # loop over k sorted hypotheses to check for significance
    p_values.sort(key=operator.itemgetter(2))
    for i in tqdm(range(k), desc='checking hypotheses'):
        new_alpha = float(alpha / (k - i)) # correct alpha with holm
        if p_values[i][2] <= new_alpha: # test if significant
            p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)
        else:
            break
    # also compute average ranks (useful for drawing the cd diagram)
    sorted_db = db.loc[db['model'].isin(classifiers)].sort_values(['model', 'configuration'])
    rank_data = np.array(sorted_db[column]).reshape(len(classifiers), max_nb_datasets)
    df_ranks = pd.DataFrame(data=rank_data, index=np.sort(classifiers), columns=np.unique(sorted_db['configuration']))
    average_ranks = df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)
    return p_values, average_ranks, max_nb_datasets

def find_cliques_numpy(adj_matrix: np.ndarray):
    """
    Iterative Bron-Kerbosch-like algorithm (NumPy-only).
    Returns generator yielding maximal cliques as lists of node indices.
    """
    m = adj_matrix.shape[0]
    adj_bool = (adj_matrix != 0)

    # Stack holds tuples of (R_set, P_mask, X_mask)
    stack = [(set(), np.ones(m, dtype=bool), np.zeros(m, dtype=bool))]

    while stack:
        R, P, X = stack.pop()

        # If both P and X are empty → R is maximal clique
        if not np.any(P) and not np.any(X):
            yield list(R)
            continue

        # Iterate through all vertices in P
        for v in np.where(P)[0]:
            Nv = adj_bool[v].astype(bool)
            stack.append((R | {v}, P & Nv, X & Nv))
            P[v] = False
            X[v] = True

def assign_cliques_to_levels(cliques):
    """
    Assign cliques to y-levels such that non-overlapping cliques share the same level.
    Two cliques can be at the same level if their node indices don't intersect.
    Returns a dict mapping level (int) to list of cliques.
    """
    levels = {}
    
    for clq in cliques:
        clq_set = set(clq)
        # Find the first available level where this clique doesn't overlap with others
        level = 0
        while level in levels:
            # Check if clq overlaps with any clique at this level
            overlaps = False
            for existing_clq in levels[level]:
                existing_set = set(existing_clq)
                if clq_set & existing_set:  # intersection is non-empty
                    overlaps = True
                    break
            if not overlaps:
                break
            level += 1
        
        if level not in levels:
            levels[level] = []
        levels[level].append(clq)
    
    return levels

def graph_ranks_plotly(avranks, names, p_values, title='Rank', reverse=False, labels=False, x_padding=2):

    avranks = np.asarray(avranks, dtype=float)
    names = np.asarray(names)
    k = len(avranks)

    # ---------- axis bounds ----------
    lowv = math.floor(min(avranks))
    highv = math.ceil(max(avranks))

    xmin, xmax, ymax = lowv - x_padding, highv + x_padding, math.ceil(k/2)+1.5
    y0, y_line_off, row_step = -0.2, 0.04, 0.6
    fig = go.Figure()

    # ---------- Configure x axis ----------
    fig.update_xaxes(range=[xmax, xmin] if reverse else [xmin, xmax], tickmode="array", tickvals=list(range(lowv - 1, highv + 2)), title=title)
    fig.update_yaxes(range=[-ymax, 0], visible=False)
    fig.add_shape(type="line", x0=lowv-1, x1=highv+1, y0=y0, y1=y0)
    for x in range(lowv-1, highv+2):
        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=y0-y_line_off)


    # ---------- Sort by rank ----------
    order = np.argsort(avranks)
    avranks = avranks[order]
    names = names[order]
    half = math.ceil(k / 2)

    # ====================================================
    # RIGHT SIDE
    # ====================================================
    for i in range(half):
        y, r = - (i + 1) * row_step - row_step, avranks[i]
        fig.add_shape(type="line", x0=r, x1=r, y0=y0, y1=y-y_line_off)
        fig.add_shape(type="line", x0=r, x1=lowv, y0=y, y1=y)
        fig.add_annotation(x=lowv, y=y, text=f"{names[i]}", showarrow=False, xanchor="left")
        if labels:
            fig.add_annotation(x=lowv, y=y + 0.5, text=f"{r:.3f}", xanchor="right", showarrow=False)

    # ====================================================
    # LEFT SIDE
    # ====================================================
    for i in range(half, k):
        y, r = - (k - i) * row_step - row_step, avranks[i]
        fig.add_shape(type="line", x0=r, x1=r, y0=y0, y1=y-y_line_off)
        fig.add_shape(type="line", x0=r, x1=highv, y0=y, y1=y)
        fig.add_annotation(x=highv, y=y, text=f"{names[i]}", showarrow=False, xanchor="right")
        if labels:
            fig.add_annotation(x=highv, y=y + 0.5, text=f"{r:.3f}", showarrow=False, xanchor="left")

    # ====================================================
    # NON SIGNIFICANT CLIQUES
    # ====================================================
    m = len(names)
    g_data = np.zeros((m, m), dtype=np.int64)
    for p in p_values:
        if p[3] == False:
            i = np.where(names == p[0])[0][0]
            j = np.where(names == p[1])[0][0]
            min_i = min(i, j)
            max_j = max(i, j)
            g_data[min_i, max_j] = 1
    g_sym = ((g_data + g_data.T) > 0).astype(np.int64)
    np_cliques = sorted([ sorted(clq) for clq in find_cliques_numpy(g_sym) if len(clq) > 1 ])
    # SANITY CHECK - compare against cliques found via networksx!
    # import networkx
    # nx_cliques = sorted([ sorted(clq) for clq in networkx.find_cliques(networkx.Graph(g_data)) if len(clq) > 1 ])
    # assert len(nx_cliques) == len(np_cliques)

    # plot cliques
    for level, clqs in assign_cliques_to_levels(np_cliques).items():
        for clq in clqs:
            min_idx, max_idx = np.min(clq), np.max(clq)
            fig.add_shape(type="line", x0=avranks[min_idx], x1=avranks[max_idx], y0=y0-row_step-(level*-0.5*row_step), y1=y0-row_step-(level*-0.5*row_step), line=dict(width=5))

    fig.update_layout(showlegend=False, xaxis={'side': 'top'}, plot_bgcolor="white")
    return fig

def get_cov_ellipse(x, y, n_std=2.0, num_points=100):
    """
    Returns coordinates of an ellipse representing n_std standard deviations.
    """
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    
    # Sort eigenvalues and corresponding eigenvectors
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    
    # Compute angle and radii
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    
    # Ellipse parameterization
    t = np.linspace(0, 2*np.pi, num_points)
    ellipse_coords = np.array([width/2 * np.cos(t), height/2 * np.sin(t)])
    
    # Rotation matrix
    R = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                  [np.sin(np.radians(theta)),  np.cos(np.radians(theta))]])
    
    ellipse_rotated = R @ ellipse_coords
    
    return ellipse_rotated[0] + np.mean(x), ellipse_rotated[1] + np.mean(y)

def prepare_df(df, average_folds=True, change_modelname=True):
    df = df[df["status"] == "FINISHED"]
    df = df.drop([col for col in df.columns if "params." not in col and "metrics." not in col], axis=1)
    df = df.rename(lambda col: col.replace("metrics.", "").replace("params.", "").replace('architecture', 'environment'), axis=1)
    if change_modelname:
        df['model'] = df.apply(lambda r: replace_prune_name(r['model'], r['prune_rate']), axis=1) # encode prune info into model name!
    df['environment'] = df['environment'].map(lambda e: f"Intel {re.match(r'.*(i\d-\d*)', e).group(1)}" if 'Intel' in e else e.replace(' GeForce', ''))
    df = df[df['model'] != 'SFCN'] # drop this baseline from the MONSTER paper
    df['train_energy_total'] /= 3.6e6 # convert to kWh
    for col in ['bal_acc', 'accuracy', 'weighted_f1', 'macro_f1', 'micro_f1']:
        df[col] *= 100
    if 'task' not in df.columns:
        df['task'] = 'Unknown'
    for task, task_df in df.groupby('task'):
        print(f'Number of evaluations found for {task}: {task_df.shape[0]}')
    if average_folds: # average individual runs over dataset folds
        grouped_over_folds = df.groupby(['task', 'dataset', 'environment', 'model', 'batch_size'])
        for index, row in grouped_over_folds.count().iterrows():
            if row['train_time_total'] < 5 and index[0] == 'Training':
                print(f"WARNING: Only found {row['train_time_total']} training results across the five splits for {index} configuration - respective inference results will also be missing!")
        fold_averages = grouped_over_folds.mean(numeric_only=True)
        fold_rest = grouped_over_folds.first().drop(columns=fold_averages.columns)
        df = pd.concat([fold_averages, fold_rest], axis=1).reset_index()
    df['prune_rate'] = np.round(df['prune_rate'], 2) # averaging over folds can lead to small differences in prune_rate
    return df

def print_init(fname):
    print(f'\n                 - -- ---  {fname:<30}  --- -- -                 \n')
    return fname

def finalize(fig, fname, show=True, ws=1, hs=1, top=0, bottom=0, yshift=0):
    fig.update_layout(font = dict(color='#000000'), font_family='Open-Sherif', width=PLOT_WIDTH*ws, height=PLOT_HEIGHT*hs, margin={'l': 0, 'r': 0, 'b': bottom, 't': top})
    fig.update_annotations(yshift=2+yshift) # to adapt tex titles
    if show:
        fig.show()
    outname = os.path.join(os.path.dirname(__file__), 'results', 'figures', f"{fname}.pdf")
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    fig.write_image(outname)

def replace_prune_name(model_name, prune_rate):
    if prune_rate == 0 or model_name not in ['Hydra', 'Quant', 'Hydrant']:
        return model_name
    # model_name = 'X' if model_name == 'Hydrant' else model_name[0]
    return f"P{prune_rate*100:.0f}{model_name}"

PLOT_WIDTH = 800
PLOT_HEIGHT = PLOT_WIDTH // 3

COLORS = [
    '#009ee3', #0 aqua
    '#983082', #1 fresh violet
    '#ffbc29', #2 sunshine
    '#35cdb4', #3 carribean
    '#e82e82', #4 fuchsia
    '#59bdf7', #5 sky blue
    '#ec6469', #6 indian red
    '#706f6f', #7 gray
    '#4a4ad8', #8 corn flower
    '#0c122b', #9 dark corn flower
    '#ffffff'
]
COL_SEL = [COLORS[i] for i in [1, 6, 0, 3, 8, 2, 7, 9, 5, 4, 1]]

if __name__ == '__main__':

    # df = pd.read_csv('tsc_hybrid_0_2026-04-17_12-08-03.csv', index_col=False)
    # df = prepare_df(df, average_folds=False, change_modelname=False)
    # for (ds, fold), ds_data in df.groupby(['dataset', 'fold']):
    #     mod_str = ' - '.join([f'{mod}: {mod_data.shape[0]}' for mod, mod_data in ds_data.groupby('model')])
    #     print(f'{ds:<30} f{fold} - {ds_data.shape[0]} results across models {mod_str}')

    # df = pd.read_csv('tsc_hybrid_-1_2026-03-31_15-31-24.csv', index_col=False)
    # df = prepare_df(df)
    # df['gt_train_energy_total'] /= 3.6e6
    # df['ene_diff'] = (df['gt_train_energy_total'] - df['train_energy_total']) / df['gt_train_energy_total'] * 100
    # cols = ['gt_train_energy_total', 'ene_diff']
    # labels = {col: label for col, label in zip(cols, ['Ground-Truth Training Energy Draw [kWh]', 'CodeCarbon Estimation Error [%]'])}
    # df['modtype'] = df['model'].map(lambda e: e[3:] if e.startswith('P') else e)
    # ccol = 'modtype' # 'dataset'
    # px.scatter(df, x=cols[0], y=cols[1], color=ccol, labels=labels, log_x=True).show()

    os.chdir('results')

    # new = prepare_df(pd.read_csv('tsc_hybrid_0_2026-03-20_14-26-42.csv', index_col=False), average_folds=False)
    # old = prepare_df(pd.read_csv('dev_runs/ts_archive_experiment_-1_2025-12-22_09-58-36.csv', index_col=False), average_folds=False)

    # for (ds, fold), new_res in new[new['model'] == 'Hydra'].groupby(['dataset', 'fold']):
    #     old_res = old[(old['dataset'] == ds) & (old['fold'] == fold) & (old['model'] == 'Hydra')]
    #     acc_new = new_res['accuracy'].values[0] if new_res.shape[0] > 0 else np.nan
    #     acc_old = old_res['accuracy'].values[0] if old_res.shape[0] > 0 else np.nan
    #     append = '(multiple found)' if new_res.shape[0] > 1 or old_res.shape[0] > 1 else ''
    #     print(f'{ds:<30} fold {fold}: ACC DIFF {acc_new-acc_old:6.2f}% - {acc_new:5.2f}% vs. old {acc_old:5.2f}% {append}')

    LOGS = {
        # hydra quant hydrant
        'tsc_hybrid_0_2026-04-17_12-08-03.csv': 'Training',
        'tsc_hybrid_-1_2026-04-28_13-56-26.csv': 'Inference', # i9
        'tsc_hybrid_-1_2026-04-27_23-02-23.csv': 'Inference', # i7
        'tsc_hybrid_0_2026-04-27_20-39-01.csv': 'Inference', # 4090
        # deep learning
        'tsc_deep_0_2026-03-29_14-17-17.csv': 'Training',
        'tsc_deep_-1_2026-04-11_12-54-24.csv': 'Inference', # i9
        'tsc_deep_-1_2026-04-10_20-13-02.csv': 'Inference', # i7
        'tsc_deep_0_2026-04-13_11-44-56.csv': 'Inference', # 4090
    }

    MOD_ORDER = {'P80Quant': 'Quant',
                 'P80Hydra': 'Hydra', 
                 'P80Hydrant': 'Hydrant',
                 'Quant': 'Quant',
                 'Hydra': 'Hydra',
                 'Hydrant': 'Hydrant',
                 'MCDCNN': 'Standard DL',
                 'MLP': 'Standard DL',
                 'ResNet': 'Standard DL',
                 'FCN': 'Standard DL',
                 'LSTMFCN': 'Special DL',
                 'InceptionTime': 'Special DL',
                 'ConvTran': 'Special DL'}
    TYPE_COL = {type: COL_SEL[i] for i, type in enumerate(['Quant', 'Hydra', 'Hydrant', 'Standard DL', 'Special DL'])}
    MOD_COL = {mod: TYPE_COL[type] for mod, type in MOD_ORDER.items()}

    # check total amount of energy consumption
    logs = {'dev': [], 'fin': []}
    for fn in tqdm([os.path.join(root, name) for root, _, files in os.walk(".") for name in files], desc='Loading all logs'):
        which = 'dev' if 'dev_runs' in fn else 'fin'
        try:
            logs[which].append(pd.read_csv(fn, index_col=False))
        except:
            pass
    ene = {key: pd.concat(dfs)[['metrics.infer_energy_total', 'metrics.train_energy_total']].sum().sum() / 3.6e6 for key, dfs in logs.items()}
    print('\n\nFollowing the calls for transparent and sustainable reporting, we estimate the total ' \
          + f'amount of energy consumed by our evaluations to {ene["dev"]:3.0f}+{ene["fin"]:3.0f}={ene["dev"]+ene["fin"]:3.0f} kWh ' \
          + f'(representing the development and testing efforts as well as final experiment runs).\n\n')
    
    # load meta information and evaluation tables
    meta, df = load_meta(), []
    unit_fmt = CustomUnitReformater()
    FMT = lambda col: f"{lookup_meta(meta, col, subdict='properties')} {unit_fmt.reformat_value(1, lookup_meta(meta, col, key='unit', subdict='properties'))[1]}"
    all_logs = [os.walk]
    for logfile, task in LOGS.items():
        df.append(pd.read_csv(logfile, index_col=False))
        df[-1]['params.task'] = task
    df = prepare_df(pd.concat(df))

    # split into train and infer, identify ideal batch sizes during inference, combine with training resource consumption
    hyprid_models = [mod for mod in pd.unique(df['model']) if mod.endswith('Hydra') or mod.endswith('Quant') or mod.endswith('Hydrant')]
    df_infer = df[df['task'] == 'Inference']
    df_abl = df[(df['task'] == 'Training') & (df['model'].isin(hyprid_models))]
    df_train = df[(df['task'] == 'Training') & (df['model'].isin(MOD_ORDER))]
    df = pd.concat([data.sort_values('energy_per_sample').iloc[0] for _, data in df_infer.groupby(['dataset', 'environment', 'model'])], axis=1).transpose().reset_index(drop=True)
    for idx, r in df.iterrows():
        res = df_train[(df_train['model'] == r['model']) & (df_train['dataset'] == r['dataset'])]
        for col in ['train_time_total', 'train_energy_total']:
            if res.shape[0] > 0:
                assert np.isnan(res[col].std())
                df.loc[idx,col] = res[col].mean()
            else:
                df.loc[idx,col] = np.nan
    # limit to datasets with all models evaluated
    # max_res_per_ds = max([data.shape[0] for _, data in df.groupby('dataset')])
    # dim_fields = ['n_samples', 'n_channels', 'length', 'n_labels']
    # ds_dim = {(np.prod(data.iloc[0][dim_fields[:-1]]), ds) for ds, data in df.groupby('dataset') if max_res_per_ds==data.shape[0]}
    # ds_sel = [ds for _, ds in sorted(ds_dim)]
    # df = df[df['dataset'].isin(ds_sel)]
    # df_infer = df_infer[df_infer['dataset'].isin(ds_sel)]q
    print(f'Complete results for N={pd.unique(df["dataset"]).shape[0]} datasets, with {df.shape[0]} evaluations')
    print(" ".join([f"{field} range: {int(df[field].min())}--{int(df[field].max())}" for field in ['n_samples', 'n_channels', 'length', 'n_labels']]))
    # index-scale the results (both for all models and only for the hybrid variants)
    scaled_results = scale_and_rate(df, meta)
    scaled_df, _, _, _, _, _ = scaled_results
    hybrid_df = df[df['model'].isin(['Hydra', 'Quant', 'Hydrant', 'P80Hydrant', 'P80Hydra', 'P80Quant'])]
    hybrid_df, _, _, _, _, _ = scale_and_rate(hybrid_df, meta)
    meta['properties']['compound'] = {'name': 'Compound Score', 'unit': 'compound'}
    meta['properties']['compound_index'] = {'name': 'Compound Score', 'unit': 'compound'}
    meta['properties']['train_energy_total'] = {'name': 'Training Energy Draw', 'unit': 'kilowatthours'}

    fname = print_init('colormap') ###############################################################################
    fig = go.Figure()
    for idx, color in enumerate(COL_SEL):
        fig.add_trace(go.Bar( y=[idx], x=[1], orientation='h', marker=dict(color=color), name=color, showlegend=False))
    fig.update_layout(yaxis_title='Color Key')
    finalize(fig, fname)

    # fname = print_init('config_impact') ###############################################################################
    # fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.015)
    # x_labels = {'Older CPU Impact on Energy [%]': [65, 165], 'GPU Impact on Energy [%]': [-5, 405], 'Batch Sizing Impact on Energy [%]': [95, 205]}
    # legend = set()
    # for m_idx, (mod, m_type) in enumerate(MOD_ORDER.items()):
    #     c = TYPE_COL[m_type]
    #     # col 1 & 2 - hardware impact (optimal batch size)
    #     ene_i9 = df[(df['model'] == mod) & (df['environment'] == 'Intel i9-13900')].sort_values('dataset')['energy_per_sample']
    #     ene_i7 = df[(df['model'] == mod) & (df['environment'] == 'Intel i7-6700')].sort_values('dataset')['energy_per_sample']
    #     ene_gpu = df[(df['model'] == mod) & (df['environment'] == 'NVIDIA RTX 4090')].sort_values('dataset')['energy_per_sample']
    #     x1 = ene_i7.values / ene_i9.values * 100
    #     x2 = ene_gpu.values / ene_i9.values * 100
    #     # col 3 - batch sizing impact across all envs
    #     x3 = []
    #     for _, ds_env_mod_data in df_infer[df_infer['model'] == mod].sort_values('energy_per_sample').groupby(['dataset', 'environment']):
    #         x3.append(ds_env_mod_data['energy_per_sample'].iloc[1:].values / ds_env_mod_data['energy_per_sample'].iloc[0] * 100)
    #     x3 = np.concat(x3)
    #     for c_idx, x in enumerate([x1, x2, x3]):
    #         fig.add_trace(go.Box(x=x, y=[mod]*x.shape[0], marker_color=c, orientation="h", name=m_type, showlegend=m_type not in legend), row=1, col=1+c_idx)
    #         legend.add(m_type)
    # for c_idx, (x_label, minmax) in enumerate(x_labels.items()):
    #     fig.add_vline(x=100, row=1, col=1+c_idx)
    #     fig.update_xaxes(title=x_label, range=minmax, row=1, col=1+c_idx)
    # fig.update_layout(legend=dict(yanchor="bottom", y=1, xanchor="center", x=0.5, orientation='h'))
    # finalize(fig, fname)

    fname = print_init('pruning_ablation') ###############################################################################
    df_abl['model'] = df_abl['model'].map(lambda e: e[3:] if 'P' in e else e)
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.02)
    cols = {'bal_acc': [100.5, 87], 'energy_per_sample': [102, 15]}
    for mod in ['Quant', 'Hydra', 'Hydrant']:
        mod_data = df_abl[df_abl['model'] == mod]
        res = {col: {} for col in cols}
        mc = MOD_COL[mod]
        omc = hex_to_alpha(mc, 0.15)
        for i, (ds, ds_data) in enumerate(mod_data.groupby('dataset')):
            ds_data = ds_data.sort_values('prune_rate')
            y = np.round(100 * ds_data['prune_rate'].values)
            for c_idx, col in enumerate(cols.keys()):
                x = ds_data[col].values
                x = x / np.max(x) * 100 if c_idx == 0 else x / np.max(x) * 100 # relative values based on max acc or min energy
                for r, yv in zip(y, x):
                    if r not in res[col]:
                        res[col][r] = []
                    res[col][r].append(yv)
                fig.add_trace(go.Scatter(x=x, y=y, name=mod, legendgroup=mod, mode='lines', line={'color': omc}, showlegend=False), row=1, col=1+c_idx)
        for c_idx, (col, crange) in enumerate(cols.items()):
            y = list(res[col].keys())
            x = [np.mean(vals) for vals in res[col].values()]
            fig.add_trace(go.Scatter(x=x, y=y, name=mod, legendgroup=mod, mode='lines', line={'color': mc, 'width': 2}, showlegend=c_idx==0), row=1, col=1+c_idx)
            fig.update_xaxes(title='Relative ' + FMT(col).split(' [')[0] + ' [%]', range=crange, row=1, col=1+c_idx)
    fig.update_yaxes(title='Prune Rate [%]', autorange='reversed', row=1, col=1)
    fig.update_layout(legend=dict(yanchor="top", y=1, xanchor="right", x=1))
    finalize(fig, fname)

    fname = print_init('pareto_performance') ###############################################################################
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.02)
    displ, ycol, xcol = set(), 'bal_acc', 'energy_per_sample'
    for mod, mtype in MOD_ORDER.items():
        col = TYPE_COL[mtype]
        for c_idx, env in enumerate(['Intel i9-13900', 'NVIDIA RTX 4090']):
            mod_data = scaled_df[(scaled_df['model'] == mod) & (scaled_df['environment'] == env)]
            x, y = mod_data[f'{xcol}_index'].values, mod_data[f'{ycol}_index'].values
            fig.add_trace(go.Scatter(x=[np.mean(x)], y=[np.mean(y)], text=[mod], name=mtype, legendgroup=mtype, mode='markers+text', textposition="bottom center", textfont={'color': col}, marker={'color': col}, showlegend=mtype not in displ), row=1, col=c_idx+1)
            ex, ey = get_cov_ellipse(x, y, n_std=1.177) # 50% coverage
            fig.add_trace(go.Scatter(x=ex, y=ey, mode='lines', line=dict(color=col), legendgroup=mtype, showlegend=False, opacity=0.5), row=1, col=c_idx+1)
            displ.add(mtype)
            fig.update_xaxes(title=f"Relative {FMT(xcol).split(' [')[0]} on {env}", range=[0, 1.02], row=1, col=c_idx+1)
    for col in [1, 2]:
        fig.add_layout_image(source=GRAD, xref="x domain", yref="y domain", x=1, y=1, xanchor="right", yanchor="top", sizex=1.0, sizey=1.0, sizing="stretch", opacity=0.3, layer="below", row=1, col=col)
    fig.update_yaxes(title=f"Relative {FMT(ycol).split(' [')[0]}", range=[0.55, 1.02], row=1, col=1)
    fig.update_layout(legend=dict(yanchor="bottom", y=0, xanchor="center", x=0.5, orientation='h'))
    finalize(fig, fname)

    fname = print_init('model_stats') ###############################################################################
    fig = make_subplots(rows=1, cols=4, shared_yaxes=True, horizontal_spacing=0.01)
    cols = ['bal_acc', 'energy_per_sample', 'train_energy_total', 'compound_index']
    col_min_max = {col: [np.inf, 0] for col in cols}
    legend = set()
    for m_idx, (mod, m_type) in enumerate(MOD_ORDER.items()):
        data = scaled_df[(scaled_df['model'] == mod) & (scaled_df['environment'] == 'Intel i9-13900')]
        c, y = TYPE_COL[m_type], [mod]*data.shape[0]
        for c_idx, col in enumerate(cols):
            fig.add_trace(go.Box(x=data[col], y=y, marker_color=c, orientation="h", name=m_type, showlegend=m_type not in legend), row=1, col=1+c_idx)
            legend.add(m_type)
            q1, q3 = np.quantile(data[col].values, [0.1, 0.9])
            col_min_max[col][0] = min(col_min_max[col][0], q1)
            col_min_max[col][1] = max(col_min_max[col][1], q3)
    for c_idx, col in enumerate(cols):
        type, minmax = 'linear', col_min_max[col]
        if col in ['energy_per_sample', 'train_energy_total']:
            type = 'log'
            minmax = [np.log10(minmax[0]), np.log10(minmax[1])]
        fig.update_xaxes(title=FMT(col), range=minmax, type=type, row=1, col=1+c_idx)
    fig.update_layout(legend=dict(yanchor="bottom", y=1, xanchor="center", x=0.5, orientation='h'))
    finalize(fig, fname)

    # statistical significance checks
    for column in ['bal_acc', 'energy_per_sample', 'compound']:
        fname = print_init(f'cd_{column}') ###############################################################################
        rel_columns = [hybrid_df['model'], hybrid_df['dataset'] + hybrid_df['environment'], hybrid_df[f'{column}_index']]
        stat_df = pd.concat(rel_columns, axis=1).rename({0: 'configuration'}, axis=1)
        p_values, average_ranks, _ = wilcoxon_holm(stat_df)
        fig = graph_ranks_plotly(average_ranks.values, average_ranks.keys(), p_values, title=f'{FMT(column).split(" [")[0]} Rank', reverse=True)
        finalize(fig, fname, ws=.33, hs=.5)
        # comparison with CD diagram implementation by hfawaz - download and rename main.py from https://github.com/hfawaz/cd-diagram
        # from cd_hfawaz import draw_cd_diagram
        # stats = stat_df.rename({'model': 'classifier_name', f'{column}_index': 'accuracy', 'configuration': 'dataset_name'})
        # draw_cd_diagram(stats, axis=1), title=f'{FMT(column).split(" [")[0]} Rank')

    # start the interactive exploration tool
    app = Visualization(scaled_results)
    app.run()
