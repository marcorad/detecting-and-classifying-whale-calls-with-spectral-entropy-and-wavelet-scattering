import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import json

plt.style.use('fast')
plt.rcParams["font.family"] = "Noto Serif"
plt.rcParams["figure.figsize"] = (4.5, 3.5)
plt.rcParams["mathtext.fontset"] = 'stix'
# plt.rcParams["axes.grid"] = 'True'

markers = ['o', '^', 's', 'x', 'd', '*', '+']

# plt.plot([1, 2, 3, 4], [4,5, 6, 2])
# plt.title('Hello')
# plt.xlabel('$x$')
# plt.ylabel('$y = x^2 + 1$')

def plot_prec_reca(ax: Axes, prec, reca, index, dashed = False):
    m = ('--' if dashed else '-') + markers[index]
    ax.plot(reca, prec, m)
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')

# ax: Axes
# fig, ax = plt.subplots()
# plot_prec_reca(ax, np.linspace(0, 1, 10), np.linspace(0.5, 1, 10), 0)
# plot_prec_reca(ax, np.linspace(0, 1, 10), np.linspace(0, 0.5, 10), 1)
# plot_prec_reca(ax, np.linspace(0, 1, 10), np.linspace(0.25, 0.5, 10), 2, True)
# ax.legend(['A', 'B', 'C'])

def get_detector_df(name):
    fname = f'results/bm_{name}_detector_results.json'
    records = []
    with open(fname, 'r') as file:
        s = json.load(file)
    for r in s:
        for k, res in r.items():
            res['detector'] = k
            records.append(res)
    return pd.DataFrame(records)

def get_clf_df(name):
    records = []
    df = get_detector_df(name)
    fname = f'results/bm_{name}_clf_results.json'
    with open(fname, 'r') as file:
        s = json.load(file)
    for r in s: 
        res = {
            'prec': r['prec_clf'],
            'reca': r['reca_clf'],
            'thresh': r['thresh'],
            'fpph': r['fpph_clf'],
        }
        res['detector'] = f'proposed_clf_{r["n_calls"]}'
        records.append(res)
    return pd.concat([df, pd.DataFrame(records)]).reset_index(drop=True)
    

def plot_prec_reca_multiple(lims, df: pd.DataFrame, config: dict):
    dets = df['detector'].unique().tolist()
    ax: Axes
    fig: Figure
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 4.5)
    
    leg = []
    for det in dets:
        if det not in config: continue
        leg.append(config[det]['name'])
        df_det: pd.DataFrame = df[df['detector'] == det]
        df_det = df_det[df_det['prec'] > lims[0]]
        df_det = df_det[df_det['reca'] > lims[1]]
        df_det.sort_values('reca', inplace=True)
        df_det.reset_index(inplace=True, drop=True)
  
        if len(df_det) > 15:
            df_det = df_det.iloc[::3]
        plot_prec_reca(ax, df_det['prec'], df_det['reca'], index=config[det]['index'], dashed=config[det]['dashed'])
    ax.legend(leg, loc='upper center',  bbox_to_anchor=(0.5, 1.25),
          ncol=len(leg)//2, fancybox=True, shadow=True)
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 - box.height*0.0, box.width, box.height*0.9])

def plot_detector_results(name, lims):
    df = get_detector_df(name)
    config = {
            "proposed_stft": {
                'name': 'Proposed (STFT)',
                'dashed': True,
                'index': 0
            },
            "proposed_ws": {
                'name': 'Proposed (WS)',
                'dashed': False,
                'index': 0
            },
            "se_stft": {
                'name': 'SE Baseline (STFT)',
                'dashed': True,
                'index': 1
            },
            "gpl_stft": {
                'name': 'GPL (STFT)',
                'dashed': True,
                'index': 2
            },
            "gpl_ws": {
                'name': 'GPL (WS)',
                'dashed': False,
                'index': 2
            },
            # "nuttall": {
            #     'name': 'Nuttall (STFT)',
            #     'dashed': True,
            #     'index': 3
            # },
            "nuttall_aw": {
                'name': 'Nuttall + AW (STFT)',
                'dashed': True,
                'index': 4
            },
            # "bled": {
            #     'name': 'BLED (STFT)',
            #     'dashed': True,
            #     'index': 5
            # },
            # "bled_aw": {
            #     'name': 'BLED + AW (STFT)',
            #     'dashed': True,
            #     'index': 6
            # }
        }
    plot_prec_reca_multiple(lims, df, config)
    
def plot_clf_results(name, lims):
    df = get_clf_df(name)
    config = {
            "proposed_ws": {
                'name': 'Proposed (WS)',
                'dashed': False,
                'index': 0
            },
            "gpl_ws": {
                'name': 'GPL (WS)',
                'dashed': False,
                'index': 2
            },
            "proposed_clf_5": {
                'name': 'Proposed (WS) + LDA (5)',
                'dashed': False,
                'index': 3
            },
            "proposed_clf_10": {
                'name': 'Proposed (WS) + LDA (10)',
                'dashed': False,
                'index': 4
            }   
        }
    plot_prec_reca_multiple(lims, df, config)
        

# bma_res = pd.read_json('results/bm_a_detector_results.json', orient='records')
# print(bma_res.head())


plot_detector_results('a', lims=[0.2, 0.2])
plot_detector_results('d', lims=[0.1, 0.05])

plot_clf_results('a', lims=[0.2, 0.2])
plot_clf_results('d', lims=[0.1, 0.05])

plt.show()

# print(get_clf_df('a'))
