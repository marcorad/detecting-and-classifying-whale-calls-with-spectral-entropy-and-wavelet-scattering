import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
import json

plt.style.use('fast')
plt.rcParams["font.family"] = "Noto Serif"
plt.rcParams["figure.figsize"] = (4.5, 3.5)
plt.rcParams["mathtext.fontset"] = 'stix'
# plt.rcParams["axes.grid"] = 'True'

markers = ['o', '^', 's', 'x', 'd', '*', '+']
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
hatches = ['///', '..']

# plt.plot([1, 2, 3, 4], [4,5, 6, 2])
# plt.title('Hello')
# plt.xlabel('$x$')
# plt.ylabel('$y = x^2 + 1$')

def plot_prec_reca(ax: Axes, prec, reca, index, dashed = False):
    m = ('--' if dashed else '-') + markers[index]
    ax.plot(reca, prec, m, color=colors[index], markersize=5)
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    
def plot_fpph_reca(ax: Axes, fpph, reca, index, dashed = False):
    m = ('--' if dashed else '-') + markers[index]
    ax.plot(fpph, reca, m, color=colors[index], markersize=5)
    ax.set_xlabel('False Positives per Hour')
    ax.set_ylabel('Recall')

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
    fname = f'results/bm_{name}_clf_results.json'
    df = pd.read_json(fname, orient='records')
    return df.groupby('thresh').mean().reset_index(), df.groupby('thresh').std().reset_index()
    

def plot_prec_reca_multiple(lims, df: pd.DataFrame, config: dict, show_leg, name) -> Figure:
    dets = df['detector'].unique().tolist()
    ax: Axes
    fig: Figure
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 4)
    
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
            df_det = df_det.iloc[1::3]
        
        if 'GPL' in config[det]['name']:
            df_det = df_det.iloc[1::2]
            
        plot_prec_reca(ax, df_det['prec'], df_det['reca'], index=config[det]['index'], dashed=config[det]['dashed'])
    if show_leg:
        ax.legend(leg, loc='upper center',  bbox_to_anchor=(0.5, 1.35),
            ncol=max(len(leg)//4, 1), fancybox=True, shadow=True)
        
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 - box.height*0.0, box.width, box.height*0.85])
    return fig
    
def plot_fpph_reca_multiple(lims, df: pd.DataFrame, config: dict, show_leg):
    dets = df['detector'].unique().tolist()
    ax: Axes
    fig: Figure
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 4)
    
    leg = []
    for det in dets:
        if det not in config: continue
        leg.append(config[det]['name'])
        df_det: pd.DataFrame = df[df['detector'] == det]
        df_det = df_det[df_det['fpph'] < lims[0]]
        df_det = df_det[df_det['reca'] > lims[1]]
        df_det.sort_values('fpph', inplace=True)
        df_det.reset_index(inplace=True, drop=True)
  
        if len(df_det) > 15:
            df_det = df_det.iloc[1::3]
        plot_fpph_reca(ax, df_det['fpph'], df_det['reca'], index=config[det]['index'], dashed=config[det]['dashed'])
    if show_leg:
        ax.legend(leg, loc='upper center',  bbox_to_anchor=(0.5, 1.25),
            ncol=max(len(leg)//4, 1), fancybox=True, shadow=True)
        
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
                'name': 'Proposed ($\mathcal{S}_1$)',
                'dashed': False,
                'index': 0
            },
            "se_stft": {
                'name': 'SE Baseline (STFT)',
                'dashed': True,
                'index': 1
            },
            "se_ws": {
                'name': 'SE Baseline ($\mathcal{S}_1$)',
                'dashed': False,
                'index': 1
            },
            "gpl_stft": {
                'name': 'GPL (STFT)',
                'dashed': True,
                'index': 2
            },
            "gpl_ws": {
                'name': 'GPL ($\mathcal{S}_1$)',
                'dashed': False,
                'index': 2
            },
            "nuttall_aw_stft": {
                'name': 'Nuttall + AW (STFT)',
                'dashed': True,
                'index': 3
            },
            "nuttall_aw_ws": {
                'name': 'Nuttall + AW ($\mathcal{S}_1$)',
                'dashed': False,
                'index': 3
            },
        }
    fig = plot_prec_reca_multiple(lims, df, config, name=='a', name)
    fig.savefig(f'fig/prec_reca_{name}.pdf')
    
def plot_clf_results():   
    leg = ["Proposed (Bm-Ant)", "Proposed + LDA (Bm-Ant)", "Proposed (Bm-D)", "Proposed + LDA (Bm-D)"]
    
    # ax: Axes
    # fig: Figure
    # fig, ax = plt.subplots()
    # fig.set_size_inches(4, 3)   
    # plot_prec_reca(ax, df['prec_orig'], df['reca_orig'], 0, False)
    # # plot_prec_reca(ax, df['prec_clf'], df['reca_clf'], 1, False)
    # p1 = [[r,p] for r, p in zip(df['reca_clf'], df['prec_clf'] + df_std['prec_clf'])]
    # p2 = [[r,p] for r, p in zip(df['reca_clf'], df['prec_clf'] - df_std['prec_clf'])]
    # p2.reverse()
    # ax.add_patch(Polygon(p1 + p2, hatch='///', edgecolor='k')) 
    
    # # ax.legend(leg, loc='upper center',  bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True, ncol=2)
        
    # df = df[(df['fpph_orig'] < fpph_lim) & (df['fpph_clf'] < fpph_lim)]
    ax: Axes
    fig: Figure
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 4)    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 - box.height*0.0, box.width, box.height*0.9])
    
    for i, name in enumerate(['a', 'd']):
        df, df_std = get_clf_df(name)
        plot_prec_reca(ax, df['prec_orig'], df['reca_orig'], i, False)
        p1 = [[f,p] for f, p in zip( df['reca_clf'] + df_std['reca_clf'], df['prec_clf'])]
        p2 = [[f,p] for f, p in zip( df['reca_clf'] - df_std['reca_clf'], df['prec_clf'])]
        
        p2.reverse()
        ax.add_patch(Polygon(p1 + p2, hatch=hatches[i], edgecolor=colors[i], facecolor='w', linewidth=1.2)) 
    # plot_fpph_reca(ax, df['fpph_clf'], df['reca_clf'], 1, False)
    
    ax.legend(leg, loc='upper center',  bbox_to_anchor=(0.5, 1.25), fancybox=True, shadow=True, ncol=2)
    # box = ax.get_position()
    # ax.set_xlim([0, fpph_lim])
    # ax.set_ylim(None)
    # ax.set_position([box.x0, box.y0 - box.height*0.0, box.width, box.height*0.9])
    fig.savefig(f'fig/clf_fpph_reca.pdf')
    
def plot_improvement_results():
    from scipy.interpolate import interp1d
    ax: Axes
    fig: Figure
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 4)
    for i, name in enumerate(['a', 'd']):
        df, df_std = get_clf_df(name)
        
        orig_fpph_from_reca = interp1d([0] + df['reca_orig'].tolist(), [0] + df['fpph_orig'].tolist(), kind='quadratic')
        
        fpph_orig = orig_fpph_from_reca(df['reca_clf'])
             
        df['fpph_improvement'] = (fpph_orig - df['fpph_clf']) / fpph_orig * 100
        df_std['fpph_uncertainty'] = df_std['fpph_clf'] / fpph_orig * 100
        
        
        
        p1 = [[f,p] for f, p in zip( df['reca_clf'], df['fpph_improvement'] + df_std['fpph_uncertainty'])]
        p2 = [[f,p] for f, p in zip( df['reca_clf'], df['fpph_improvement'] - df_std['fpph_uncertainty'])]
        p2.reverse()
        ax.add_patch(Polygon(p1 + p2, hatch=hatches[i], edgecolor=colors[i], facecolor='w', linewidth=1.2)) 
        # plot_fpph_reca(ax, df['fpph_clf'], df['reca_clf'], 1, False)
        # box = ax.get_position()
        # ax.set_xlim([0, fpph_lim])
        # ax.set_ylim(None)
        # ax.set_position([box.x0, box.y0 - box.height*0.0, box.width, box.height*0.9])
        
    leg = ["Bm-Ant", "Bm-D"]        
    ax.legend(leg, loc='upper center',  bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True, ncol=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('False Positives per Hour Improvement (%)')
    ax.set_ylim([0, 100])
    fig.savefig(f'fig/clf_fpph_reca_impr.pdf')
        

# bma_res = pd.read_json('results/bm_a_detector_results.json', orient='records')
# print(bma_res.head())


plot_detector_results('a', lims=[0.25, 0.25])
plot_detector_results('d', lims=[0.1, 0.05])

plot_clf_results()
plot_improvement_results()

plt.show()

# print(get_clf_df('a'))
