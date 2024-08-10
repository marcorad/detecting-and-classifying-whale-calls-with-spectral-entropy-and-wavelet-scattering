import matplotlib.pyplot as plt
import pickle as pkl
from torch import Tensor
import torch
import numpy as np
import load_path
from annotations import get_annotations
import os
import math
from matplotlib.ticker import NullFormatter
from matplotlib.patches import Rectangle
import detect.detectors as det

plt.style.use('fast')
plt.rcParams["font.family"] = "Noto Serif"
plt.rcParams["figure.figsize"] = (4.5, 3.5)
plt.rcParams["mathtext.fontset"] = 'stix'
plt.rcParams["image.cmap"] = 'magma'
# plt.rcParams["axes.grid"] = 'True'

markers = ['o', '^', 's', 'x', 'd', '*', '+']
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

BM_D_PATH = 'tmp/bm-d/ws'
BM_D_PKL_FILES = os.listdir(BM_D_PATH) 


BM_ANT_PATH = 'tmp/bm-ant/ws'
BM_ANT_PKL_FILES = os.listdir(BM_ANT_PATH) 


def load(cls, i):
    files = BM_D_PKL_FILES if cls == 'D' else BM_ANT_PKL_FILES
    path = BM_D_PATH if cls == 'D' else BM_ANT_PATH
    f_plot = files[i]
    with open(f'{path}/{f_plot}', 'rb') as file:
        S1 = pkl.load(file)
    anns = get_annotations(f_plot[:-4], 0, 1e6, cls)
    return S1, anns

def s1_to_img(S1, d, fs=250):
    X = []
    f = []
    for _lambda in sorted(S1.keys()):        
        X.append(S1[_lambda][:, None])
        f.append(_lambda / np.pi / 2 * fs)
    X = torch.concat(X, dim=1).numpy().T
    return np.arange(X.shape[1]) / fs * d, f, X

def plot_s(t, f, x, t1, dur, ax, fs=250, file = None):   
    n1 = math.floor(fs * t1)
    n2 = n1 + math.ceil(fs * dur)
    T, F = np.meshgrid(t[n1:n2], f)   
    
    
    ax.pcolor(T, F, (x[:, n1:n2]), edgecolor='face')
    ax.set_yscale('log')
    ax.get_yaxis().set_major_formatter(NullFormatter())
    ax.get_yaxis().set_minor_formatter(NullFormatter())
    ax.set_yticks([], minor = True)
    ax.set_yticks(f[::4])
    ax.set_yticklabels([f'{fi:.1f}' for fi in f[::4]])
    
            
    return n1, n2

def plot_sample(cls, i, j):
    S1, anns = load(cls, i)
    a = anns[j]
    dur = a['duration']
    p = 0.2
    t1 = a['t_start'] - p*dur
    dur *= (1 + p*2)
    fs = 250 / (1 if cls == 'D' else 3)
    d = 32 if cls == 'D' else 64
    t, f, x = s1_to_img(S1, d, fs)
    fig, ax = plt.subplots() 
    fig.set_size_inches(4, 3)    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    fig.subplots_adjust(left=0.2, bottom = 0.15)
    plot_s(t - t1, f, x, t1, dur, ax, fs/d)
    fig.savefig(f'fig/{cls}_example.pdf')
            
    
 
plot_sample('A', 10, 25)
plot_sample('D', 2, 13)
  


# plt.show()
