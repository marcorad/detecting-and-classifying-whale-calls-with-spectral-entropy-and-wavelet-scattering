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
from scattering.filterbank import _filterbank_1d
from scattering.config import cfg

cfg.set_alpha(12, 2.5)

plt.style.use('fast')
plt.rcParams["font.family"] = "Noto Serif"
plt.rcParams["figure.figsize"] = (4.5, 3.5)
plt.rcParams["mathtext.fontset"] = 'stix'
plt.rcParams["image.cmap"] = 'hot'
# plt.rcParams["axes.grid"] = 'True'

markers = ['o', '^', 's', 'x', 'd', '*', '+']
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

N = 4096*8
d = 32
Q = 4
fb, _ = _filterbank_1d(N, d, Q, input_ds_factors=[1], include_negative_lambdas=False)
psi = fb[1]['psi']
phi = fb[1]['phi']

n = np.arange(N)

fig, ax = plt.subplots()
for lambda_ in psi.keys():    
    ax.plot(n, psi[lambda_], 'k', linewidth=1.0)
    
ax.plot(n, phi, 'k--', linewidth=1.0)
ax.set_xlim([0, n[-1]//(2)+200])
ax.set_ylim([0, 1.05])
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_yticks([])
ax.set_yticklabels([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_linewidth(1.0)
fig.set_size_inches(5, 1)
fig.subplots_adjust(left=0.001, bottom = 0.05, right=0.99)
    
fig.savefig('fig/fb_example.pdf')

