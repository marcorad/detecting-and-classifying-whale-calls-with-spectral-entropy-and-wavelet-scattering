import torch
import numpy as np
import load_path
import detect.detectors as det
import tf_loader as loader
from typing import Tuple, Dict
from detection_processing import s1_to_tf
from parameters import Parameters, BM_ANT_PARAMETERS, BM_D_PARAMETERS
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.stats.distributions import beta

fname = 'tmp/bm-d/ws/20161221_160000.wav.pkl'

with open(fname, 'rb') as file:
    S1 = pkl.load(file)
    
X = s1_to_tf(S1)
params = BM_D_PARAMETERS
# adaptive whitening params
Mf = params.Mf  
Mn = params.Mn
Mt = params.Mt

# SE parameters
Mh = 0
K_ws = X.shape[0]
kappa = params.kappa

d: det.DetectorChain = det.proposed_detector(0, K_ws-1, Mf, Mt, Mn, Mh, t_dim=1, f_dim=0, sig=0.05, kappa=kappa, nu=params.nu_proposed_se)
d.apply(X)
H = d.results[-2].numpy() #after median filtering
H_max = np.log(K_ws)

z = 1 - H / H_max
# z = np.sort(z, axis=None)[0:int(kappa * len(H))]
p: det.SignalProbBeta = d.chain[-1]

plt.style.use('fast')
plt.rcParams["font.family"] = "Noto Serif"
plt.rcParams["figure.figsize"] = (4.5, 3.5)
plt.rcParams["mathtext.fontset"] = 'stix'
plt.rcParams["image.cmap"] = 'hot'
plt.rcParams["font.size"] = 12

ax: Axes
fig, ax = plt.subplots()

z_p = np.linspace(0, 1, 1000)
z_fit = beta.pdf(z_p, p.alpha, p.beta)

ax.hist(z, bins=30, density=True, color='w', edgecolor='k')
ax.plot(z_p, z_fit, 'k--')
fig.set_size_inches((4, 3))
ax.set_xlabel('$z$')
ax.set_ylabel('$f_{\ \mathbf{Z}}$')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim(0, 0.5)
fig.subplots_adjust(bottom = 0.2, left=0.15)
fig.savefig('fig/beta_fit_example.pdf')

