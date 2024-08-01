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

BM_D_PATH = 'tmp/bm-d/ws'
BM_D_PKL_FILES = os.listdir(BM_D_PATH) 


BM_ANT_PATH = 'tmp/bm-ant/ws'
BM_ANT_PKL_FILES = os.listdir(BM_ANT_PATH) 

f_plot = BM_D_PKL_FILES[7]
with open(f'{BM_D_PATH}/{f_plot}', 'rb') as file:
    S1, S2 = pkl.load(file)
anns = get_annotations(f_plot[:-4], 0, 1e6, 'D')
print(anns)

def s1_to_img(S1, fs=250):
    X = []
    f = []
    for _lambda in sorted(S1.keys()):        
        X.append(S1[_lambda][:, None])
        f.append(_lambda / np.pi / 2 * 250)
    X = torch.concat(X, dim=1).numpy().T
    return np.arange(X.shape[1]) / fs, f, X

def plot_s(t, f, x, t1, dur, ax, fs=250, file = None):   
    n1 = math.floor(fs * t1)
    n2 = n1 + math.ceil(fs * dur)
    T, F = np.meshgrid(t[n1:n2], f)   
    
    
    ax.pcolor(T, F, (x[:, n1:n2]))
    ax.set_yscale('log')
    ax.get_yaxis().set_major_formatter(NullFormatter())
    ax.get_yaxis().set_minor_formatter(NullFormatter())
    ax.set_yticks(f[::2])
    ax.set_yticklabels([f'{fi:.1f}' for fi in f[::2]])
    
    if file != None:
        anns = get_annotations(file, t1 - 50, t1+dur+50, 'D')
        for ann in anns:
            tst, te, fst, fe = ann['t_start'], ann['t_end'], ann['f_start'], ann['f_end']
            r = Rectangle((tst, fst), ann['duration'], ann['freq_range'], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(r)
            
    return n1, n2
            
    
fig, ax = plt.subplots(1, 1)        
    
# t1 = 310
t1 = 1810
dur = 40
fname = f_plot[:-4] 
fs = 250/32

t, f, x = s1_to_img(S1, fs)
plot_s(t, f, x, t1, dur, ax, file=fname, fs=fs)

print(x.shape)
det1 = det.proposed_detector(0, x.shape[0]-1, 1, 100, int(fs*60*5), 2, 0.05, f_dim=0, t_dim=1, kappa=0.9)
det2 = det.helble_gpl(f_dim=0, t_dim=1)
det3 = det.aw_gpl(2, 12, 150, f_dim=0, t_dim=1)
det1.apply(torch.from_numpy(x))
det2.apply(torch.from_numpy(x))
det3.apply(torch.from_numpy(x))

fig, ax = plt.subplots(5, 1, sharex='col')  
n1, n2 = plot_s(t, f, det1.results[0], t1, dur, ax[0], file=fname, fs=fs)
ax[1].plot(t[n1:n2], det1.results[-2][n1:n2])
ax[2].plot(t[n1:n2], det1.results[-1][n1:n2])
ax[3].plot(t[n1:n2], det2.results[-1][n1:n2])
ax[4].plot(t[n1:n2], det3.results[-1][n1:n2])

ax[0].set_xlim([t1, t1+dur])

def print_prec_rec(detector: det.DetectorChain, thresh, Tmin, Tmax, Text, fs):
    T = detector.get_statistic()
    detections, _ = det.get_detections(T, thresh, Tmin, Tmax, Text, fs)
    print(T.min(), T.max())

    anns = get_annotations(fname, 0, 10e6, 'D')
    for a in anns: a['detected'] = False
    Nann = len(anns)
    Ndet = len(detections)
    Nt = 0

    for d in detections:
        for a in anns:            
            if not a['detected']:
                if np.maximum(d[0], a['t_start']) <= np.minimum(d[1], a['t_end']):
                    Nt += 1
                    a['detected'] = True
        
    print(Nann, Ndet, Nt)
    prec = Nt / Ndet
    rec = Nt / Nann

    print(prec, rec)
        


Tmin = 0.75
Tmax = 5
Text = 0
print('GPL')
print_prec_rec(det2, 1000e-7, Tmin, Tmax, Text,  fs)
print('Proposed')
print_prec_rec(det1, 0.5, Tmin, Tmax, Text, fs)

plt.show()
