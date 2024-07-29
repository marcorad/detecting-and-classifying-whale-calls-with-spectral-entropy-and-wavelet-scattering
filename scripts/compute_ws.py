from config import PROCESSED_DATASET_PATH

import load_path

from sepws.scattering.separable_scattering import SeparableScattering
from sepws.scattering.config import cfg

from scipy.signal import decimate
from scipy.io.wavfile import read, write
import librosa

import os
import numpy as np
from torch import Tensor
import torch
import math
import pickle as pkl


def compute_scattering(ds, block_size, Q, d, f_start, subdir, allow_ds=True):
    print(subdir)
    cfg.cuda()
    cfg.set_precision('single')
    
    fs = 250 / ds
    f_start = f_start / fs # works with normalised freq
    ws = SeparableScattering([block_size], [d], [[q] for q in Q], [f_start], allow_ds=allow_ds) #compatibility with sep-ws
    wav_path = PROCESSED_DATASET_PATH + "/raven/all"
    files = os.listdir(wav_path)
    wav_files = []
    for f in files: 
        if f.endswith('.wav'): 
            wav_files.append(f)
    os.makedirs(f'tmp/{subdir}', exist_ok=True)
    fs_S = 250/ds/d # sample frequency of scattering coeffs
    for fname in wav_files:
        _, x = read(wav_path + '/' + fname)        
        x  = x.astype(np.float32) / (2**15 - 1)
        if ds > 1: x = decimate(x, ds)
        
        # now pad so that we can split it into blocks for processing
        L = x.shape[0]
        n_blocks = math.ceil(L/block_size)
        pad = n_blocks * block_size - L
        x = np.concatenate((x, np.zeros((pad,))), axis=0)[:, None]
        x = np.reshape(x, (n_blocks, block_size)) #we need a tensor of shape (n_blocks, block_size)   
        # print(x.shape)     
        x = torch.from_numpy(x.astype(np.float32)).cuda()
        _, Sp, _, _ = ws._scattering(x, returnSpath=True) #Sp is a dict such that Sp[p], where p is the lambda path (expressed as tuples) stores the scattering coefficients, with sample frequency 250/ds/d
        padding = 0 if allow_ds else ws.pad[0] # does not unpad if there is no ds
        S1 = {}
        S2 = {}
        for p in Sp.keys():
            s: Tensor = Sp[p]
            N = L//d if allow_ds else L
            s = s[:, padding:(padding+ N)]
            s = s.reshape((s.shape[0] * s.shape[1],)).cpu()
            if len(p) == 1:
                if(p[0] != 0): # not interested in S0
                    S1[p[0]] = s
            elif len(p) == 2:
                # additional tuple depth from separable scattering implementation
                p1 = p[0][0]
                p2 = p[1][0]
                if p1 not in S2.keys():
                    S2[p1] = {}
                S2[p1][p2] = s # more convenient to process than the current format
        with open(f'tmp/{subdir}/{fname}.pkl', 'wb') as pickle_file:
            pkl.dump((S1, S2), pickle_file)
        

if __name__ == '__main__':
        
    # Bm-Ant calls
    # 83 1/3 Hz sampling rate
    # invariance downsampling factor of 64 ~ 760 ms
    # Q of 16 followed by 8
    compute_scattering(3, 2**15, [8, 4], 64, 15, 'bm-ant')
    compute_scattering(3, 2**15, [8, 4], 64, 15, 'bm-ant-no-ds', False) #for plots

    # Bm-D calls
    # 250 Hz sampling rate
    # invariance downsampling factor of 32 ~ 128 ms
    # Q of 16 followed by 8
    compute_scattering(1, 2**15, [8, 4], 32, 20, 'bm-d')
    compute_scattering(1, 2**15, [8, 4], 32, 20, 'bm-d-no-ds', False) #for plots
            
        
        

