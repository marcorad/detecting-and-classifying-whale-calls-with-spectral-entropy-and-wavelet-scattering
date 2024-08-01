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

from scipy.signal import stft

from parameters import BM_ANT_PARAMETERS, BM_D_PARAMETERS


def compute_tf(ds, block_size, Q, d, f_start, subdir, allow_ds=True):
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
    os.makedirs(f'tmp/{subdir}/ws', exist_ok=True)
    os.makedirs(f'tmp/{subdir}/stft', exist_ok=True)
    fs_S = 250/ds/d # sample frequency of scattering coeffs
    for fname in wav_files:
        
        _, x = read(wav_path + '/' + fname)        
        x  = x.astype(np.float32) / (2**15 - 1)
        if ds > 1: x = decimate(x, ds)
        
        # Scattering
        # now pad so that we can split it into blocks for processing
        L = x.shape[0]
        n_blocks = math.ceil(L/block_size)
        pad = n_blocks * block_size - L
        xb = np.concatenate((x, np.zeros((pad,))), axis=0)[:, None]
        xb = np.reshape(xb, (n_blocks, block_size)) #we need a tensor of shape (n_blocks, block_size)   
        # print(x.shape)     
        xb = torch.from_numpy(xb.astype(np.float32)).cuda()
        _, Sp, _, _ = ws._scattering(xb, returnSpath=True) #Sp is a dict such that Sp[p], where p is the lambda path (expressed as tuples) stores the scattering coefficients, with sample frequency 250/ds/d
        padding = 0 if allow_ds else ws.pad[0] # does not unpad if there is no ds
        S1 = {}
        S2 = {}
        for p in Sp.keys():
            print(p)
            s: Tensor = Sp[p]
            N = L//d if allow_ds else L
            s = s[:, padding:(padding+ N)]
            s = s.reshape((s.shape[0] * s.shape[1],)).cpu()
            s[s < 0] = 0 # sometimes, floating point error due to FFT convolution with phi can cause it to be slightly smaller than 0, so clamp it!
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
        with open(f'tmp/{subdir}/ws/{fname}.pkl', 'wb') as pickle_file:
            pkl.dump((S1, S2), pickle_file)
            
        # STFT
        # configured such that the TF decomposition is equivalent to the WS configuration in terms of output frequency with half-overlapping windows
        
        f, t, X_stft = stft(x, window='hann', nperseg=2*d, noverlap=d, boundary='even', padded=True, fs=fs_S)
        X_stft = np.abs(X_stft)
        with open(f'tmp/{subdir}/stft/{fname}.pkl', 'wb') as pickle_file:
            pkl.dump((f, t, X_stft), pickle_file)
        

if __name__ == '__main__':
        
    # Bm-Ant calls
    Q = [BM_ANT_PARAMETERS.Q1, BM_ANT_PARAMETERS.Q2]
    d = BM_ANT_PARAMETERS.d
    d_audio = BM_ANT_PARAMETERS.d_audio
    fmin = BM_ANT_PARAMETERS.f0
    compute_tf(d_audio, 2**15, Q, d, fmin, 'bm-ant')
    # compute_tf(3, 2**15, [8, 4], 64, 15, 'bm-ant-no-ds', False) #for plots

    # Bm-D calls  
    Q = [BM_D_PARAMETERS.Q1, BM_D_PARAMETERS.Q2]
    d = BM_D_PARAMETERS.d
    d_audio = BM_D_PARAMETERS.d_audio
    fmin = BM_D_PARAMETERS.f0
    compute_tf(d_audio, 2**15, Q, d, fmin, 'bm-d')
    # compute_tf(1, 2**15, [8, 4], 32, 20, 'bm-d-no-ds', False) #for plots
            
        
        

