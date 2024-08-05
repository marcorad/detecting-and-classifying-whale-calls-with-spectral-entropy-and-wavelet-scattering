import load_path
import numpy as np
from parameters import Parameters
from typing import Dict, List, Any, Tuple
from torch import Tensor
import torch
from annotations import get_annotations
from scipy.fftpack import dct
from torch.nn.functional import pad
from scattering.scattering import Scattering1D
from compute_tf import sp_to_s1_s2
import librosa
from config import PROCESSED_DATASET_AUDIO_PATH
import soundfile as sf
from scattering.config import cfg
from scipy.signal import decimate, butter, filtfilt

def prec_reca_counts(detections, anns):
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
    return Nt, Ndet, Nann

def calc_prec_rec(Nt, Ndet, Nann):
    prec = 0 if Ndet == 0 else Nt / Ndet
    reca = Nt / Nann
    return prec, reca    

def calc_fp_ph(Nt, Ndet, Nann, n_hours = 19): # false positives per hour, based on the number of hours of recordings and the number of detections
    if Ndet == 0 or Nann == 0 or Nt == 0: return 0
    detph = Ndet / n_hours
    return detph * (1 - Nt / Ndet)

def prec_reca(detections, anns):
    Nt, Ndet, Nann = calc_prec_rec(detections, anns)     
    return calc_prec_rec(Nt, Ndet, Nann)

def s1_to_tf(S1):
    X = []
    f = []
    for _lambda in sorted(S1.keys()):
        X.append(S1[_lambda][:, None])
        f.append(_lambda / np.pi / 2 * 250)
    X = torch.concat(X, dim=1).T
    return X

def compute_clf_window_idx(idx: Tuple[int, int], params: Parameters):
    mid = (idx[0] + idx[1])//2
    w = params.N_clf_win
    idx_start = mid - w//2
    idx_end = idx_start + w
    return idx_start + w, idx_end + w # offset to correspond to padding

def pool_detection(s: Tensor, idx: Tuple[int, int], params: Parameters) -> Tensor:    
    sfeats = s[:, idx[0]:idx[1]]
    if sfeats.isnan().any():
        pass
    L_seg = params.N_clf_win
    
    # now pool the segments
    sfeats = sfeats.unfold(dimension=-1, size=L_seg, step=L_seg)
    pt = params.pooling_type
    if pt == 'max':
        sfeats, _ = torch.max(sfeats, dim=-1, keepdim=False)
    elif pt == 'mean':
        sfeats = torch.mean(sfeats, dim=-1, keepdim=False)
    elif pt == 'median':
        sfeats, _ = torch.median(sfeats, dim=-1, keepdim=False)
    return sfeats

def interval_overlaps_annotation(t1, t2, anns):
    for a in anns:    
        if np.maximum(t1, a['t_start']) <= np.minimum(t2, a['t_end']):
            return True
    return False

def _compute_ws_features(S1, S2, params: Parameters) -> Tensor:
    # transform s1 coeffs
    s1_tf = []        
    for lambda1 in sorted(S1.keys()): # each entry is of shape (Nbatch, Ntime)
        s1_tf.append(S1[lambda1][:, :, None]) # add a dimension for the frequency vals
    s1_tf = torch.concat(s1_tf, dim=-1)

    if params.log_scattering_coeffs: s1_tf = torch.log(s1_tf + params.eps)  
    if params.scattering_dct: s1_tf = Tensor(dct(s1_tf.cpu().numpy(), axis=-1)) # last dimension is indexed by lambda1
    if params.time_dct: s1_tf = Tensor(dct(s1_tf.cpu().numpy(), axis=1)) # last dimension is indexed by lambda1
    
    # transform s2 coeffs
    s2_tfs = []
    for lambda1 in sorted(S1.keys()):
        if lambda1 in S2:
            s2_tf = []
            for lambda2 in sorted(S2[lambda1].keys()):
                s_lambda2 = S2[lambda1][lambda2]
                s_lambda1 = S1[lambda1]
                if params.normalise_s2: s_lambda2 = s_lambda2 / (s_lambda1 + params.eps)
                s2_tf.append(s_lambda2[:, :, None])
            s2_tf = torch.concat(s2_tf, dim=-1)                
            
            if params.log_scattering_coeffs: s2_tf = torch.log(s2_tf + params.eps)   
            if params.scattering_dct: s2_tf = Tensor(dct(s2_tf.cpu().numpy(), axis=-1)) # last dimension is indexed by lambda2
            if params.time_dct: s2_tf = Tensor(dct(s2_tf.cpu().numpy(), axis=1)) # last dimension
            s2_tfs.append(s2_tf)   
                     
    # create feature matrix    
    return torch.concat([s1_tf.flatten(1), *[s2.flatten(1) for s2 in s2_tfs]], dim=1)

def get_audio_segment(t_mid, fname, params: Parameters):
    path = f'{PROCESSED_DATASET_AUDIO_PATH}/{fname}'
    audio = sf.SoundFile(path)
    
    #shift window into the sample range (i.e., don't pad)
    Lw = params.N_clf_win * params.d_audio
    n_mid = int(t_mid * 250)
    n_start = max(n_mid - Lw//2, 0)
    if n_start + Lw > audio.frames:
        n_start -= (n_start + Lw - audio.frames)
    
    audio.seek(n_start)
    x = audio.read(Lw).astype(np.float32) / (2**15 - 1)  # files are read as int16  
    xd = decimate(x, params.d_audio) 
    if params.normalise_clf_audio: 
        xf = xd
        # normalise by the power in the relevant frequency band
        if params.f0*0.9 > 0:
            b, a = butter(4, params.f0 * 0.9, fs=250/params.d_audio, btype='low')
            xf = filtfilt(b, a, xf)
        if params.f1*1.1 < 125/params.d_audio * 0.9:
            b, a = butter(4, params.f1*1.1, fs=250/params.d_audio, btype='high')
            xf = filtfilt(b, a, xf)
        xd = xd / np.std(xf) 
    return xd

def compute_features(det_times: List[Tuple[float, float]], ws: Scattering1D, params: Parameters, fname) -> np.ndarray:
    
    if isinstance(fname, str):
        fname = [fname] * len(det_times)
    X = []
    y = []
    # compute the data and class tensors for a specific file, given a list of detections and the clf parameters  
    for t_det, file_name in zip(det_times, fname):
        anns = get_annotations(file_name, 0, 10e6, params.cls)
        x = get_audio_segment(t_det[0]/2 + t_det[1]/2, file_name, params)[None, :] # get the segment centered on the detection window
        x = torch.from_numpy(x.copy()).type(cfg.REAL_DTYPE).to(cfg.DEVICE)
        y.append(1 if interval_overlaps_annotation(*t_det, anns) else 0)
        X.append(x)
        
    if len(X) == 0: return None, y
    
    X = torch.concat(X, dim=0)
        
    _, Sp, _, _ = ws._scattering(X, returnSpath=True) # compute its features, returned as a single vector
    S1, S2 = sp_to_s1_s2(Sp, params.N_clf_win, same_signal_batches=False)       
    D = _compute_ws_features(S1, S2, params).cpu().numpy().T # m * n data matrix         
     
    return D, y