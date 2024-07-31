import load_path
import numpy as np
from parameters import Parameters
from typing import Dict, List, Any, Tuple
from torch import Tensor
import torch
from annotations import has_class

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
    return detph * (1 - Nt / Nann)

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
    return idx_start, idx_end

def pool_detection(s: Tensor, idx: Tuple[int, int], params: Parameters):
    s = s.squeeze()
    L = s.shape[0]
    pad_left = -idx[0] if idx[0] < 0 else 0
    pad_right = idx[0] - L if idx[0] > L else 0
    L_seg = params.N_clf_seg_length
    sfeats = s[max(idx[0],0):min(idx[1], L)]
    # add mirror padding, if required
    if pad_left > 0: sfeats = torch.concat([sfeats[1:pad_left+1].flip(), sfeats])
    if pad_right > 0: sfeats = torch.concat([sfeats, sfeats[L-pad_right-1:L].flip()])
    
    # now pool the segments
    sfeats = sfeats.unfold(dimension=0, size=L_seg, step=L_seg)
    pt = params.pooling_type
    if pt == 'max':
        sfeats, _ = torch.max(sfeats, dim=1, keepdim=False)
    elif pt == 'mean':
        sfeats = torch.mean(sfeats, dim=1, keepdim=False)
    elif pt == 'median':
        sfeats, _ = torch.median(sfeats, dim=1, keepdim=False)
    return sfeats

def compute_features(S1: Dict[float, Tensor], S2: Dict[float, Dict[float, Tensor]], 
                     det_idx: List[Tuple[int, int]], params: Parameters, fname):
    
    # compute the data and class tensors for a specific file, given a list of detections and the clf parameters
    
    X = []
    y = []
    
    for idx in det_idx:
        feats = []        
        # assign the class before window extension - 
        # we don't want a neighbouring call to be considered if it is not centered on the detection window
        t_det = (idx[0]/params.fs_tf, idx[1]/params.fs_tf)
        y.append(1 if has_class(fname, t_det[0], t_det[1], params.cls) else 0)
        
        # compute the window (indices may be out of bounds, handled by the pool_detection function)
        idx = compute_clf_window_idx(idx, params)
        
        for _lambda1, s_lambda1 in S1.items():
            # add S1 coeffs
            s1_feats = pool_detection(s_lambda1, idx, params)
            if params.log_scattering_coeffs: s1_feats = torch.log(s1_feats + params.log_scattering_eps)
            feats.append(s1_feats)
            
            # add the S2 coeffs of the corresponding S1 (there may not be S2 coeffs, depending on the BW of the S1 wavelet)
            if _lambda1 in S2:
                S2_lambda2 = S2[_lambda1]
                for lambda2, s_lambda2 in S2_lambda2.items():
                    if params.normalise_s2: s_lambda2 = s_lambda2 / s_lambda1
                    s2_feats = pool_detection(s_lambda2, idx, params)
                    if params.log_scattering_coeffs: s2_feats = torch.log(s2_feats + params.log_scattering_eps)
                    feats.append(s2_feats)
        x = torch.concat(feats, dim=0)
        X.append(x[None, :])
        
    X = torch.concat(X, dim=0) # N x d feature matrix, as is standard in many python ML libs
    return X, y