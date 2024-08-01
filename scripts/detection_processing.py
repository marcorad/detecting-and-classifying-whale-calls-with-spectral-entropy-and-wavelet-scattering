import load_path
import numpy as np
from parameters import Parameters
from typing import Dict, List, Any, Tuple
from torch import Tensor
import torch
from annotations import get_annotations
from scipy.fftpack import dct
from torch.nn.functional import pad


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

def compute_features(S1: Dict[float, Tensor], S2: Dict[float, Dict[float, Tensor]], 
                     det_idx: List[Tuple[int, int]], params: Parameters, fname):
    
    # compute the data and class tensors for a specific file, given a list of detections and the clf parameters  
    
    # transform s1 coeffs
    s1_tf = []        
    for lambda1 in sorted(S1.keys()):
        s1_tf.append(S1[lambda1][None, :])
    s1_tf = torch.concat(s1_tf, dim=0)

    if params.log_scattering_coeffs: s1_tf = torch.log(s1_tf + params.eps)  
    if params.scattering_dct: s1_tf = Tensor(dct(s1_tf.numpy(), axis=0))

    
    # transform s2 coeffs
    s2_tfs = []
    for lambda1 in sorted(S1.keys()):
        if lambda1 in S2:
            s2_tf = []
            for lambda2 in sorted(S2[lambda1].keys()):
                s_lambda2 = S2[lambda1][lambda2]
                s_lambda1 = S1[lambda1]
                if params.normalise_s2: s_lambda2 = s_lambda2 / (s_lambda1 + params.eps)
                s2_tf.append(s_lambda2[None, :])
            s2_tf = torch.concat(s2_tf, dim=0)
                
            
            if params.log_scattering_coeffs: s2_tf = torch.log(s2_tf + params.eps)   
            if params.scattering_dct: s2_tf = Tensor(dct(s2_tf.numpy(), axis=0))
            s2_tfs.append(s2_tf)   

         
    # create feature matrix
    if len(s2_tfs) > 0: 
        s2_tfs = torch.concat(s2_tfs, dim=0)
        S_feats = torch.concat([s1_tf, s2_tfs], dim=0) # [N_paths, L]  
    else: S_feats = s1_tf # [N_paths, L]  
    
    # pad the matrix with reflection in the time dimension to deal with detections on the boundaries
    S_feats = pad(S_feats, (params.N_clf_win, params.N_clf_win), mode='reflect')
    
    anns = get_annotations(fname, 0, 10e6, params.cls)
    
    X = []
    y = []
    for idx in det_idx:
        feats = []        
        # assign the class before window extension - 
        # we don't want a neighbouring call to be considered if it is not centered on the detection window
        t_det = (idx[0]/params.fs_tf, idx[1]/params.fs_tf)
        y.append(1 if interval_overlaps_annotation(*t_det, anns) else 0)
        
        # compute the window (indices may be out of bounds, handled by the pool_detection function)
        w_idx = compute_clf_window_idx(idx, params)      
        x: Tensor = pool_detection(S_feats, w_idx, params).flatten()
        if params.normalise_feature_vector: x = x / (x*x).sum()
        x = torch.concat([x, Tensor([idx[1] - idx[0]])]) # add the detection length as a feature
        X.append(x[None, :])
      
    if len(X) > 0:  
        X = torch.concat(X, dim=0) # N x d feature matrix, as is standard in many python ML libs 
    else: X = None   
    return X, y