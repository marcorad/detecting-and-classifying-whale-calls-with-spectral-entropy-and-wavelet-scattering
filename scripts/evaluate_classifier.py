
import torch
import numpy as np
import load_path
import detect.detectors as det
import tf_loader as loader
from typing import Tuple, Dict
from detection_processing import s1_to_tf, compute_features, prec_reca_counts, calc_prec_rec, calc_fp_ph
from parameters import Parameters, BM_ANT_PARAMETERS, BM_D_PARAMETERS
from pprint import pprint
import json
from tqdm import tqdm
from typing import List
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from annotations import get_random_annotation
from scattering.scattering import Scattering1D
from scipy.stats.distributions import chi2
from scattering.config import cfg


def classify_trial(X: np.ndarray, y, n_calls, cls, ws: Scattering1D, params: Parameters, fname):
    example_call_times = []
    example_call_files = []
    for _ in range(n_calls):
        t1, t2, example_call_file = get_random_annotation(cls)
        example_call_times.append((t1, t2))
        example_call_files.append(example_call_file)
    
    X_train, _ = compute_features(example_call_times, ws, params, example_call_files)
    
    gamma = params.gamma_clf
    rho = params.rho_clf
    
    mu = np.mean(X_train, axis=1, keepdims=True)
    Sigma_hat = np.cov(X_train, bias=False)
    Sigma_reg = (1 - gamma) * Sigma_hat + gamma * np.diag(np.diag(Sigma_hat))
    X_mu = X - mu
    
    n = X_mu.shape[1]
    m = X_mu.shape[0]
    
    D_m = np.zeros(shape=(n,))
    Sigma_reg_inv = np.linalg.inv(Sigma_reg)
    for n in range(n):
        xn = X_mu[:, [n]]
        D_m[n] = (xn.T @ Sigma_reg_inv @ xn).item() # Mahalanobis distance
    
    D_m_crit = chi2.ppf(1 - rho, df=m) # critical distance value, df = number of features

    
    y_pred = (D_m < D_m_crit).astype(np.uint)
    
    tp = np.sum((y_pred == 1) & (y == 1))
    tn = np.sum((y_pred == 0) & (y == 0))
    fp = np.sum((y_pred == 1) & (y == 0))
    fn = np.sum((y_pred == 0) & (y == 1))
    
    return tp, tn, fp, fn
    


def iteration(n_calls, thresh, ws_a: Scattering1D, ws_d: Scattering1D, N_trails = 50):
    ant_ws = loader.bm_ant_ws_loader()    
    d_ws = loader.bm_d_ws_loader()
    
    def iteration_cls(params: Parameters, l_ws: loader.Loader, ws: Scattering1D):
        
        # prepare the data
        X = []
        Y = []
        
        NT, NDET, NANN = 0, 0, 0
        
        for (S1, annotations) in l_ws:
            tf_ws = s1_to_tf(S1) 
            fs_tf = params.fs_tf_det
            
            # SE parameters
            Mh = params.Mh       
            K_ws = tf_ws.shape[0]   
            
            # adaptive whitening params
            Mf = params.Mf  
            Mn = params.Mn
            Mt = params.Mt
            
            # Detection parameters
            Tmin = params.Tmin
            Tmax = params.Tmax
            Text = 0 # we do not want to extend boundaries, since we have clf windows            
            
            # run the detector to find calls
            proposed_ws = det.proposed_detector(0, K_ws-1, Mf, Mt, Mn, Mh, thresh, t_dim=1, f_dim=0, kappa=params.kappa)
            
            T = proposed_ws.apply(tf_ws)
            detections, det_idx = det.get_detections(T, 0.5, Tmin, Tmax, Text, fs_tf)
            
            Nt, Ndet, Nann = prec_reca_counts(detections, annotations)
            NT += Nt
            NDET += Ndet
            NANN += Nann
            
            # gather the features of each detection
            x, y = compute_features(detections, ws, params, l_ws.get_current_fname()) # returns m x n feature matrix and the labels
            if not x is None: 
                X.append(x)
                Y.extend(y) # y will remain a list, not a tensor
        
        if len(X) == 0: return {
                'prec_orig': 0,
                'reca_orig': 0,
                'prec_cls': 0,
                'reca_cls': 0,
                'thresh': thresh,
                'n_calls': n_calls
                }
        X = np.concatenate(X, axis=1)
        Y = np.array(Y)
        
        # print(params.cls)
        # now, perform classification in a number of randomised trails
        results = []
        for i in tqdm(range(N_trails)):
            r = classify_trial(X, Y, n_calls, l_ws.cls, ws, params, l_ws.get_current_fname())
            results.append(r)
            
        
        mu = np.median(np.array(results), axis=0).tolist() #[tp tn fp fn]
        sigma = np.array(results).std(axis=0).tolist()
        Nt_clf = mu[0] #true positives
        Ndet_clf = mu[0] + mu[2] #true positives + false positives, since we no discard the output where the classifier rejected the detections
        Nann_clf = NANN # we tested on all calls
        
        prec_orig, reca_orig = calc_prec_rec(NT, NDET, NANN)
        prec_clf, reca_clf = calc_prec_rec(Nt_clf, Ndet_clf, Nann_clf)
        fpph_orig = calc_fp_ph(NT, NDET, NANN)
        fpph_clf = calc_fp_ph(Nt_clf, Ndet_clf, Nann_clf, n_hours=19)
        
        return {
                'prec_orig': prec_orig,
                'reca_orig': reca_orig,
                'prec_clf': prec_clf,
                'reca_clf': reca_clf,
                'thresh': thresh,
                'n_calls': n_calls,
                'fpph_orig': fpph_orig,
                'fpph_clf': fpph_clf,
                }
           
            
    bma = iteration_cls(BM_ANT_PARAMETERS, ant_ws, ws_a) 
    bmd = iteration_cls(BM_D_PARAMETERS, d_ws, ws_d)
    return bma, bmd
                

if __name__ == "__main__":
    
    n_calls = [20]  
    N = 50
    exp_spacing = lambda low, high: np.exp(np.linspace(np.log(low), np.log(high), N)).tolist()
    thresh = exp_spacing(0.3, 0.005)
    # thresh = [0.05]
    # print(thresh)     
    
    cfg.cuda()
    p = BM_ANT_PARAMETERS
    ws_a = Scattering1D(p.N_clf_win, p.d_clf, [p.Q1_clf, p.Q2_clf], startfreq=p.f0 / p.fs_audio)
    p = BM_D_PARAMETERS
    ws_d = Scattering1D(p.N_clf_win, p.d_clf, [p.Q1_clf, p.Q2_clf], startfreq=p.f0 / p.fs_audio)
    
    results_a = []
    results_d = []
    for t in tqdm(thresh):
        for n in n_calls:
            # print(n)
            a, d = iteration(n, t, ws_a, ws_d)
            results_a.append(a)
            results_d.append(d)
        
    with open('results/bm_a_clf_results.json', 'w') as file:
        json.dump(results_a, file, indent=4)
    with open('results/bm_d_clf_results.json', 'w') as file:
        json.dump(results_d, file, indent=4)
        
    print('BM-ANT')
    pprint(results_a)
    print('BM-D')
    pprint(results_d)
    
    
        