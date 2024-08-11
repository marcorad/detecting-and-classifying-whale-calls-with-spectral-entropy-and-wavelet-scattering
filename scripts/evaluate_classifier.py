
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
from annotations import get_annotations
    


def iteration(n_calls, thresh, ws_a: Scattering1D, ws_d: Scattering1D):
    ant_ws = loader.bm_ant_ws_loader()    
    d_ws = loader.bm_d_ws_loader()
    
    def iteration_cls(params: Parameters, l_ws: loader.Loader, ws: Scattering1D):
        
        # prepare the data
        fp_train = []
        X = []
        X_train = []
        y_train = []
        all_detections = []
        all_files = []
        
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
            Text = 0  
            
            # get the random annotated calls from this file
            if len(annotations) >= n_calls:
                ann_train = [annotations[i] for i in np.random.choice(np.arange(len(annotations)), replace=False, size=n_calls)]
                times = [(a['t_start'], a['t_end']) for a in ann_train]
                x, _ = compute_features(times, ws, params, l_ws.get_current_fname())    
                X_train.append(x)
                y_train.extend([1]*n_calls)  
            
            # run the detector to find calls
            proposed_ws = det.proposed_detector(0, K_ws-1, Mf, Mt, Mn, Mh, thresh, t_dim=1, f_dim=0, kappa=params.kappa)
            
            T = proposed_ws.apply(tf_ws)
            detections, det_idx = det.get_detections(T, 0.5, Tmin, Tmax, Text, fs_tf)
            
            all_detections.extend(detections)
            all_files.extend([l_ws.get_current_fname()] * len(detections))
            
            Nt, Ndet, Nann = prec_reca_counts(detections, annotations)
            NT += Nt
            NDET += Ndet
            NANN += Nann
            
            # gather the features of each detection
            x, y = compute_features(detections, ws, params, l_ws.get_current_fname()) # returns m x n feature matrix and the labels
            if not x is None: 
                X.append(x)
                # now get an equal number of counter examples
                fp_idx = (np.array(y) == 0).nonzero()[0]
                if len(fp_idx) >= n_calls:
                    fp_idx = np.random.choice(fp_idx, size=n_calls, replace=False).tolist()
                    X_train.append(x[:, fp_idx])
                    y_train.extend([0]*n_calls)
              

        X = np.concatenate(X, axis=1)
        X_train = np.concatenate(X_train, axis=1)
        # train a classifier to accept or reject a detection based on the training data gathered from each file
        clf = LDA(solver='eigen', shrinkage='auto', priors=[0.5, 0.5])
        clf.fit(X_train.T, y_train)
        y_pred = clf.predict(X.T)
        
        # get the files for each detection
        clf_idx = y_pred.nonzero()[0].tolist()
        clf_detections = [all_detections[i] for i in clf_idx]
        clf_files = [all_files[i] for i in clf_idx]
        clf_output = {}
        for fname, d in zip(clf_files, clf_detections):
            if fname not in clf_output:
                clf_output[fname] = []
            clf_output[fname].append(d)
        # now evaluate the clf detections per file        
        Nt_clf, Ndet_clf = 0, 0
        for fname, dets in clf_output.items():
            Nt, Ndet, _ = prec_reca_counts(dets, get_annotations(fname, 0, 10e6, params.cls))
            Nt_clf += Nt
            Ndet_clf += Ndet
            
                
        prec_orig, reca_orig = calc_prec_rec(NT, NDET, NANN)
        prec_clf, reca_clf = calc_prec_rec(Nt_clf, Ndet_clf, NANN)
        fpph_orig = calc_fp_ph(NT, NDET, NANN)
        fpph_clf = calc_fp_ph(Nt_clf, Ndet_clf, NANN)
        
        N_noise = (NDET - NT)
    
        return {
                'prec_orig': prec_orig,
                'reca_orig': reca_orig,
                'prec_clf': prec_clf,
                'reca_clf': reca_clf,
                'thresh': thresh,
                'n_calls': n_calls,
                'fpph_orig': fpph_orig,
                'fpph_clf': fpph_clf,
                'clf_tp_acc': reca_clf/reca_orig, #probability of accepting TP
                'pct_rejected': (NDET - Ndet_clf)/NDET, #rejected samples
                }
           
            
    bma = iteration_cls(BM_ANT_PARAMETERS, ant_ws, ws_a) 
    bmd = iteration_cls(BM_D_PARAMETERS, d_ws, ws_d)
    return bma, bmd
                

if __name__ == "__main__":
    
    n_calls = 3 
    N_trials = 10
    exp_spacing = lambda low, high: np.exp(np.linspace(np.log(low), np.log(high), 10)).tolist()
    thresh = exp_spacing(0.25, 0.01)
    # thresh = [0.08]
    # print(thresh)     
    
    cfg.cuda()
    p = BM_ANT_PARAMETERS
    ws_a = Scattering1D(p.N_clf_win, p.d_clf, [p.Q1_clf, p.Q2_clf], startfreq=p.f0 / p.fs_audio)
    p = BM_D_PARAMETERS
    ws_d = Scattering1D(p.N_clf_win, p.d_clf, [p.Q1_clf, p.Q2_clf], startfreq=p.f0 / p.fs_audio)
    
    results_a = []
    results_d = []
    for t in thresh:
        print(t)
        for _ in tqdm(range(N_trials)):
            a, d = iteration(n_calls, t, ws_a, ws_d)
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
    
    
        