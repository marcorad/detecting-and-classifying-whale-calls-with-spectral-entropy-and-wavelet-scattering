
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


def classify_trial(X: np.ndarray, y: List[int], n_calls, prop_noise_to_calls):
    y = np.array(y)
    n_noise = int(n_calls * prop_noise_to_calls)
    y_cls_idx = np.nonzero(y == 1)[0]
    np.random.shuffle(y_cls_idx)
    y_noise_idx = np.nonzero(y == 0)[0]
    np.random.shuffle(y_noise_idx)
    train_idx = np.concatenate([y_cls_idx[0:n_calls], y_noise_idx[0:n_noise]], axis=0)
    test_idx = np.concatenate([y_cls_idx[n_calls:], y_noise_idx[n_noise:]], axis=0)
    X_train, y_train = X[train_idx, :], y[train_idx]
    X_test, y_test = X[test_idx, :], y[test_idx]
    
    mu = np.median(X_train, axis=0)
    sigma = np.median(np.abs(X_train - mu), axis=0)
    sigma[sigma == 0] = 1.0
    
    X_train = (X_train - mu)/sigma
    X_test = (X_test - mu)/sigma
    
    clf = LDA(solver='eigen', shrinkage='auto', priors=[0.5, 0.5])
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    tp = np.sum((y_pred == 1) & (y_test == 1))
    tn = np.sum((y_pred == 0) & (y_test == 0))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    fn = np.sum((y_pred == 0) & (y_test == 1))
    
    return tp, tn, fp, fn
    


def iteration(n_calls, thresh, N_trails = 100):
    ant_ws = loader.bm_ant_ws_loader()
    
    d_ws = loader.bm_d_ws_loader()
    
    def iteration_cls(params: Parameters, l_ws: loader.Loader, prop_noise_to_calls = 2.0):
        
        # prepare the data
        X = []
        Y = []
        
        NT, NDET, NANN = 0, 0, 0
        
        for ((S1, S2), annotations) in l_ws:
            tf_ws = s1_to_tf(S1) 
            fs_tf = params.fs_tf
            
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
            x, y = compute_features(S1, S2, det_idx, params, l_ws.get_current_fname())
            if x != None: 
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
        X = torch.concat(X, dim=0).numpy()
        
        # print(params.cls)
        # now, perform classification in a number of randomised trails
        results = []
        for i in range(N_trails):
            r = classify_trial(X, Y, n_calls, prop_noise_to_calls)
            results.append(r)
            
        
        mu = np.array(results).mean(axis=0).tolist() #[tp tn fp fn]
        sigma = np.array(results).std(axis=0).tolist()
        Nt_clf = mu[0] #true positives
        Ndet_clf = mu[0] + mu[2] #true positives + false positives, since we no discard the output where the classifier rejected the detections
        Nann_clf = NANN - n_calls # we used some calls for the training, so take them away!
        
        prec_orig, reca_orig = calc_prec_rec(NT, NDET, NANN)
        prec_clf, reca_clf = calc_prec_rec(Nt_clf, Ndet_clf, Nann_clf)
        fpph_orig = calc_fp_ph(NT, NDET, NANN)
        fpph_clf = calc_fp_ph(Nt_clf, Ndet_clf, Nann_clf, n_hours=19*(Nann_clf/NANN))
        
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
           
            
    bma = iteration_cls(BM_ANT_PARAMETERS, ant_ws) 
    bmd = iteration_cls(BM_D_PARAMETERS, d_ws)
    return bma, bmd
                

if __name__ == "__main__":
    
    n_calls = [10, 25, 50]  
    N = 50
    exp_spacing = lambda low, high: np.exp(np.linspace(np.log(low), np.log(high), N)).tolist()
    thresh = exp_spacing(0.3, 0.005)
    # thresh = [0.05]
    # print(thresh)     
    
    results_a = []
    results_d = []
    for t in tqdm(thresh):
        for n in n_calls:
            # print(n)
            a, d = iteration(n, t)
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
    
    
        