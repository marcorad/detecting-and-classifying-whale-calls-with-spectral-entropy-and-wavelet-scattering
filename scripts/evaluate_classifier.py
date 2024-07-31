
import torch
import numpy as np
import load_path
import detect.detectors as det
import tf_loader as loader
from typing import Tuple, Dict
from detection_processing import s1_to_tf, compute_features
from parameters import Parameters, BM_ANT_PARAMETERS, BM_D_PARAMETERS
from pprint import pprint
import json
from tqdm import tqdm
from typing import List


def classify_trial(X: np.ndarray, y: List[int], n_calls, prop_noise_to_calls):
    pass


def iteration(n_calls, N_trails = 20, prop_noise_to_calls = 1.0):
    ant_ws = loader.bm_ant_ws_loader()
    
    d_ws = loader.bm_d_ws_loader()
    
    def iteration_cls(params: Parameters, l_ws: loader.Loader, thresh):
        
        # prepare the data
        X = []
        Y = []
        
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
            _, det_idx = det.get_detections(T, 0.5, Tmin, Tmax, Text, fs_tf)
            
            # gather the features of each detection
            x, y = compute_features(S1, S2, det_idx, params, l_ws.get_current_fname())
            X.append(X)
            Y.extend(y) # y will remain a list, not a tensor
        
        X = torch.concat(X, dim=0).numpy()
        
        # now, perform classification in a number of randomised trails
        results = []
        for i in range(N_trails):
            r = classify_trial(X, y, n_calls, prop_noise_to_calls)
            results.append(r)
        
        #average the results
        
        return 0
           
                
    return iteration_cls(BM_ANT_PARAMETERS, ant_ws), iteration_cls(BM_D_PARAMETERS, d_ws)

if __name__ == "__main__":
    
    n_calls = [10, 25, 50, 100]
        
    
    results_a = []
    results_d = []
    for n in tqdm(n_calls):
        a, d = iteration(n)
        results_a.append(a)
        results_d.append(d)
        
    with open('results/bm_a_clf_results.json', 'w') as file:
        json.dump(results_a, file, indent=4)
    with open('results/bm_d_clf_results.json', 'w') as file:
        json.dump(results_d, file, indent=4)
        
    # print('BM-ANT')
    # pprint(results_a)
    # print('BM-D')
    # pprint(results_d)
    
    
        