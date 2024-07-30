
import torch
import numpy as np
import load_path
import detect.detectors as det
import tf_loader as loader
from typing import Tuple, Dict
from detection_processing import prec_reca_counts, calc_prec_rec, calc_fp_ph
from parameters import Parameters, BM_ANT_PARAMETERS, BM_D_PARAMETERS
from pprint import pprint


def s1_to_tf(S1):
    X = []
    f = []
    for _lambda in sorted(S1.keys()):
        X.append(S1[_lambda][:, None])
        f.append(_lambda / np.pi / 2 * 250)
    X = torch.concat(X, dim=1).T
    return X


def iteration(thresholds: Dict, iter_num = None):
    ant_ws = loader.bm_ant_ws_loader()
    ant_stft = loader.bm_ant_stft_loader()  
    
    d_ws = loader.bm_d_ws_loader()
    d_stft = loader.bm_d_stft_loader()
    
    def iteration_cls(params: Parameters, l_ws: loader.Loader, l_stft: loader.Loader): 
        thresh = thresholds
        if iter_num != None:
            thresh = { k: t[iter_num] for k, t in thresh[params.cls].items()} 
        
        results = {}  
        # min and max statistics for the entire dataset
        T_stats = {}
        for (X_ws, annotations), (X_stft, _) in zip(l_ws, l_stft):
            # audio and TF sample frequency
            fs_tf = params.fs_tf
            
            # adaptive whitening params
            Mf = params.Mf  
            Mn = params.Mn
            Mt = params.Mt
            
            # TF decompositions (magnitude), the detectors will square if needed
            tf_ws = s1_to_tf(X_ws[0])
            tf_stft = torch.from_numpy(X_stft[2].astype(np.float32))  
            
            # STFT parameters - signal band limits
            k0_stft = params.k0_stft
            k1_stft = params.k1_stft
            
            # SE parameters
            Mh = params.Mh
            K_ws = tf_ws.shape[0]
            
            # Detection parameters
            Tmin = params.Tmin
            Tmax = params.Tmax
            Text = 0 # we do not want to extend boundaries in the evaluation phase
            
            # DETECTORS
            # Proposed: SE with adaptive whitening
            proposed_stft = det.proposed_detector(k0_stft, k1_stft, Mf, Mt, Mn, Mh, thresh['proposed_stft'], t_dim=1, f_dim=0, kappa=params.kappa)
            proposed_ws = det.proposed_detector(0, K_ws-1, Mf, Mt, Mn, Mh, thresh['proposed_ws'], t_dim=1, f_dim=0, kappa=params.kappa)
            
            thresh['proposed_stft'] = 0.5
            thresh['proposed_ws'] = 0.5
            
            # Baseline SE, with no modifications
            se_stft = det.se(k0_stft, k1_stft, t_dim=1, f_dim=0)
            
            # GPL
            gpl_stft = det.helble_gpl(k0 = k0_stft, k1=k1_stft, t_dim=1, f_dim=0)
            gpl_ws = det.helble_gpl(k0 = 0, k1=K_ws-1, t_dim=1, f_dim=0)
            
            # Nuttall
            nuttall = det.nuttall(k0_stft, k1_stft, t_dim=1, f_dim=0)
            nuttall_aw = det.aw_nuttall(k0_stft, k1_stft, Mf, Mt, Mn, t_dim=1, f_dim=0)   
            
            # BLED
            bled = det.bled(k0_stft, k1_stft, t_dim=1, f_dim=0)
            bled_aw = det.aw_bled(k0_stft, k1_stft, Mf, Mt, Mn, t_dim=1, f_dim=0)
            
            detectors: Dict[str, det.DetectorChain] = {
                'proposed_stft': proposed_stft,
                'proposed_ws': proposed_ws,
                'se_stft': se_stft,
                'gpl_stft': gpl_stft,
                'gpl_ws': gpl_ws,
                'nuttall': nuttall,
                'nuttall_aw': nuttall_aw,
                'bled': bled,
                'bled_aw': bled_aw,
            }         
            
            # counts all the detections across all files
            for key, detector in detectors.items():
                tf = tf_ws if key.endswith('_ws') else tf_stft
                T = detector.apply(tf)
                
                # set the min and max statistics
                if not key in T_stats: T_stats[key] = [10e12, 0, 0] #min, max, mean                
                
                
                T_stats[key][0] = min(T_stats[key][0], T.min().item())
                T_stats[key][1] = max(T_stats[key][1], T.max().item())
                T_stats[key][2] = T_stats[key][2]/2 + T.type(torch.float32).mean().item()/2
                
                t = thresh[key]
                detections, _ = det.get_detections(T, t, Tmin, Tmax, Text, fs_tf)
                if not key in results: results[key] = [0, 0, 0]
                Nt, Ndet, Nann = prec_reca_counts(detections, annotations)
                results[key][0] += Nt
                results[key][1] += Ndet
                results[key][2] += Nann
                
        for k, counts in results.items():
            results[k] = (calc_prec_rec(*counts), calc_fp_ph(*counts) )
            
        if iter_num == None:
            print(params.cls)
            pprint(T_stats)  
                
        return results
                
    return iteration_cls(BM_ANT_PARAMETERS, ant_ws, ant_stft), iteration_cls(BM_D_PARAMETERS, d_ws, d_stft)

if __name__ == "__main__":
    t = {
                'proposed_stft': 0.05,
                'proposed_ws': 0.05,
                'se_stft': 0,
                'gpl_stft': 0,
                'gpl_ws': 0,
                'nuttall': 0,
                'nuttall_aw': 0,
                'bled': 0,
                'bled_aw': 0,
            } 
    
    N = 2
    even_spacing = lambda low, high: np.linspace(low, high, N).tolist()
    exp_spacing = lambda low, high: np.exp(np.linspace(np.log(low), np.log(high), N)).tolist()
    log_spacing = lambda low, high: np.log(np.linspace(np.exp(low), np.exp(high), N)).tolist()
    
    thresholds = {
        'A': {
                'bled':             exp_spacing(6e-5, 0.00003),
                'bled_aw':          exp_spacing(10, 60),
                'gpl_stft':         exp_spacing(0.0002, 0.004),
                'gpl_ws':           exp_spacing(0.0002, 0.002),
                'nuttall':          exp_spacing(10e-12, 100e-12),
                'nuttall_aw':       exp_spacing(20000, 1e5),
                'proposed_stft':    exp_spacing(0.3, 0.001),
                'proposed_ws':      exp_spacing(0.3, 0.001),
                'se_stft':          even_spacing(0.3, 0.5),
            },
        'D' : {
                'bled':             exp_spacing(0.1e-5, 6e-5),
                'bled_aw':          exp_spacing(200, 300),
                'gpl_stft':         exp_spacing(4e-5, 0.002),
                'gpl_ws':           exp_spacing(6e-5, 0.002),
                'nuttall':          exp_spacing(10e-12, 2e-10),
                'nuttall_aw':       exp_spacing(20000, 1e5),
                'proposed_stft':    log_spacing(0.3, 0.001),
                'proposed_ws':      log_spacing(0.3, 0.001),
                'se_stft':          even_spacing(0.35, 0.8),
            }        
    }
        
    
    results_a = []
    results_d = []
    for i in range(N):
        print(i)
        a, d = iteration(thresholds, iter_num=i)
        results_a.append(a)
        results_d.append(d)
        
    print('BM-ANT')
    pprint(results_a)
    print('BM-D')
    pprint(results_d)
        