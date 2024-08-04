
import torch
import numpy as np
import load_path
import detect.detectors as det
import tf_loader as loader
from typing import Tuple, Dict
from detection_processing import prec_reca_counts, calc_prec_rec, calc_fp_ph, s1_to_tf
from parameters import Parameters, BM_ANT_PARAMETERS, BM_D_PARAMETERS
from pprint import pprint
import json
from tqdm import tqdm


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
            fs_tf = params.fs_tf_det
            
            # adaptive whitening params
            Mf = params.Mf  
            Mn = params.Mn
            Mt = params.Mt
            
            # TF decompositions (magnitude), the detectors will square if needed
            tf_ws = s1_to_tf(X_ws)
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
                if not key in T_stats: T_stats[key] = [10e20, 0, 0] #min, max, mean                
                
                
                T_stats[key][0] = min(T_stats[key][0], T.min().item())
                T_stats[key][1] = max(T_stats[key][1], T.max().item())
                T_stats[key][2] = T_stats[key][2]/2 + T.type(torch.float32).mean().item()/2
                
                t = thresh[key]
                if 'proposed' in key: t = 0.5
                detections, _ = det.get_detections(T, t, Tmin, Tmax, Text, fs_tf)
                if not key in results: results[key] = [0, 0, 0]
                Nt, Ndet, Nann = prec_reca_counts(detections, annotations)
                results[key][0] += Nt
                results[key][1] += Ndet
                results[key][2] += Nann
                
        for k, counts in results.items():
            prec, reca = calc_prec_rec(*counts)
            fpph = calc_fp_ph(*counts) #false positives per hour
            results[k] = {
                'prec': prec,
                'reca': reca,
                'fpph': fpph,
                'thresh': thresh[k]
            }
            
        if iter_num == None:
            print(params.cls)
            pprint(T_stats)  
                
        return results if iter_num != None else T_stats
                
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
    
    bm_a_stats, bm_d_stats = iteration(t) # will return the min, max, mean stats
    
    N = 50
    even_spacing = lambda low, high: np.linspace(low, high, N).tolist()
    exp_spacing = lambda low, high: np.exp(np.linspace(np.log(low), np.log(high), N)).tolist()
    log_spacing = lambda low, high: np.log(np.linspace(np.exp(low), np.exp(high), N)).tolist()
    
    # automatically choose
    thresholds = {
        'A': {
                'bled'          :    exp_spacing (bm_a_stats['bled'          ][0], bm_a_stats['bled'          ][1]/2),
                'bled_aw'       :    exp_spacing (bm_a_stats['bled_aw'       ][0], bm_a_stats['bled_aw'       ][1]/2),
                'gpl_stft'      :    exp_spacing (bm_a_stats['gpl_stft'      ][0], bm_a_stats['gpl_stft'      ][1]/2),
                'gpl_ws'        :    exp_spacing (bm_a_stats['gpl_ws'        ][0], bm_a_stats['gpl_ws'        ][1]/2),
                'nuttall'       :    exp_spacing (bm_a_stats['nuttall'       ][0], bm_a_stats['nuttall'       ][1]/2),
                'nuttall_aw'    :    exp_spacing (bm_a_stats['nuttall_aw'    ][0], bm_a_stats['nuttall_aw'    ][1]/2),
                'proposed_stft' :    exp_spacing (0.3                            , 0.005                            ),
                'proposed_ws'   :    exp_spacing (0.3                            , 0.005                            ),
                'se_stft'       :    even_spacing(bm_a_stats['se_stft'       ][0], bm_a_stats['se_stft'       ][1]/2),
            },
        'D' : {
                'bled'          :    exp_spacing (bm_d_stats['bled'          ][0], bm_d_stats['bled'          ][1]/2),
                'bled_aw'       :    exp_spacing (bm_d_stats['bled_aw'       ][0], bm_d_stats['bled_aw'       ][1]/2),
                'gpl_stft'      :    exp_spacing (bm_d_stats['gpl_stft'      ][0], bm_d_stats['gpl_stft'      ][1]/2),
                'gpl_ws'        :    exp_spacing (bm_d_stats['gpl_ws'        ][0], bm_d_stats['gpl_ws'        ][1]/2),
                'nuttall'       :    exp_spacing (bm_d_stats['nuttall'       ][0], bm_d_stats['nuttall'       ][1]/2),
                'nuttall_aw'    :    exp_spacing (bm_d_stats['nuttall_aw'    ][0]*2, bm_d_stats['nuttall_aw'    ][1]/4),
                'proposed_stft' :    exp_spacing (0.3                            , 0.005                            ),
                'proposed_ws'   :    exp_spacing (0.3                            , 0.005                            ),
                'se_stft'       :    even_spacing(bm_d_stats['se_stft'       ][0], bm_d_stats['se_stft'       ][1]/2),
            }        
    }
        
    
    results_a = []
    results_d = []
    for i in tqdm(range(N)):
        a, d = iteration(thresholds, iter_num=i)
        results_a.append(a)
        results_d.append(d)
        
    with open('results/bm_a_detector_results.json', 'w') as file:
        json.dump(results_a, file, indent=4)
    with open('results/bm_d_detector_results.json', 'w') as file:
        json.dump(results_d, file, indent=4)
        
    # print('BM-ANT')
    # pprint(results_a)
    # print('BM-D')
    # pprint(results_d)
    
    
        