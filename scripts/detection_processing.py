import load_path
import numpy as np

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

def calc_fp_ph(Nt, Ndet, Nann, n_hours = 19):
    return (Ndet - Nt) / n_hours

def prec_reca(detections, anns):
    Nt, Ndet, Nann = calc_prec_rec(detections, anns)     
    return calc_prec_rec(Nt, Ndet, Nann)

def compute_features(S1, S2, dets):
    pass