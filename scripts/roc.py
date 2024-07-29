import load_path
import detect as det
from annotations import get_annotations
import numpy as np

def prec_reca(detector: det.DetectorChain, thresh, Tmin, Text, fs, fname, cls):
    T = detector.get_statistic()
    detections = det.get_detections(T, thresh, Tmin, Text, fs)

    anns = get_annotations(fname, 0, 10e6, cls)
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
        
    
    prec = Nt / Ndet
    reca = Nt / Nann

    return prec, reca