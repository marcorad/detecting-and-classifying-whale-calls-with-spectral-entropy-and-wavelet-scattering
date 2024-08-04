# functions to interact with annotations

import pandas as pd

from config import PROCESSED_DATASET_PATH
import math
import numpy as np
from typing import Tuple

ANNOTATION_FILE_PATH = PROCESSED_DATASET_PATH + '/raven/all'

ANN_DF: pd.DataFrame = pd.read_csv(ANNOTATION_FILE_PATH + '/Bm.selections.txt', sep='\t')
ANN_DF = ANN_DF[ANN_DF['View'] == 'Spectrogram 1']
ANN_DF['Begin Time File (s)'] = ANN_DF['Beg File Samp (samples)'] / 250
ANN_DF['End Time File (s)'] = ANN_DF['End File Samp (samples)'] / 250

ANN_DF.reset_index(inplace=True, drop=True)

def get_annotations(fname, t1, t2, cls=None):
    df: pd.DataFrame = ANN_DF[ANN_DF['Begin File'] == fname]
    # print(df.head())
    overlaps = (df['Begin Time File (s)'] >= t1) & (df['End Time File (s)'] <= t2)
    if cls != None:
        overlaps = overlaps & (df['From'] == cls)
    df = df.loc[overlaps]
    anns = []
    for i, row in df.iterrows():
        tb = row['Begin Time File (s)']
        te = row['End Time File (s)']
        anns.append(
            {
                'class': row['From'],
                't_start': tb,
                't_end': te,
                'f_start': row['Low Freq (Hz)'],
                'f_end': row['High Freq (Hz)'],
                'overlap': (min(te, t2) - max(tb, t1)) / (te - tb),
                'duration': te - tb,
                'freq_range': row['High Freq (Hz)'] - row['Low Freq (Hz)'],
                'fname': fname
            }
        )
    return anns

def get_random_annotation(cls) -> Tuple[float, float]:
    ann = ANN_DF[ANN_DF['From'] == cls].sample(1)
    return ann['Begin Time File (s)'].item(), ann['End Time File (s)'].item(), ann['Begin File'].item()


if __name__ == "__main__":
    # import os
    # BM_D_NO_DS_PATH = 'tmp/bm-d-no-ds'
    # BM_D_NO_DS_PKL_FILES = os.listdir(BM_D_NO_DS_PATH) 

    # BM_ANT_NO_DS_PATH = 'tmp/bm-ant-no-ds'
    # BM_ANT_NO_DS_PKL_FILES = os.listdir(BM_ANT_NO_DS_PATH) 
    
    # f_plot = BM_D_NO_DS_PKL_FILES[2]
    # anns = get_annotations(f_plot[:-4], 0, 1e6)
    # print(anns)
    print(get_random_annotation('D'))