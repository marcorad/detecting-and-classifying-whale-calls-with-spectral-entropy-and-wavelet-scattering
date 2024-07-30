import os
from typing import Dict, List
import pickle as pkl
from annotations import get_annotations

BM_D_PATH = 'tmp/bm-d'

BM_ANT_PATH = 'tmp/bm-ant'

class Loader:
    def __init__(self, path, cls) -> None:
        self.counter = 0
        self.path = path
        self.files = os.listdir(path)
        self.cls = cls
    
    def __next__(self):
        if self.counter == len(self.files):
            self.counter = 0
            raise StopIteration
        self.counter += 1
        return self.__getitem__(self.counter - 1)
    
    def __getitem__(self, i: int):
        fname = self.files[i]
        with open(f'{self.path}/{fname}', 'rb') as file:
            V = pkl.load(file) 
        anns = get_annotations(fname[:-4], 0, 10e6, self.cls)
        return V, anns 
    
    def __iter__(self):
        return self
    
def bm_d_ws_loader():
    return Loader(BM_D_PATH + '/ws', 'D')

def bm_d_stft_loader():
    return Loader(BM_D_PATH + '/stft', 'D')

def bm_ant_ws_loader():
    return Loader(BM_ANT_PATH + '/ws', 'A')

def bm_ant_stft_loader():
    return Loader(BM_ANT_PATH + '/stft', 'A')
        