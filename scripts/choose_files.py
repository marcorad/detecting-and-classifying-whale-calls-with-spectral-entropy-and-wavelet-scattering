from config import ORIGINAL_BM_D_ANNOTATION_FILE, ORIGINAL_BM_ANT_A_ANNOTATION_FILE, ORIGINAL_BM_ANT_B_ANNOTATION_FILE, ORIGINAL_BM_ANT_Z_ANNOTATION_FILE, ORIGINAL_AUDIO_PATH, PROCESSED_DATASET_PATH

import pandas as pd
import os
import math
from typing import List

orig_audio_files = os.listdir(ORIGINAL_AUDIO_PATH)
bm_d_annotations = pd.read_csv(ORIGINAL_BM_D_ANNOTATION_FILE, delimiter='\t')
bm_d_annotations.loc[:, 'From'] = 'D'
bm_ant_a_annotations = pd.read_csv(ORIGINAL_BM_ANT_A_ANNOTATION_FILE, delimiter='\t')
bm_ant_b_annotations = pd.read_csv(ORIGINAL_BM_ANT_B_ANNOTATION_FILE, delimiter='\t')
bm_ant_z_annotations = pd.read_csv(ORIGINAL_BM_ANT_Z_ANNOTATION_FILE, delimiter='\t')
# join the Bm-Ant-x files
bm_ant_annotations = pd.concat([bm_ant_a_annotations, bm_ant_b_annotations, bm_ant_z_annotations])
bm_ant_annotations.loc[:, 'From'] = 'A'
# join all the Bm files
bm_annotations = pd.concat([bm_ant_annotations, bm_d_annotations], ignore_index=True, sort=False)

# print(bm_annotations.head())

n_target_files = 20

annotated_files = bm_annotations['Begin File'].unique()
n_ann_files = 0
n_files = len(orig_audio_files)
noise_files = []
for f in orig_audio_files:
    if f in annotated_files:
        n_ann_files+=1
    noise_files.append(f)
ann_prop = n_ann_files/n_files
print(f'{n_ann_files} contain Bm annotations out of {n_files} ({ ann_prop :.2%})')
n_noise_files = math.floor(n_target_files*(1 - ann_prop))
print(f'At least {n_noise_files} files must contain no annotations.')

print("Files with only noise:")
noise_files.sort()
even_noise_files = noise_files[5::n_files//n_noise_files]
print("\n".join(noise_files))
print("Evenely chosen noise files:")
print("\n".join(even_noise_files))

print('Evenly chosen annotation files:')
ann_counts = bm_annotations.groupby('Begin File')['Begin File'].count().sort_index()

n_ann_files = n_target_files - n_noise_files
win_size = n_files // n_ann_files
even_ann_files = []
for w in ann_counts.rolling(win_size, step=win_size):
    even_ann_files.append(w.index[w.argmax()])
print("\n".join(even_ann_files))

# now copy the files to a temp directory, to ensure we don't overwrite our workspace
import shutil
from scipy.signal import decimate
from scipy.io.wavfile import read, write
import librosa
d_factor = 4 # downsample to 250 Hz

TEMP_NOISE_PATH = PROCESSED_DATASET_PATH + '/tmp/noise'
TEMP_ANN_PATH = PROCESSED_DATASET_PATH + '/tmp/ann'
# remove anything already there
shutil.rmtree(TEMP_NOISE_PATH, ignore_errors=True)
shutil.rmtree(TEMP_ANN_PATH, ignore_errors=True)
# make fresh dirs
os.makedirs(TEMP_NOISE_PATH, exist_ok=False)
os.makedirs(TEMP_ANN_PATH, exist_ok=False)

def copy_and_downsample(filenames, src_path, dest_path, d_factor):
    for f in filenames:
        fs, x = read(src_path + '/' + f)
        xd = decimate(x, d_factor, ftype='iir', zero_phase=True)
        # now change the data type back to that of x, since decimate converts to float64
        xd = xd.round(0) # quantise the floats
        xd = xd.astype(x.dtype) # back to original data type (int16 in the case of casey2017)
        fsd = fs//d_factor
        write(dest_path + '/' + f, fsd, xd)
        
        
copy_and_downsample(noise_files, ORIGINAL_AUDIO_PATH, TEMP_NOISE_PATH, d_factor)
copy_and_downsample(even_ann_files, ORIGINAL_AUDIO_PATH, TEMP_ANN_PATH, d_factor)

# now also copy the annotations of those files for use in Raven Pro
def copy_annotations(ann_df: pd.DataFrame, ann_files: List[str], path, fname, d_factor):
    ann_df = ann_df.copy()[ann_df['Begin File'].isin(ann_files)]
    # we have to adjust the annotations according to the downsampling amount
    ann_df['Beg File Samp (samples)'] = ann_df['Beg File Samp (samples)']//d_factor
    ann_df['End File Samp (samples)'] = ann_df['End File Samp (samples)']//d_factor    
    # we also have to adjust the beginning and end time of the annotations, since the files are treated as consecutive in Raven
    fname_sorted = ann_files.copy()
    fname_sorted.sort()
    f_times = {}
    f_times[fname_sorted[0]] = {
            'fs': librosa.get_samplerate(path + '/' + fname_sorted[0]),
            'offset_time': 0
            }
    # read the duration of the previous files to find the base time offset and store in dict    
    for i in range(1, len(fname_sorted)):
        f_curr = fname_sorted[i]
        f_prev = fname_sorted[i-1]
        f_times[f_curr] = {
            'fs': librosa.get_samplerate(path + '/' + f_curr),
            'offset_time': librosa.get_duration(path=path + '/' + f_prev) + f_times[f_prev]['offset_time']
            }    
    # recompute the time
    for index, row in ann_df.iterrows():
        n1 = row['Beg File Samp (samples)']
        n2 = row['End File Samp (samples)']
        fs = f_times[row['Begin File']]['fs']        
        t0 = f_times[row['Begin File']]['offset_time']         
        ann_df.loc[index, 'Begin Time (s)'] = n1/fs + t0
        ann_df.loc[index, 'End Time (s)']   = n2/fs + t0        
    
    ann_df = ann_df.sort_values('Begin Time (s)').reset_index(drop=True)       
    ann_df['Selection'] = ann_df.index + 1 # renumber the selections, since we may have merged files, with Raven expecting it to start at 1
    ann_df.to_csv(path + '/' + fname, sep='\t', index=False) # tabs are standard for Raven Pro selection files
    
copy_annotations(bm_annotations, even_ann_files, TEMP_ANN_PATH, "Bm.selections.txt", d_factor)



        

