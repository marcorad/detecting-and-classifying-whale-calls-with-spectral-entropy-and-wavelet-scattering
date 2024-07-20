from config import ORIGINAL_BM_D_ANNOTATION_FILE, ORIGINAL_BM_ANT_A_ANNOTATION_FILE, ORIGINAL_BM_ANT_B_ANNOTATION_FILE, ORIGINAL_BM_ANT_Z_ANNOTATION_FILE, ORIGINAL_AUDIO_PATH

import pandas as pd
import os
import math

orig_audio_files = os.listdir(ORIGINAL_AUDIO_PATH)
bm_d_annotations = pd.read_csv(ORIGINAL_BM_D_ANNOTATION_FILE, delimiter='\t')
bm_d_annotations.loc[:, 'Label'] = 'Bm-D'
bm_ant_a_annotations = pd.read_csv(ORIGINAL_BM_ANT_A_ANNOTATION_FILE, delimiter='\t')
bm_ant_b_annotations = pd.read_csv(ORIGINAL_BM_ANT_B_ANNOTATION_FILE, delimiter='\t')
bm_ant_z_annotations = pd.read_csv(ORIGINAL_BM_ANT_Z_ANNOTATION_FILE, delimiter='\t')
# join the Bm-Ant-x files
bm_ant_annotations = pd.concat([bm_ant_a_annotations, bm_ant_b_annotations, bm_ant_z_annotations])
bm_ant_annotations.loc[:, 'Label'] = 'Bm-Ant'
# join all the Bm files
bm_annotations = pd.concat([bm_ant_annotations, bm_d_annotations])

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
