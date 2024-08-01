# a collection of parameters for the global consistency of all tests and scripts

# we use pydantic to be explicit about how parameters are set and calculated
from pydantic import BaseModel, computed_field
import math

FS = 250 # the base sampling frequency of the dataset


class Parameters(BaseModel):
    
    cls: str # the class to which these parameters belong to
    
    d: int # the downsampling factor of the TF decomposition for both WS and STFT
    d_audio: int # the downsampling factor of the input audio before calculating WS or STFT
    
    f0: float # the start frequency of detection (Hz)
    f1: float # the end frequency of detection (Hz)
    
    Th: float# SE stabilisation median filter half window length (s)
    
    Tmin: float # minimum detection length (s)
    Tmax: float # maximum detection length (s)
    Text: float = 0 # detection boundary extension for classification after removing short/long calls (s)
    
    Q1: float # WS first level filters per octave
    Q2: float # WS second level filters per octave
    
    T_clf_w: float # classification window length (s)
    N_clf_seg: int = 3 # the number of classification time segments in the identified window to be pooled
    pooling_type: str = 'mean' # the type of pooling to perform on the classification window (max/median/mean)  
    normalise_s2: bool = False # whether to normalise scattering level 2 as s2 / s1 
    log_scattering_coeffs: bool = True # whether to use the logarithm of scattering coeffs
    eps: float = 1e-12 # numerical safety when taking the logarithm of the scattering coeffs
    scattering_dct: bool = True # whether to take the DCT of the coeffs (after log and normalisation)
    normalise_feature_vector: bool = False # whether the scattering feature vec sould be normalised

    
    Mf: int # tone suppression median filter half window length for AW (in frequency bins)
    Tt: float # transient suppresion median filter half window length for AQ (s)
    Tn: float # rolling average half window length for AW to compute estimated noise power (s)    
    
    nu_proposed_se: float = 2.0 # nu for the SE calculation in the proposed detector
    
    kappa: float = 0.85 # percentage of samples to use for the proposed detector's noise SE beta distribution estimate
    
    
    
    @computed_field
    @property
    def fs_audio(self) -> float: return FS / self.d_audio
    
    @computed_field
    @property
    def fs_tf(self) -> float: return self.fs_audio / self.d
    
    @computed_field
    @property
    def Mn(self) -> int: return int(self.fs_tf*self.Tn)
    
    @computed_field
    @property
    def Mt(self) -> int: return int(self.fs_tf*self.Tt)
    
    @computed_field
    @property
    def Mh(self) -> int: return int(self.fs_tf*self.Th)
    
    @computed_field
    @property
    def Nfft(self) -> int: return int(self.d * 2)
    
    @computed_field
    @property
    def k0_stft(self) -> int: return int(self.f0 / self.fs_audio * self.Nfft)
    
    @computed_field
    @property
    def k1_stft(self) -> int: return int(self.f1 / self.fs_audio * self.Nfft)
    
    @computed_field
    @property
    def N_clf_win(self) -> int: # ensure the clf window length is divisible by the number of segments
        n = int(self.T_clf_w * self.fs_tf)
        return math.ceil(n / self.N_clf_seg) * self.N_clf_seg
    
    @computed_field
    @property
    def N_clf_seg_length(self) -> int: return self.N_clf_win//self.N_clf_seg
        
        
    
    
    
BM_D_PARAMETERS = Parameters(
    d=32, # 128 ms
    d_audio=1, # 250 Hz sampling rate
    f0=35,
    f1=120,
    Tmin=1.0,
    Tmax=7,
    # Text=0.25,
    Th=0.75,
    Q1=12,
    Q2=4,
    Mf = 1,
    Tn = 60*5,
    Tt = 0.5,
    cls = 'D',
    T_clf_w = 3,
    N_clf_seg = 5,
)

BM_ANT_PARAMETERS = Parameters(
    d=64, # 512 ms 
    d_audio=2, # 125 Hz sampling rate
    f0=15,
    f1=40,
    Tmin=3,
    Tmax=20,
    # Text=1.25,
    Th=1.25,
    Q1=12,
    Q2=6,
    Mf = 1,
    Tn = 60*10,
    Tt = 2.0,
    T_clf_w = 10,
    cls = 'A',
    N_clf_seg = 5,
)