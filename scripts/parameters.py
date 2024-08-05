# a collection of parameters for the global consistency of all tests and scripts

# we use pydantic to be explicit about how parameters are set and calculated
from pydantic import BaseModel, computed_field
import math

FS = 250 # the base sampling frequency of the dataset


class Parameters(BaseModel):
    
    cls: str # the class to which these parameters belong to
    d_audio: int # the downsampling factor of the input audio before calculating WS or STFT
    
    # DETECTION
    
    d_det: int # the downsampling factor of the TF decomposition for detection
    
    f0: float # the start frequency of detection (Hz)
    f1: float # the end frequency of detection (Hz)
    
    Th: float# SE stabilisation median filter half window length (s)
    
    Tmin: float # minimum detection length (s)
    Tmax: float # maximum detection length (s)
    Text: float = 0 # detection boundary extension for classification after removing short/long calls (s)
    
    Q1_det: float # WS first level filters per octave
    
    Mf: int # tone suppression median filter half window length for AW (in frequency bins)
    Tt: float # transient suppresion median filter half window length for AQ (s)
    Tn: float # rolling average half window length for AW to compute estimated noise power (s)    
    
    nu_proposed_se: float = 1.0 # nu for the SE calculation in the proposed detector
    
    kappa: float = 0.85 # percentage of samples to use for the proposed detector's noise SE beta distribution estimate
    
    #CLASSIFICATION
    
    d_clf: int # the downsampling factor of the WS features for each detection window
    Q1_clf: float
    Q2_clf: float
    T_clf_win: float
    normalise_s2: bool = False # whether to normalise scattering level 2 as s2 / s1 
    log_scattering_coeffs: bool = True # whether to use the logarithm of scattering coeffs
    eps: float = 1e-12 # numerical safety when taking the logarithm of the scattering coeffs
    scattering_dct: bool = True # whether to take the DCT of the coeffs (after log and normalisation)
    normalise_clf_audio: bool = True
    time_dct: bool = False
    gamma_clf: float =  0.8
    rho_clf: float = 0.0001   
    
    @computed_field
    @property
    def fs_audio(self) -> float: return FS / self.d_audio
    
    @computed_field
    @property
    def fs_tf_det(self) -> float: return self.fs_audio / self.d_det
    
    @computed_field
    @property
    def fs_tf_clf(self) -> float: return self.fs_audio / self.d_clf
    
    @computed_field
    @property
    def Mn(self) -> int: return int(self.fs_tf_det*self.Tn)
    
    @computed_field
    @property
    def Mt(self) -> int: return int(self.fs_tf_det*self.Tt)
    
    @computed_field
    @property
    def Mh(self) -> int: return int(self.fs_tf_det*self.Th)
    
    @computed_field
    @property
    def Nfft(self) -> int: return int(self.d_det * 2)
    
    @computed_field
    @property
    def k0_stft(self) -> int: return int(self.f0 / self.fs_audio * self.Nfft)
    
    @computed_field
    @property
    def k1_stft(self) -> int: return int(self.f1 / self.fs_audio * self.Nfft)
    
    @computed_field
    @property
    def N_clf_win(self) -> int: 
        return int(self.T_clf_win * self.fs_audio)
    
        
        

BM_ANT_PARAMETERS = Parameters(
    cls = 'A',
    d_det=64, # 512 ms 
    d_audio=3, # 83 1/3 Hz sampling rate
    f0=15,
    f1=40,
    Tmin=3,
    Tmax=20,
    # Text=1.25,
    Th=1.25,
    Q1_det=12,
    Mf = 1,
    Tn = 60*10,
    Tt = 2.0,
    
    d_clf=512,
    Q1_clf=4,
    Q2_clf=2,
    T_clf_win=9,
    rho_clf = 0.1,
    gamma_clf =  0.5,
)    
    

   
BM_D_PARAMETERS = Parameters(
    cls = 'D',
    d_det=32, # 128 ms
    d_audio=1, # 250 Hz sampling rate
    f0=35,
    f1=120,
    Tmin=2,
    Tmax=7,
    Th=0.75,
    Q1_det=12,
    Mf = 1,
    Tn = 60*5,
    Tt = 0.5,
    
    d_clf=512,
    Q1_clf=2,
    Q2_clf=1,
    T_clf_win=5,
    rho_clf = 0.01,
    gamma_clf =  0.5
)
