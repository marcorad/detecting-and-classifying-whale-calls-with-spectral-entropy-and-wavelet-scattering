# a collection of parameters for the global consistency of all tests and scripts

# we use pydantic to be explicit about how parameters are set and calculated
from pydantic import BaseModel, computed_field

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
    Text: float # detection boundary extension for classification after removing short/long calls (s)
    
    Q1: float # WS first level filters per octave
    Q2: float # WS second level filters per octave
    
    # default parameters - common to both bm-d and bm-ant classes
    
    Mf: int = 2 # tone suppression median filter half window length for AW (in frequency bins)
    Tt: float = 2 # transient suppresion median filter half window length for AQ (s)
    Tn: float = 60*5 # rolling average half window length for AW to compute estimated noise power (s)    
    
    nu_proposed_se: float = 2 # nu for the SE calculation in the proposed detector
    
    kappa: float = 0.85 # percentage of samples to use for the proposed detector's beta distribution estimate
    
    
    
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
    
    
    
BM_D_PARAMETERS = Parameters(
    d=32, # invariance downsampling factor of 32 ~ 128 ms
    d_audio=1, # 250 Hz sampling rate
    f0=30,
    f1=120,
    Tmin=0.75,
    Tmax=7,
    Text=0.25,
    Th=0.25,
    cls = 'D',
    Q1=8,
    Q2=4
)

BM_ANT_PARAMETERS = Parameters(
    d=64, # invariance downsampling factor of 64 ~ 1024 ms 
    d_audio=2, # 125 Hz sampling rate
    f0=15,
    f1=40,
    Tmin=3,
    Tmax=20,
    Text=1.25,
    Th=1.25,
    cls = 'A',
    Q1=12,
    Q2=4,
    Mf = 1,
    Tn = 60*10
)