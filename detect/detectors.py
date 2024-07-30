from torch import Tensor
import torch 
import torch.nn.functional as func
from math import ceil, floor, log
from typing import List, Dict
import numpy as np


class Callable:
    def __init__(self) -> None:
        pass
    
    def apply(self, X: Tensor) -> Tensor: 
        # subclasses must override this function
        return X
    
    def __call__(self, X: Tensor) -> Tensor:
        return self.apply(X)

class DetectorBase(Callable):
    def __init__(self, t_dim=0, f_dim=1) -> None:
        super().__init__()
        self.t_dim = t_dim
        self.f_dim = f_dim   
    
    def _get_tf_sizes(self, x: Tensor):
        return x.shape[self.t_dim], x.shape[self.f_dim]
    
    def _sum_time(self, x: Tensor):
        return torch.sum(x, self.t_dim, keepdim=False)
        
    def _sum_freq(self, x: Tensor):
        return torch.sum(x, dim=self.f_dim, keepdim=False)    
    
class NarrowSelection(DetectorBase):
    def __init__(self, t_dim=0, f_dim=1, n0=0, n1=-1, k0=0, k1=-1) -> None:
        super().__init__(t_dim, f_dim)
        self.n0 = n0
        self.n1 = n1
        self.k0 = k0
        self.k1 = k1
        
    def apply(self, x: Tensor):
        K = x.shape[self.f_dim]
        k0 = self.k0 % K
        k1 = self.k1 % K        
        L = x.shape[self.t_dim]
        n0 = self.n0 % L
        n1 = self.n1 % L
        return x.narrow(self.f_dim, start=k0, length=(k1-k0+1)).narrow(self.t_dim, start=n0, length=(n1-n0+1))
    
def _rolling_func(X: Tensor, dim, M, f, pad_mode = 'reflect', val=None):
    if len(X.shape) == 1: 
        X = X[:, None]
    L = X.shape[dim]
    Xc = X.swapaxes(-1, dim)
    W = 2 * M + 1
    padded_len = Xc.shape[-1] + W
    # pad_zeros = ceil(padded_len / W) * W - padded_len
    Xc = func.pad(Xc, (M, M), mode=pad_mode, value=val)
    # Xc = func.pad(Xc, (0, pad_zeros)).contiguous() #add zeros for even division
    Xu = Xc.unfold(-1, W, 1) # last dimension is of the window size
    y : Tensor
    if f == torch.median or f == torch.max or f == torch.min:
        y, _ = f(Xu, dim=-1, keepdim=False) # compute the function over this dimension
    else: y = f(Xu, dim=-1, keepdim=False)
    return y.narrow(-1, 0, L).swapaxes(-1, dim).contiguous() # return the computed value with original shape intact

class RollingFunction(Callable):
    def __init__(self, dim, M, f, pad_mode='reflect', val=None) -> None:
        super().__init__()
        self.f = lambda x: _rolling_func(x, dim, M, f, pad_mode, val)
        
    def apply(self, X: Tensor) -> Tensor:
        return self.f(X)

class RollingMedian(RollingFunction):
    def __init__(self, dim, M) -> None:
        super().__init__(dim, M, torch.median)
    
class RollingAverage(RollingFunction):
    def __init__(self, dim, M) -> None:
        super().__init__(dim, M, torch.mean)         
   

class SpectralWhitener(DetectorBase):
    def __init__(self, M_f, M_t,  M_n, nu=1, t_dim=0, f_dim=1) -> None:
        super().__init__(t_dim, f_dim)   
        self.nu = nu
        self.frequency_suppressor = RollingMedian(self.f_dim, M_f)
        self.transient_suppressor = RollingMedian(self.t_dim, M_t)
        self.noise_estimator = RollingAverage(self.t_dim, M_n)
        
    def apply(self, X: Tensor) -> Tensor:
        xf = self.frequency_suppressor(X)
        xt = self.transient_suppressor(xf)
        self.noise = self.noise_estimator( 
                                     xt ** (2*self.nu) 
                                    ) ** (1 / 2 / self.nu) 
        return X / self.noise 
    
    
class SpectralEntropy(DetectorBase):
    def __init__(self, nu = 1, eps = 1e-15, t_dim=0, f_dim=1) -> None:
        self.nu = nu
        self.eps = eps
        super().__init__(t_dim, f_dim)
        
    def apply(self, X: Tensor) -> Tensor: 
        P = self._P(X)
        return -torch.sum(P * torch.log(P + self.eps), dim=self.f_dim, keepdim=False)
    
    def _P(self, X: Tensor) -> Tensor:
        X_nu = X.abs() ** (2 * self.nu)
        P = X_nu / self._sum_freq(X_nu)
        return P
    
class Nuttall(DetectorBase): 
    #A. Nuttall "Detection performance of power-law processors for random signals of unknown location, structure, extent, and strength", 1994
    #A. Nuttall "Near-optimum detection performance of power-law process-ors for random signals of unknown locations, structure, extent, and arbitrary strengths", 1996
    def __init__(self, nu = 2.5, t_dim=0, f_dim=1) -> None: 
        # nu > 2.5 for narrow-band sounds
        # nu = 2.5 for general-purpose
        # nu = 1 optimal for sounds half the bandwidth of the spectrum
        # note that BLED is a subclass of nuttall, with nu = 1 and [k1, k2] not covering the entire spectrum
        self.nu = nu
        super().__init__(t_dim, f_dim)
        
    def apply(self, X: Tensor) -> Tensor:
        #Implementation from Tyler A. Helble et. al., "A generalized power-law detection algorithm for humpback whale vocalizations", JASA 2012
        return self._sum_freq(X.abs() ** (2 * self.nu))
    

    
class HelbleWhitening(DetectorBase):
    #Tyler A. Helble et. al., "A generalized power-law detection algorithm for humpback whale vocalizations", JASA 2012
    def __init__(self, t_dim=0, f_dim=1, gamma=1) -> None:
        super().__init__(t_dim, f_dim)
        self.gamma = gamma
        
    def apply(self, X: Tensor) -> Tensor:
        X_abs = X.abs()
        return (X_abs**self.gamma - self._mu_k(X_abs)).abs()
    
    def _mu_k(self, X_abs: Tensor):
        L = X_abs.shape[self.t_dim]
        X_sort, _ = torch.sort(X_abs, dim=self.t_dim)
        j_min = L//4
        return torch.mean(X_sort.narrow(self.t_dim, j_min, L//2), dim=self.t_dim, keepdim=True)
    
  
class GPLBase(DetectorBase):
    # Tyler A. Helble et. al., "A generalized power-law detection algorithm for humpback whale vocalizations", JASA 2012
    def __init__(self, nu_1=1, nu_2=2, t_dim=0, f_dim=1) -> None:
        super().__init__(t_dim, f_dim)  
        self.nu_1 = nu_1
        self.nu_2 = nu_2
        
    def apply(self, X: Tensor) -> Tensor:
        return self._sum_freq(self._N(X))
    
    def _N(self, X: Tensor):
        X = X.abs()
        A = X / torch.sqrt(torch.sum(X ** 2, dim=self.t_dim, keepdim=True))
        B = X / torch.sqrt(torch.sum(X ** 2, dim=self.f_dim, keepdim=True))
        N = (A ** (2 * self.nu_1)) * (B ** (2 * self.nu_2))
        return N 
 
def prob_not_drawn_from_noise_normal(X: Tensor, mu: Tensor | float, sigma: Tensor | float) -> Tensor:
    # one-tailed probability that X is not drawn from the noise distribution
    Z = (X - mu)/sigma
    normal_cdf = lambda z: 0.5 * (1 + torch.erf(z))
    return 1 - normal_cdf(Z) 
    
class AdaptiveSignalProbNormal(Callable):
    def __init__(self, M) -> None:
        super().__init__() 
        self.rolling_median = RollingMedian(0, M)
    
    def apply(self, X: Tensor) -> Tensor:
        mu = self.rolling_median.apply(X) # rolling median to estimate the instantaneous mean
        sigma = self.rolling_median.apply((X - mu).abs()) * 1.4826 # estimate a rolling std using a rolling median absolute deviation (MAD)
        return prob_not_drawn_from_noise_normal(X, mu, sigma)
        
    
class SignalProbNormal(Callable):
    def __init__(self) -> None:
        super().__init__()
    
    def apply(self, X: Tensor) -> Tensor:
        mu = torch.median(X) # mean of entire sequence estimated by median
        sigma = torch.median((X - mu).abs()) * 1.4826 # std estimated with median absolute deviation (MAD)
        return prob_not_drawn_from_noise_normal(X, mu, sigma)
    
class SignalProbPDF(Callable):
    def __init__(self, kappa) -> None:
        super().__init__()
        self.kappa = kappa
        
    def _get_estimate_moments(self, X: Tensor):
        L = X.shape[0]
        N = ceil(self.kappa * L)
        X_d, _ = X.sort()
        X_d = X_d[:N] # the data to use for the estimate        
        mu = torch.mean(X_d)
        var = torch.var(X_d)
        return mu, var
    
class SignalProbGamma(SignalProbPDF):
    def __init__(self, kappa) -> None:
        super().__init__(kappa)
        
    def apply(self, X: Tensor) -> Tensor:
        mu, var = self._get_estimate_moments(X)
        # alpha and beta related to mean and variance
        alpha = (mu / var) ** 2
        beta = alpha / mu
        Gamma = torch.distributions.gamma.Gamma(alpha, beta)
        return Gamma.cdf(X)
    
class EntropyScaler(Callable):
    # scales SE between 0-1 for easier threshold selection
    
    def __init__(self, K) -> None:
        super().__init__()        
        self.K = K    
        
    def apply(self, X: Tensor) -> Tensor:        
        return 1 - X / log(self.K)
    
class SignalProbBeta(SignalProbPDF):
    def __init__(self, kappa, minimum: float, maximum: float, invert = False, sig = 0.05) -> None:
        super().__init__(kappa)        
        self.max = maximum
        self.min = minimum
        self.invert = invert
        self.sig = sig
        
    def build_cdf_lut(self, N=1023):
        x = torch.linspace(0, 1, N)
        pdf = self.beta_dist.log_prob(x).exp()
        self.beta_cdf_lut = torch.cumsum(pdf, dim=0).numpy() / N
        self.crit_val = np.interp(1 - self.sig, self.beta_cdf_lut, x) # find the inverse cdf critical value
        
        
    def apply(self, X: Tensor) -> Tensor:
        
        # this probability models the noise ditribution of the statistic X provided as (X - X_min) / (X_max - X_min) ~ Beta(alpha, beta)
        # if inverted then 1 - (X - X_min) / (X_max - X_min) ~ Beta(alpha, beta)
        
        X = (X - self.min) / (self.max - self.min)
        if self.invert: X = 1 - X
        
        mu, var = self._get_estimate_moments(X)
        # alpha and beta related to mean and variance
        nu = mu * (1 - mu) / var -1
        alpha = mu * nu
        beta = (1 - mu) * nu
        self.beta_dist = torch.distributions.beta.Beta(alpha, beta)
        self.build_cdf_lut()
        return X >= self.crit_val
    
class DetectorChain(Callable):
    def __init__(self, chain: List[Callable]) -> None:
        self.chain = chain
        self.results: List[Tensor] = []
        
    def apply(self, X: Tensor) -> Tensor:
        for c in self.chain:
            X = c(X)
            self.results.append(X)
        return X
    
    def get_statistic(self):
        return self.results[-1]
    
def helble_gpl(nu_1=1, nu_2=2, gamma=1, t_dim=0, f_dim=1, k0=0, k1=-1) -> DetectorChain:
    # Tyler A. Helble et. al., "A generalized power-law detection algorithm for humpback whale vocalizations", JASA 2012
    # This function returns the detector as presented in their work
    # This detector performs whitening via the HelbleWhitening procedure (compute the noise mean of mu_k |X|, for each frequency bin k over time, then compute ||X| - mu_k|)
    # After whitening, the normalised "spectrogram" N is summed to produce the test statistic
    return DetectorChain([
            NarrowSelection(k0=k0, k1=k1, t_dim=t_dim, f_dim=f_dim ),
            HelbleWhitening(t_dim, f_dim, gamma), 
            GPLBase(nu_1, nu_2, t_dim, f_dim)
        ])  
    
def aw_gpl(Mf, Mt, Mn, nu_1=1, nu_2=2, t_dim=0, f_dim=1, k0=0, k1=-1) -> DetectorChain:
    return DetectorChain([
        NarrowSelection(k0=k0, k1=k1, t_dim=t_dim, f_dim=f_dim ),
        SpectralWhitener(Mf, Mt, Mn, nu=1, t_dim=t_dim, f_dim=f_dim), 
        GPLBase(nu_1, nu_2, t_dim, f_dim)]) 

def proposed_detector(k0, k1, Mf, Mt, Mn, Mh, sig, t_dim=0, f_dim=1, nu=2, kappa = 0.85) -> DetectorChain:
    # the detector proposed by this paper
    # 1. select relevant frequency indices k0 to k1
    # 2. whiten the spectrum with the proposed whitening method
    # 3. compute spectral entropy as test statistic
    # 4. convert test statistic to the probability that the entropy is not drawn from the noise entropy distribution
    return DetectorChain([
        NarrowSelection(k0=k0, k1=k1, t_dim=t_dim, f_dim=f_dim ),
        SpectralWhitener(Mf, Mt, Mn, nu=1, t_dim=t_dim, f_dim=f_dim), 
        SpectralEntropy(nu=nu, t_dim=t_dim, f_dim=f_dim), 
        RollingMedian(0, Mh),
        SignalProbBeta(kappa=kappa, minimum=0, maximum=log(k1 - k0 + 1), invert=True, sig=sig),
        ]) 
    
def se(k0, k1, t_dim=0, f_dim=1, nu=1) -> DetectorChain:
    return DetectorChain([
        NarrowSelection(k0=k0, k1=k1, t_dim=t_dim, f_dim=f_dim ),
        SpectralEntropy(nu=nu, t_dim=t_dim, f_dim=f_dim), 
        EntropyScaler(k1 - k0 + 1)
        ]) 
    
def nuttall(k0, k1, nu = 2.5, t_dim=0, f_dim=1) -> DetectorChain:
    return DetectorChain(
        [
            NarrowSelection(k0=k0, k1=k1, t_dim=t_dim, f_dim=f_dim ),
            Nuttall(nu, t_dim=t_dim, f_dim=f_dim)
        ]
    )
    
def bled(k0, k1, t_dim=0, f_dim=1) -> DetectorChain:
    # BLED is a special case of Nuttall, where we sum the squares
    return DetectorChain(
        [
            NarrowSelection(k0=k0, k1=k1, t_dim=t_dim, f_dim=f_dim ),
            Nuttall(1, t_dim=t_dim, f_dim=f_dim)
        ]
    )

def aw_nuttall(k0, k1, Mf, Mt, Mn, nu = 2.5, t_dim=0, f_dim=1) -> DetectorChain:
    return DetectorChain(
        [
            NarrowSelection(k0=k0, k1=k1, t_dim=t_dim, f_dim=f_dim ),
            SpectralWhitener(Mf, Mt, Mn, nu=1, t_dim=t_dim, f_dim=f_dim), 
            Nuttall(nu, t_dim=t_dim, f_dim=f_dim)
        ]
    )
    
def aw_bled(k0, k1, Mf, Mt, Mn, t_dim=0, f_dim=1) -> DetectorChain:
    # BLED is a special case of Nuttall, where we sum the squares
    return DetectorChain(
        [
            SpectralWhitener(Mf, Mt, Mn, nu=1, t_dim=t_dim, f_dim=f_dim), 
            NarrowSelection(k0=k0, k1=k1, t_dim=t_dim, f_dim=f_dim ),
            Nuttall(1, t_dim=t_dim, f_dim=f_dim)
        ]
    )
    
def _find_start_end_pairs(T_th: Tensor):
    T_th[-1] = False #must always end the detection on the last sample, otherwise we won't have matching pairs
    prev = torch.zeros_like(T_th)
    prev[1:] = T_th[:-1]
    # edge detection
    start_mask = T_th & ~prev #whether the current sample is a new detection start, i.e., prev low -> curr high
    end_mask = ~T_th & prev #whether the current sample is a new detection end, i.e., prev high -> curr low
    start_idx = torch.nonzero(start_mask[:]).tolist()
    end_idx = torch.nonzero(end_mask[:]).tolist()
    return [s[0] for s in start_idx], [e[0] for e in end_idx]
    
def get_detections(T: Tensor, thresh, Tmin, Tmax, Text, fs):
    Nmin = floor(Tmin * fs)
    Nmax = floor(Tmax * fs)
    Next = floor(Text * fs)
    T = T.squeeze() # get rid of empty dims
    T_th = T > thresh
    start_idx, end_idx = _find_start_end_pairs(T_th)
    # remove short and long detections
    for i, j in zip(start_idx, end_idx):
        if j - i < Nmin or j - i > Nmax:
            T_th[i:j] = False
    # now do it again, except with boundary extension, using morphological dilation (rolling max)
    T_th = _rolling_func(T_th, 0, Next, torch.max, pad_mode='constant', val=0)
    start_idx, end_idx = _find_start_end_pairs(T_th)
    detections = [(i/fs, j/fs) for i, j in zip(start_idx, end_idx)]
    idx = [(i, j) for i, j in zip(start_idx, end_idx)]
    return detections, idx
    
    



    

        
        
    
