from .config import cfg
from .torch_backend import TorchBackend
from .filterbank import scattering_filterbanks, get_Lambda_set, get_wavelet_filter, filterbank_to_tensor, calculate_padding_1d, get_output_downsample_factor, calculate_sigma_psi_w, calculate_sigma_phi_w
from typing import List, Tuple, Dict
from torch import Tensor

class Scattering1D:
    backend: TorchBackend = TorchBackend()
    def __init__(self, N: List[int], d: List[int], Q: List[List[float]], startfreq: List[float] = None, allow_ds = True) -> None:
        """Create a separable scattering object, which precalculates the filters.

        Args:
            N (List[int]): The size of each dimension in scattering, corresponding to the input shape (Nbatch, *N).
            d (List[int]): A list of downsampling factors for each dimension, ordered according their appearance in the input.
            Q (List[List[float]]): A list containing the Qs used for each level of the scattering operation. Each item should be a list with Q corresponding to a specific dimension.
            startfreq (List[float], optional): The starting frequencies to place the filters of first-level scattering. When None, the frequency domain is fully covered. Defaults to None.
            allow_ds (bool, optional): Allow downsampling to occur for efficient computation. Defaults to True.
        """
        self.pad = []
        self.d = d
        self.Q = Q
        self.Nlevels = len(Q)
        self.allow_ds = allow_ds
        assert self.Nlevels <= 3, f'Requested {self.Nlevels} scattering levels. A maximum of 3 levels is supported. More than 3 levels are typically not useful.'
        
        l, r, n = calculate_padding_1d(N, d)
        self.pad.extend([l, r])
        self.Npad = n
            
        self.fb = scattering_filterbanks(self.Npad, d, Q, startfreq, allow_ds)
        filterbank_to_tensor(self.fb)
        
    def _mul_and_downsample(self, X: Tensor, level: int, input_ds: int, lambdas: float):        
        mul1d = lambda x, y: self.backend.mul1d(x, y, -1)
        freqds = lambda x, d: self.backend.freq_downsample1d(x, d, -1)
        filter = get_wavelet_filter(self.fb, level, input_ds, lambdas)  
        ds = get_output_downsample_factor(self.fb, level, input_ds, lambdas)
        X = mul1d(X, filter)
        if self.allow_ds: X = freqds(X, ds)
        return X
    
    def _ifft_mul_and_downsample(self, X: Tensor, level: int, input_ds: int, lambdas: float):
        mul1d = lambda x, y: self.backend.mul1d(x, y, -1)
        freqds = lambda x, d: self.backend.freq_downsample1d(x, d, -1)
        ifft1d = lambda x: self.backend.ifft1d(x, -1)
        filter = get_wavelet_filter(self.fb, level, input_ds, lambdas)  
        ds = get_output_downsample_factor(self.fb, level, input_ds, lambdas)
        X = mul1d(X, filter)
        if self.allow_ds: X = freqds(X, ds)
        X = ifft1d(X)
        return X    
    
    
    def _get_compounded_downsample_factor(self, level, current_ds, lambdas) -> int:
        if self.allow_ds:
            return get_output_downsample_factor(self.fb, level, current_ds, lambdas) * current_ds
        return 1
    
    def _should_prune(self, lambda_filt: List[float], lambda_demod: List[float], level: int):        
        beta = cfg.get_beta(self.Q[level])
        sigma_psi_w_demod = max(calculate_sigma_psi_w(self.Q[level-1]) * abs(lambda_demod), calculate_sigma_phi_w(self.d, self.Q[level-1]))
        # prune only when demodulated filter's centre freq is not at least within beta standard deviations of the current filter
        # note that we use the abs value of the lambdas since lambdas can be positive or negative
        # |----------*----------|
        # |-----ddddd*ddddd-----|
        #            <---->          
        # |----------*----x-----|
        #               <->       
        # these intervals must overlap, 
        # * is the centre of the spectrum, d is the significant bandwidth of the demodulated filter (via modulus), and f is the morlet filter under consideration which has a center x
        EPS = 1e-9 #for floating point error
        if beta*sigma_psi_w_demod < abs(lambda_filt) + EPS: return True 
        return False
    
    def scattering(self, x: Tensor, normalise = False) -> Tensor:
        """Perform a separable scattering transform on a real signal x.

        Args:
            x (Tensor): A tensor with shape (Nbatch, ...), the dimensions from 1 onwards are the scattering dimensions.
            normalise (bool, optional): normalise scattering coefficients with respect to the previous level. Defaults to False.

        Returns:
            Tensor: The scattering features, with the last axis corresponding to the various filters paths.
        """
        S, _, _, _ = self._scattering(x, False, False, normalise)        
        return self.backend.stack(S)   
    
    
    def _normalise(self, x1: Tensor, xn: Tensor):
        EPS = 1e-10
        return x1 / (xn + EPS)
        
    def _scattering(self, x: Tensor, returnU = False, returnSpath = False, normalise=False):      
        
        # Kymatio's scattering has a near-identical implementation
          
        #function aliases for clarity        
        unpad = lambda x: self.backend.unpad(x) if self.allow_ds else x #disable unpadding when DS occurs
        pad = lambda x, s: self.backend.pad(x, s)
        fft = lambda x: self.backend.fft(x, -1)
        ifft = lambda x: self.backend.ifft(x, -1)        
        mulds = lambda x, level, ids, lambdas: self._mul_and_downsample(x, level, ids, lambdas)
        modulus = lambda x: self.backend.modulus(x)        
        
        #pad the tensor
        x = pad(x, self.pad)
        #get the fft of the input signal across all dimensions
        X = fft(x)
        s_0 = unpad(ifft(mulds(X, 0, 1, 0)).real)
        S = [s_0]
        Up = {}
        Sp = {}
        
        l0_compounded_ds = 1 #no downsampling on the input
        
        if returnSpath: Sp[0] = s_0
        
        #first level
        Lambda_1 = get_Lambda_set(self.fb, 0, 1)
        for lambda1 in Lambda_1:            
            u_1 = modulus(ifft(mulds(X, 0, l0_compounded_ds, lambda1)))
            U_1 = fft(u_1)
            l1_compounded_ds = self._get_compounded_downsample_factor(0, l0_compounded_ds, lambda1)
            s_1 = ifft(mulds(U_1, 0, l1_compounded_ds, 0)).real
            s_1 = unpad(s_1)
            if normalise: s_1 = self._normalise(s_1, s_0)
            S.append(s_1)            
            
            if returnU:     Up[lambda1] = u_1
            if returnSpath: Sp[lambda1] = s_1
            
            if self.Nlevels == 1: continue
            # print(f'{lambda1} ->')
            #second level
            Lambda_2 = get_Lambda_set(self.fb, 1, l1_compounded_ds)
            for lambda2 in Lambda_2:                
                if self._should_prune(lambda2, lambda1, 1): continue #prune the paths, since downsampling prunes to an inexact extent
                # print(f'\t{lambda2}')
                u_2 = modulus(ifft(mulds(U_1, 1, l1_compounded_ds, lambda2)))
                U_2 = fft(u_2)
                l2_compounded_ds = self._get_compounded_downsample_factor(1, l1_compounded_ds, lambda2)
                s_2 = unpad(ifft(mulds(U_2, 1, l2_compounded_ds, 0)).real)
                if normalise: s_2 = self._normalise(s_2, s_1)
                S.append(s_2)   
                
                if returnU:     Up[(lambda1, lambda2)] = u_2
                if returnSpath: Sp[(lambda1, lambda2)] = s_2
                
                if self.Nlevels == 2: continue
                
                #third level
                Lambda_3 = get_Lambda_set(self.fb, 2, l2_compounded_ds)
                for lambda3 in Lambda_3:    
                    if self._should_prune(lambda3, lambda2, 2): continue #prune the paths
                    u_3 = modulus(ifft(mulds(U_2, 2, l2_compounded_ds, lambda3)))
                    U_3 = fft(u_3)
                    l3_compounded_ds = self._get_compounded_downsample_factor(2, l2_compounded_ds, lambda3)
                    s_3 = unpad(ifft(mulds(U_3, 2, l3_compounded_ds, 0)).real)
                    if normalise: s_3 = self._normalise(s_3, s_2)
                    S.append(s_3) 
                    
                    if returnU:     Up[(lambda1, lambda2, lambda3)] = u_3
                    if returnSpath: Sp[(lambda1, lambda2, lambda3)] = s_3
                    
        return S, Sp, Up, x
        
        
        
        