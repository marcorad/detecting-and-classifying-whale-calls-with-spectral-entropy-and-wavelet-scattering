import numpy as np
from numpy.fft import fft

def sample_gauss(t, sigma):
    """
    Sample a 0-mean Gaussian at time instances t with SD sigma.
    """
    return 1.0 / np.sqrt(2.0 * np.pi) / sigma * np.exp(-0.5 * (t / sigma)**2.0)
    
def sample_morlet(t, lambda_, sigma):
    """
    Sample a morlet wavelet at time instances t with scaling factor lambda_ and Guassian envelope SD sigma.
    """
    t_times_lambda = t * lambda_
    g = sample_gauss(t_times_lambda, sigma)
    beta = (sample_gauss(-1.0, 1.0 / sigma) / sample_gauss(0.0, 1.0 / sigma))
    return lambda_ * (np.exp(t_times_lambda*(1j)) - beta) * g

def morlet_filter_freq(N, lambda_, sigma):
    """
    Generate a discrete morlet filter in the frequency domain containing N samples with scaling factor lambda_ and Guassian envelope SD sigma. N should be the convolution length. 
    This filter only has real frequency components (IR has Hermitian symmetry with respect to half of the buffer length).
    """
    hN = N//2
    n = np.arange(-hN, N - hN)
    morlet = sample_morlet(n, lambda_, sigma)
    return np.abs(fft(morlet))

def gauss_filter_freq(N, sigma):
    """
    Generate a discrete Gaussian filter in the frequency domain containing N samples with SD sigma. N should be the convolution length. 
    This filter only has real frequency components (IR has Hermitian symmetry with respect to half of the buffer length).
    """
    hN = N//2
    n = np.arange(-hN, N - hN)
    gauss = sample_gauss(n, sigma)
    return np.abs(fft(gauss))

