import numpy as np
from scipy.special import digamma


K = 10

N_trails = 10000
alpha = 0.5

X = np.random.dirichlet([alpha]*K, N_trails)

print(np.mean(X, axis=0))

H = -np.sum(X * np.log(X),axis=1)

print(H)

mc_m1 = np.mean(H)
mc_m2 = np.mean(H**2)
print(mc_m1, mc_m2)

m1 = digamma(K*alpha + 1) - digamma(alpha + 1)
print(m1)
