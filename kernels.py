import numpy as np

def imagt_kernel(taus, omegas, beta):
    return np.exp(-np.outer(taus, omegas)) / (1 + np.exp(-beta * omegas))

