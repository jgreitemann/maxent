import numpy as np

def imagt_kernel(taus, omegas, beta):
    domega = omegas[1] - omegas[0]
    return domega * np.exp(-np.outer(taus, omegas)) / (1 + np.exp(-beta * omegas))

