#!/usr/bin/python2

import numpy as np
import kernels
import fileinput
import argparse
import itertools

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--beta", type=float)
parser.add_argument("--cutoff", type=float, default=1e-10)
args, unk = parser.parse_known_args()

# read in Green function data
taus = []
Gs = []
Gerrs = []
for line in fileinput.input(unk):
    words = line.split(' ')
    taus.append(float(words[0]))
    Gs.append(float(words[1]))
    Gerrs.append(float(words[2]))
taus = np.array(taus)
Gs = np.array(Gs)
Gerrs = np.array(Gerrs)

# perform SVD on the kernel
omegas = np.linspace(0, 10, 1001)
K = kernels.imagt_kernel(taus, omegas, args.beta)
V, Sigma, U = np.linalg.svd(K)
U = U.T.conj()

# project into singular space
Sigma = np.array(list(itertools.takewhile(lambda x: x > args.cutoff, Sigma)))
s = Sigma.size
V = V[:,:s]
U = U[:,:s]

