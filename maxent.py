#!/usr/bin/python2

import numpy as np
import kernels
import fileinput
import argparse
import itertools

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float)
parser.add_argument("--beta", type=float)
parser.add_argument("--cutoff", type=float, default=1e-10)
args, unk = parser.parse_known_args()
alpha = args.alpha

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
W = np.diag(Gerrs**(-2))

# perform SVD on the kernel
omegas = np.linspace(0, 10, 1001)
K = kernels.imagt_kernel(taus, omegas, args.beta)
V, Sigma, U = np.linalg.svd(K)
U = U.T

# project into singular space
Sigma = np.array(list(itertools.takewhile(lambda x: x > args.cutoff, Sigma)))
s = Sigma.size
V = V[:,:s]
U = U[:,:s]
VSigma = np.dot(V, np.diag(Sigma))

# default model
ms = 0.1 * np.ones(1001)
norm = np.linalg.norm(ms, ord=1)

# Newton iteration
mu = 0.
u = np.zeros(s)
u[0] = 1.
for i in range(10):
    As = ms * np.exp(np.dot(U, u))
    F = np.dot(VSigma, np.dot(U.T, As))
    g = np.dot(VSigma.T, np.dot(W, (F - Gs)))
    T = np.dot(U.T, np.dot(np.diag(As), U))
    M = np.dot(VSigma.T, np.dot(W, VSigma))
    Gamma, P = np.linalg.eigh(T)
    Psqgamma = np.dot(P, np.diag(np.sqrt(Gamma)))
    B = np.dot(Psqgamma.T, np.dot(M, Psqgamma))
    Lambda, R = np.linalg.eigh(B)
    Yinv = np.dot(R.T, np.dot(np.diag(np.sqrt(Gamma)), P.T))
    Yinv_du = -np.dot(Yinv, alpha*u + g) / (alpha + mu + Lambda)
    du = (-alpha * u - g - np.dot(M, np.dot(Yinv.T, Yinv_du))) / (alpha + mu)
    u += du

# convert solution back to spectrum
As = ms * np.exp(np.dot(U, u))
for omega, A in itertools.izip(omegas, As):
    print omega, A
