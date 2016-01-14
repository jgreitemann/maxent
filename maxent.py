#!/usr/bin/python2

import numpy as np
import kernels
import fileinput
import argparse
import itertools
import operator

def product(it):
    return reduce(operator.__mul__, it)

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float)
parser.add_argument("--beta", type=float)
parser.add_argument("--cutoff", type=float, default=1e-10)
parser.add_argument("-t", "--threshold", type=float, default=1e-5)
parser.add_argument("--Nalpha", type=int, default=100)
args, unk = parser.parse_known_args()
alphas = np.linspace(args.alpha / args.Nalpha, args.alpha, args.Nalpha)

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

# loop over alpha parameters
mu = 0.
u = np.zeros(s)
u[0] = 1.
As = ms * np.exp(np.dot(U, u))
posteriors = []
Ass = []
for alpha in alphas:
    # Newton iteration
    converged = False
    while not converged:
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
        Tu = np.dot(T, u)
        Tg = np.dot(T, g)
        u += du
        t = 2 * np.linalg.norm(alpha*Tu + Tg)**2\
                / (alpha*np.linalg.norm(Tu) + np.linalg.norm(Tg))**2
        converged = t < args.threshold
        As = ms * np.exp(np.dot(U, u))
    F = np.dot(VSigma, np.dot(U.T, As))
    KsqAs = np.dot(K, np.diag(np.sqrt(As)))
    C = np.dot(KsqAs.T, np.dot(W, KsqAs))
    Lambda = np.linalg.eigvalsh(C)
    chi_sq = np.linalg.norm((Gs - F)/Gerrs)**2
    S = sum(As - ms - As * np.log(As/ms))
    posteriors.append(product(np.sqrt(alpha / (alpha + Lambda)))
                      * np.exp(alpha * S - .5 * chi_sq) / alpha)
    Ass.append(As)
posteriors = np.array(posteriors)
posteriors /= args.alpha / args.Nalpha * sum(posteriors)
meanAs = sum(p * A for p, A in itertools.izip(posteriors, Ass))
for a, p in itertools.izip(alphas, posteriors):
    print a, p
print ''
print ''
for omega, A in itertools.izip(omegas, meanAs):
    print omega, A
