# maxent -- Maximum Entropy Method
# Copyright (C) 2016  Jonas Greitemann

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

#!/usr/bin/python3

import numpy as np
import kernels
import fileinput
import argparse
import itertools
import functools
import operator
import sys

def product(it):
    return functools.reduce(operator.__mul__, it)

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float)
parser.add_argument("--beta", type=float)
parser.add_argument("--cutoff", type=float, default=1e-10)
parser.add_argument("-t", "--threshold", type=float, default=1e-5)
parser.add_argument("--Nalpha", type=int, default=100)
parser.add_argument("--skip", type=int, default=0)
args, unk = parser.parse_known_args()
alphas = np.linspace(args.alpha / args.Nalpha, args.alpha, args.Nalpha)
alphas = alphas[args.skip:]

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
omegas = np.linspace(0, 5, 501)
K = kernels.imagt_kernel(taus, omegas, args.beta)
V, Sigma, U = np.linalg.svd(K)
U = U.T

# print the singular values
for s in Sigma:
    print(s)
print('\n')

# project into singular space
Sigma = np.array(list(itertools.takewhile(lambda x: x > args.cutoff, Sigma)))
s = Sigma.size
V = V[:,:s]
U = U[:,:s]
VSigma = np.dot(V, np.diag(Sigma))
print('Singular space dimension: ', s, ' (down from ',
      min(taus.size, omegas.size), ')', sep='', file=sys.stderr)

# default model
ms = 0.1 * np.ones(omegas.size)
norm = np.linalg.norm(ms, ord=1)

# loop over alpha parameters
nu = 2.
u = np.zeros(s)
u[0] = 1.
As = ms * np.exp(np.dot(U, u))
posteriors = []
Ass = []
for alpha, percent in zip(alphas, np.linspace(0, 100, alphas.size)):
    print('Calculating alpha = ', alpha, ' (', int(percent), '%) ',
          sep='', end='\r', file=sys.stderr)
    # Newton iteration
    converged = False
    mu = alpha
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
        print(mu, np.linalg.norm(Yinv_du)**2, file=sys.stderr)
        if np.linalg.norm(Yinv_du)**2 > 2 * norm:
            mu *= nu
            continue
        elif np.linalg.norm(Yinv_du)**2 < 0.01 * norm:
            mu /= nu
            continue
        du = (-alpha * u - g - np.dot(M, np.dot(Yinv.T, Yinv_du))) / (alpha+mu)
        Tu = np.dot(T, u)
        Tg = np.dot(T, g)
        u += du
        t = 2 * np.linalg.norm(alpha*Tu + Tg)**2\
                / (alpha*np.linalg.norm(Tu) + np.linalg.norm(Tg))**2
        print(t, file=sys.stderr)
        converged = t < args.threshold
        As = ms * np.exp(np.dot(U, u))
    F = np.dot(VSigma, np.dot(U.T, As))
    KsqAs = np.dot(K, np.diag(np.sqrt(As)))
    C = np.dot(KsqAs.T, np.dot(W, KsqAs))
    Lambda = np.linalg.eigvalsh(C)
    chi_sq = np.linalg.norm((Gs - F)/Gerrs)**2
    S = np.sum(As - ms) - np.nansum(As * np.log(As/ms))
    posteriors.append(product(np.sqrt(alpha / (alpha + Lambda)))
                      * np.exp(alpha * S - .5 * chi_sq) / alpha)
    Ass.append(As)
posteriors = np.array(posteriors)
posteriors /= args.alpha / args.Nalpha * sum(posteriors)
meanAs = args.alpha / args.Nalpha * sum(p * A for p, A in zip(posteriors, Ass))
for a, p in zip(alphas, posteriors):
    print(a, p)
print('\n')
for omega, A in zip(omegas, meanAs):
    print (omega, A)
