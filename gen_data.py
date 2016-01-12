#!/usr/bin/python2

import numpy as np
import argparse
import kernels

from itertools import izip

def two_peaks(omegas):
    s1 = 0.5
    m1 = 2.
    s2 = 1.5
    m2 = 5.
    return 0.5/np.sqrt(2*np.pi)/s1 * np.exp(-0.5*(omegas-m1)**2/s1**2) \
         + 0.5/np.sqrt(2*np.pi)/s2 * np.exp(-0.5*(omegas-m2)**2/s2**2)

parser = argparse.ArgumentParser()
parser.add_argument('--eta', type=float, default=1e-4)
parser.add_argument('--beta', type=float, default=1.)
parser.add_argument('--spectrum', action='store_true')
parser.add_argument('--exact', action='store_true')
args = parser.parse_args()

taus = np.linspace(0, args.beta/2, 1001)
omegas = np.linspace(0, 10, 1001)

if args.spectrum:
    for omega, A in izip(omegas, two_peaks(omegas)):
        print omega, A
    exit(0)

green = np.dot(kernels.imagt_kernel(taus, omegas, args.beta), two_peaks(omegas))
if not args.exact:
    green *= 1 + args.eta * np.random.standard_normal(green.shape)

for tau, G in izip(taus, green):
    print tau, G
