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

def bcs(omegas):
    gap = 0.91
    W = 8.4
    N = 1./np.sqrt(W**2/2 - gap**2)
    return np.array([N * o / np.sqrt(o**2 - gap**2) if gap < o <= W/2 else 0.
                     for o in omegas])

parser = argparse.ArgumentParser()
parser.add_argument('--eta', type=float, default=1e-4)
parser.add_argument('--beta', type=float, default=1.)
parser.add_argument('--spectrum', action='store_true')
parser.add_argument('--exact', action='store_true')
parser.add_argument('--bcs', action='store_true')
args = parser.parse_args()

model = bcs if args.bcs else two_peaks

taus = np.linspace(0, args.beta/2, 1001)
omegas = np.linspace(0, 10, 1001)

if args.spectrum:
    for omega, A in izip(omegas, model(omegas)):
        print omega, A
    exit(0)

green = np.dot(kernels.imagt_kernel(taus, omegas, args.beta), model(omegas))
if not args.exact:
    green *= 1 + args.eta * np.random.standard_normal(green.shape)

for tau, G in izip(taus, green):
    print tau, G, args.eta * G if not args.exact else 0
