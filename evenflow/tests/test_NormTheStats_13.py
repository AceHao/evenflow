import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from normTheStats_13 import *


def vec_transpose(array):
    for ind in range(len(array)):
        array[ind] = array[ind].transpose()
    return array


def reorder(array, *argv):
    if len(argv) == 2:
        return array.reshape((argv[0], argv[1]))
    elif len(argv) == 3:
        return reorder_3d_helper(array.reshape(argv[0], argv[2], argv[1]), argv[0], argv[1], argv[2])


def reorder_3d_helper(array_3d, d1, d2, d3):
    first = array_3d[0]
    second = array_3d[1]
    third = array_3d[2]
    out_array = np.empty((d3, d1, d2))
    for ind in range(len(first)):
        out_array[ind] = np.vstack((first[ind], second[ind], third[ind]))
    return out_array
# Inputs
inpath = 'in_mat_NormTheStats_13/'
nBinMat = reorder(np.genfromtxt(inpath+'nBinMat.csv', delimiter=","), 3, 1)
I = reorder(np.genfromtxt(inpath+'I.csv', delimiter=","), 3, 3, 37)
T = reorder(np.genfromtxt(inpath+'T.csv', delimiter=","), 3, 3, 37)
SigThreshI = reorder(np.genfromtxt(inpath+'SigThreshI.csv', delimiter=","), 3, 3)
SigThreshT = reorder(np.genfromtxt(inpath+'SigThreshT.csv', delimiter=","), 3, 3)
meanShuffI = reorder(np.genfromtxt(inpath+'meanShuffI.csv', delimiter=","), 3, 3)
sigmaShuffI = reorder(np.genfromtxt(inpath+'sigmaShuffI.csv', delimiter=","), 3, 3)
meanShuffT = reorder(np.genfromtxt(inpath+'meanShuffT.csv', delimiter=","), 3, 3)
sigmaShuffT = reorder(np.genfromtxt(inpath+'sigmaShuffT.csv', delimiter=","), 3, 3)
HXt = reorder(np.genfromtxt(inpath+'HXt.csv', delimiter=","), 3, 3, 37)
HYw = reorder(np.genfromtxt(inpath+'HYw.csv', delimiter=","), 3, 3, 37)
HYf = reorder(np.genfromtxt(inpath+'HYf.csv', delimiter=","), 3, 3, 37)

# Outputs
outpath = 'out_mat_NormTheStats_13/'
InormByDist = reorder(np.genfromtxt(outpath+'InormByDist.csv', delimiter=","), 3, 3, 37)
TnormByDist = reorder(np.genfromtxt(outpath+'TnormByDist.csv', delimiter=","), 3, 3, 37)
SigThreshInormByDist = reorder(np.genfromtxt(outpath+'SigThreshInormByDist.csv', delimiter=","), 3, 3)
SigThreshTnormByDist = reorder(np.genfromtxt(outpath+'SigThreshTnormByDist.csv', delimiter=","), 3, 3)
Ic = reorder(np.genfromtxt(outpath+'Ic.csv', delimiter=","), 3, 3, 37)
Tc = reorder(np.genfromtxt(outpath+'Tc.csv', delimiter=","), 3, 3, 37)
TvsIzero = reorder(np.genfromtxt(outpath+'TvsIzero.csv', delimiter=","), 3, 3, 37)
SigThreshTvsIzero = reorder(np.genfromtxt(outpath+'SigThreshTvsIzero.csv', delimiter=","), 3, 3)
RelEnt = reorder(np.genfromtxt(outpath+'RelEnt.csv', delimiter=","), 3, 3, 37)
RelT = reorder(np.genfromtxt(outpath+'RelT.csv', delimiter=","), 3, 3, 37)
HXtNormByDist = reorder(np.genfromtxt(outpath+'HXtNormByDist.csv', delimiter=","), 3, 3, 37)
IvsIzero = reorder(np.genfromtxt(outpath+'IvsIzero.csv', delimiter=","), 3, 3, 37)
SigThreshIvsIzero = reorder(np.genfromtxt(outpath+'SigThreshIvsIzero.csv', delimiter=","), 3, 3)

outputs = [InormByDist, TnormByDist, SigThreshInormByDist, SigThreshTnormByDist, Ic, Tc, TvsIzero, SigThreshTvsIzero,
           RelEnt, RelT, HXtNormByDist, IvsIzero, SigThreshIvsIzero]

result = normthestats(nBinMat, I, T, SigThreshI, SigThreshT, meanShuffI, sigmaShuffI, meanShuffT, sigmaShuffT, HXt, HYw,
                      HYf)

for i in range(len(result)):
    np.testing.assert_array_almost_equal(result[i], outputs[i])
