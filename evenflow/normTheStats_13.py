import numpy as np
from math import *


def normthestats(nbinmat, matrix_i, matrix_t, sigthreshi, sigthresht, meanshuffi, sigmashuffi, meanshufft, sigmashufft, hxt, HYw, hyf):
    nlags, nsignals, junk = np.shape(matrix_i)

    inormbydist = np.empty((nlags, nsignals, nsignals))
    inormbydist.fill(np.nan)

    tnormbydist = np.empty((nlags, nsignals, nsignals))
    tnormbydist.fill(np.nan)

    sigthreshinormbydist = np.empty((nsignals, nsignals))
    sigthreshinormbydist.fill(np.nan)

    sigthreshtnormbydist = np.empty((nsignals, nsignals))
    sigthreshtnormbydist.fill(np.nan)

    ic = np.empty((nlags, nsignals, nsignals))
    ic.fill(np.nan)

    tc = np.empty((nlags, nsignals, nsignals))
    tc.fill(np.nan)

    tvsizero = np.empty((nlags, nsignals, nsignals))
    tvsizero.fill(np.nan)

    sigthreshtvsizero = np.empty((nsignals, nsignals))
    sigthreshtvsizero.fill(np.nan)

    relent = np.empty((nlags, nsignals, nsignals))
    relent.fill(np.nan)

    relt = np.empty((nlags, nsignals, nsignals))
    relt.fill(np.nan)

    hxtnormbydist = np.empty((nlags, nsignals, nsignals))
    hxtnormbydist.fill(np.nan)

    ivsizero = np.empty((nlags, nsignals, nsignals))
    ivsizero.fill(np.nan)

    sigthreshivsizero = np.empty((nsignals, nsignals))
    sigthreshivsizero.fill(np.nan)

    for i in range(nsignals):
        for j in range(nsignals):
            for t in range(nlags):
                n = min(nbinmat[i], nbinmat[j])
                inormbydist[t, i, j] = matrix_i[t, i, j]/log2(n)
                tnormbydist[t, i, j] = matrix_t[t, i, j]/log2(n)
                ic[t, i, j] = 0.5*(1+erf((matrix_i[t, i, j]-meanshuffi[i, j])/(sqrt(2)*sigmashuffi[i, j])))
                tc[t, i, j] = 0.5*(1+erf((matrix_t[t, i, j]-meanshufft[i, j])/(sqrt(2)*sigmashufft[i, j])))
                sigthreshinormbydist[i, j] = sigthreshi[i, j]/log2(n)
                sigthreshtnormbydist[i, j] = sigthresht[i, j]/log2(n)
                tvsizero[t, i, j] = matrix_t[t, i, j]/matrix_i[0, i, j]
                sigthreshtvsizero[i, j] = sigthresht[i, j]/matrix_i[0, i, j]
                relent[t, i, j] = matrix_i[t, i, j]/hyf[t, i, j]
                relt[t, i, j] = matrix_t[t, i, j]/hyf[t, i, j]

                hxtnormbydist[t, i, j] = hxt[t, i, j]/log2(n)
                ivsizero[t, i, j] = matrix_i[t, i, j]/matrix_i[0, i, j]
                sigthreshivsizero[i, j] = sigthreshi[i, j]/matrix_i[0, i, j]

    return inormbydist, tnormbydist, sigthreshinormbydist, sigthreshtnormbydist, ic, tc, tvsizero, sigthreshtvsizero,\
        relent, relt, hxtnormbydist, ivsizero, sigthreshivsizero
