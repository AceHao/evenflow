import numpy as np


def doproduction(martrix_t):
    # Extract the dimensions(has to be 3) from matrix T.
    ntaus, nvars, p = np.shape(martrix_t)

    # Creates two 2D arrays filled with 0s.
    tplus = np.zeros((nvars, ntaus))
    tminus = np.zeros((nvars, ntaus))

    # Creates one 3D array filled with NaN.
    tnet = np.empty((nvars, ntaus))
    tnet.fill(np.NAN)
    tnetbinary = np.empty((ntaus, nvars, nvars))
    tnetbinary.fill(np.NAN)

    # Calculation
    for i in range(nvars):
        for j in range(nvars):
            for t in range(ntaus):
                if martrix_t[t, i, j] != np.NAN:
                    tplus[i, t] = tplus[i, t] + martrix_t[t, i, j]
                    tminus[j, t] = tminus[j, t] + martrix_t[t, i, j]

    tnet = tplus - tminus

    for t in range(ntaus):
        sqrmat = martrix_t[t, :, :]

        tnetbinary[t, :, :] = sqrmat - sqrmat.transpose()
        # transpose of non-complex conjugate, but original matlab code uses complex conjugate transpose

    return tplus, tminus, tnet, tnetbinary
