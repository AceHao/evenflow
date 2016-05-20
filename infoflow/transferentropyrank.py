import numpy as np
from infoflow import quantentr as qtn
from math import log2

def transferEntropyRank(X,Y,l,k,t,w,Q):

    """
    % This function computes the transfer entropy between time series X and Y,
    % with the flow of information directed from X to Y, after ranking both X and Y. Probability density
    % estimation is based on bin counting with fixed and equally-spaced bins.
    %
    % For details, please see T Schreiber, "Measuring information transfer", Physical Review Letters, 85(2):461-464, 2000.
    %
    % Inputs:
    % X: first time series in 1-D vector
    % Y: second time series in 1-D vector
    % l: block length for X
    % k: block length for Y
    % t: time lag in X from present to where the block of length l ends
    % w: time lag in Y from present to where the block of length k ends
    % Q: number of quantization levels for both X and Y
    %
    % Outputs:
    % T: transfer entropy (bits)
    %
    %
    % Copyright 2011 Joon Lee
    %
    % This program is free software: you can redistribute it and/or modify
    % it under the terms of the GNU General Public License as published by
    % the Free Software Foundation, either version 3 of the License, or
    % (at your option) any later version.
    %
    % This program is distributed in the hope that it will be useful,
    % but WITHOUT ANY WARRANTY; without even the implied warranty of
    % MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    % GNU General Public License for more details.
    %
    % You should have received a copy of the GNU General Public License
    % along with this program.  If not, see <http://www.gnu.org/licenses/>.
    """

    xflat = X.flatten(order='f')
    yflat = Y.flatten(order='f')

    # ordinal sampling (ranking)
    Nt = np.size(xflat)
    B, IX = np.sort(X), np.argsort(xflat)
    xflat[IX] = np.arange(Nt)
    B, IY = np.sort(Y), np.argsort(yflat)
    yflat[IY] = np.arange(Nt)

    # quantize X and Y according to fixed, equally-spaced bins
    Xq = qtn.quantentr(xflat, Q)
    Yq = qtn.quantentr(yflat, Q)

    # go through the time series X and Y, and populate Xpat, Ypat, and Yt
    Xpat, Ypat, Yt = [], [], []
    codeX = (Q ** (np.arange((l-1), -1, -1)))
    codeY = (Q ** (np.arange((k-1), -1, -1)))

    for i in range(max([l+t, k+w])-1, min([len(Xq), len(Yq)]), 1):
        Xpat.append(np.dot(Xq[i-l-t+1:i-t+1], codeX))
        Ypat.append(np.dot(Yq[i-k-w+1:i-w+1], codeY))
        Yt.append(Yq[i])

    Xpat, Ypat, Yt = np.array(Xpat), np.array(Ypat), np.array(Yt)
    # compute transfer entropy
    T = 0
    idxDone = np.array([])
    N = len(Xpat)

    for i in range(N):
        if not np.any(i == idxDone):
            p1 = sum(np.logical_and(np.logical_and(Xpat==Xpat[i], Ypat==Ypat[i]), Yt==Yt[i])) / N
            p2 = sum(np.logical_and(np.logical_and(Xpat==Xpat[i], Ypat==Ypat[i]), Yt==Yt[i])) / \
                 sum(np.logical_and(Xpat==Xpat[i], Ypat==Ypat[i]))
            p3 = sum(np.logical_and(Ypat==Ypat[i], Yt==Yt[i])) / sum(Ypat==Ypat[i])
            T += p1 * log2(p2/p3)
            idxDone = np.append(idxDone, np.nonzero(np.logical_and(np.logical_and(Xpat==Xpat[i], Ypat==Ypat[i]), Yt==Yt[i])))
            idxDone = np.unique(idxDone)

    return T