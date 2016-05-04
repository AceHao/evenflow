import numpy as np
import sys
from infoflow import dvpartition3d as dvp3
import math


def transferentropypartition(X, Y, t, w):

    '''
    % This function computes the transfer entropy between time series X and Y,
    % with the flow of information directed from X to Y. Probability density
    % estimation is based on the Darbellay-Vajda partitioning algorithm.
    %
    % For details, please see T Schreiber, "Measuring information transfer", Physical Review Letters, 85(2):461-464, 2000.
    %
    % Inputs:
    % X: source time series in 1-D vector
    % Y: target time series in 1-D vector
    % t: time lag in X from present
    % w: time lag in Y from present
    %
    % Outputs:
    % T: transfer entropy (bits)
    % nPar: number of partitions
    % dimPar: 1-D vector containing the length of each partition (same along all three dimensions)
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
    '''

    def rank(np_arr):
        flat = np_arr.flatten()
        flat[flat.argsort()] = range(len(flat))
        return flat.reshape(len(flat), 1)


    # fix block lengths at 1
    l, k=1, 1

    X=np.transpose(X)
    Y=np.transpose(Y)

    # go through the time series X and Y, and populate Xpat, Ypat, and Yt
    Xpat, Ypat, Yt= [], [], []
    for i in range(max([l+t, k+w])-1, min([len(X), len(Y)]), 1):
        Xpat.append(X[i-l-t+1:i-t+1])
        Ypat.append(Y[i-k-w+1:i-w+1])
        #print(Y[i-k-w:i-w])

        Yt.append(Y[i])
    Xpat, Ypat, Yt = np.array(Xpat), np.array(Ypat), np.array(Yt)

    # ordinal sampling (ranking)
    Nt = len(Xpat)
    #print(Xpat[0])
    Xpat = rank(Xpat)
    #B,IX = np.sort(Xpat), np.argsort(Xpat)
    #Xpat[IX] = list(range(Nt))
    #print(Ypat[0])
    Ypat = rank(Ypat)
    #B,IX = np.sort(Ypat), np.argsort(Ypat)
    #Ypat[IX] = list(range(Nt))
    Yt = rank(Yt)
    #B, IX = np.sort(Yt), np.argsort(Yt)
    #Yt[IX] = list(range(Nt))

    # compute transfer entropy
    # dlmwrite('Xpat.csv',Xpat,'Precision',16);
    # dlmwrite('Ypat.csv',Ypat,'Precision',16);
    # dlmwrite('Yt.csv',Yt,'Precision',16);

    partitions = dvp3.dvpartition3d(Xpat,Ypat,Yt,1,Nt,1,Nt,1,Nt)
    # %dlmwrite('partitions.csv',partitions,'Precision',16);
    for element in partitions:
        print(element)
    nPar = len(partitions)
    dimPar = np.zeros((nPar,1))
    for i in range(nPar):
        dimPar[i] = partitions[i]["Xmax"] - partitions[i]["Xmin"] + 1
        T = 0
    for i in range(len(partitions)):
        a = partitions[i]["N"] / Nt
        #print(a)
        b = sum(np.logical_and(
                np.logical_and(
                np.logical_and(
                Xpat >= partitions[i]["Xmin"], Xpat <= partitions[i]["Xmax"]),
                Ypat >= partitions[i]["Ymin"]),
                Ypat <= partitions[i]["Ymax"])) / Nt

        c = sum(np.logical_and(
                np.logical_and(
                np.logical_and(
                Yt >= partitions[i]["Zmin"], Yt <= partitions[i]["Zmax"]),
                Ypat >= partitions[i]["Ymin"]),
                Ypat <= partitions[i]["Ymax"])) / Nt

        d = (partitions[i]["Ymax"] - partitions[i]["Ymin"] + 1) / Nt
        T += a * math.log2((a*d) / (b*c))

    return T, nPar, dimPar
