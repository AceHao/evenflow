import numpy as np
from math import *

def normthestats(nBinMat,I,T,SigThreshI,SigThreshT,meanShuffI,sigmaShuffI,meanShuffT,sigmaShuffT,HXt,HYw,HYf):
    nLags, nSignals,junk=np.shape(I);

    InormByDist=np.empty((nLags, nSignals, nSignals))
    InormByDist.fill(np.nan)

    TnormByDist=np.empty((nLags, nSignals, nSignals))
    TnormByDist.fill(np.nan)

    SigThreshInormByDist=np.empty((nSignals,nSignals))
    SigThreshInormByDist.fill(np.nan)

    SigThreshTnormByDist=np.empty((nSignals,nSignals))
    SigThreshTnormByDist.fill(np.nan)

    Ic=np.empty((nLags, nSignals, nSignals))
    Ic.fill(np.nan)

    Tc=np.empty((nLags, nSignals, nSignals))
    Tc.fill(np.nan)

    TvsIzero=np.empty((nLags, nSignals, nSignals))
    TvsIzero.fill(np.nan)

    SigThreshTvsIzero=np.empty((nSignals,nSignals))
    SigThreshTvsIzero.fill(np.nan)

    RelEnt=np.empty((nLags, nSignals, nSignals))
    RelEnt.fill(np.nan)

    RelT=np.empty((nLags, nSignals, nSignals))
    RelT.fill(np.nan)

    HXtNormByDist=np.empty((nLags, nSignals, nSignals))
    HXtNormByDist.fill(np.nan)

    IvsIzero=np.empty((nLags, nSignals, nSignals))
    IvsIzero.fill(np.nan)

    SigThreshIvsIzero=np.empty((nSignals,nSignals))
    SigThreshIvsIzero.fill(np.nan)

    for i in range(nSignals):
        for j in range(nSignals):
            for t in range(nLags):
                n = min(nBinMat[i],nBinMat[j])
                InormByDist[t,i,j]=I[t,i,j]/log2(n)
                TnormByDist[t,i,j]=T[t,i,j]/log2(n)
                Ic[t,i,j]=0.5*(1+erf((I[t,i,j]-meanShuffI[i,j])/(sqrt(2)*sigmaShuffI[i,j])))
                Tc[t,i,j]=0.5*(1+erf((T[t,i,j]-meanShuffT[i,j])/(sqrt(2)*sigmaShuffT[i,j])))
                SigThreshInormByDist[i,j]=SigThreshI[i,j]/log2(n)
                SigThreshTnormByDist[i,j]=SigThreshT[i,j]/log2(n)
                TvsIzero[t,i,j]=T[t,i,j]/I[0,i,j]
                SigThreshTvsIzero[i,j]=SigThreshT[i,j]/I[0,i,j]
                RelEnt[t,i,j]=I[t,i,j]/HYf[t,i,j]
                RelT[t,i,j]=T[t,i,j]/HYf[t,i,j]

                HXtNormByDist[t,i,j]=HXt[t,i,j]/log2(n)
                IvsIzero[t,i,j]=I[t,i,j]/I[0,i,j]
                SigThreshIvsIzero[i,j]=SigThreshI[i,j]/I[0,i,j]

    return (InormByDist,TnormByDist,SigThreshInormByDist,SigThreshTnormByDist,Ic,Tc,TvsIzero,SigThreshTvsIzero,RelEnt,RelT,HXtNormByDist,IvsIzero,SigThreshIvsIzero)
