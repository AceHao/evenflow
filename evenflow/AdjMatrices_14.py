import numpy as np

# Computes an Adjacency Matrix A
# Computes the Characteristic Lag Which is the First Significant Lag
# Takes the Transfer Information Matrix and the Significance Thresholds


def adjmatrices(t, sigthresht, tvsizero):
    def helper1(sx, sy, lag):
        abinary[lag, sx,sy] = 1
        awtdcut[lag, sx,sy] = t[lag, sx,sy]
        lastsiglag[sx,sy] = lag+1
        nsiglags[sx, sy] = nsiglags[sx, sy] + 1
        return

    def helper2(sx, sy, lag):
        charlagmaxpeak[sx, sy] = lag+1
        tcharlagmaxpeak[sx, sy] = t[lag, sx,sy]
        return

    def helper3(sx, sy, lag):
        charlagfirstpeak[sx, sy] = lag+1
        tcharlagfirstpeak[sx, sy] = t[lag, sx,sy]
        return

    nlags, nsignals, junk = np.shape(t)

    abinary = np.zeros((nlags, nsignals,nsignals))
    awtd = np.empty((nlags, nsignals, nsignals))
    awtd.fill(np.NAN)

    awtdcut = np.zeros((nlags, nsignals, nsignals))
    charlagfirstpeak = np.zeros((nsignals, nsignals))
    tcharlagfirstpeak = np.zeros((nsignals,nsignals))
    charlagmaxpeak = np.zeros((nsignals,nsignals))
    tcharlagmaxpeak = np.zeros((nsignals, nsignals))
    tvsizerocharlagmaxpeak = np.zeros((nsignals, nsignals))
    nsiglags = np.zeros((nsignals, nsignals))
    firstsiglag = np.empty((nsignals, nsignals))
    firstsiglag.fill(np.NAN)

    lastsiglag = np.empty((nsignals, nsignals))
    lastsiglag.fill(np.NAN)

    for sX in range(nsignals):
        for sY in range(nsignals):
            firstpeakflag = 0
            firstsigflag = 0

            awtd = t

            # check the first Lag
            lag = 0
            if t[lag, sX, sY] > sigthresht[sX,sY]:
                helper1(sX, sY, lag)
                helper2(sX, sY, lag)
                
                tvsizerocharlagmaxpeak[sX, sY] = tvsizero[lag, sX, sY]
                firstsigflag = 1

                if nlags > 1:
                    if t[lag, sX, sY] > t[lag+1, sX, sY]:
                        helper3(sX, sY, lag)
                        firstpeakflag = 1
                else:
                    helper3(sX, sY, lag)
                    firstpeakflag = 1

            # nlags: number of lags
            # re-check indexing becaus matlab uses indexing start with 1, python uses 0.
            # create function for 2 lag checkings (note the difference) for possible improvement
            # check the other lag
            if nlags > 1:
                for lag in range(1, nlags - 1):
                    if t[lag, sX, sY] > sigthresht[sX, sY]:
                        helper1(sX, sY, lag)
                        if firstsigflag == 0:
                                firstsiglag[sX, sY] = lag+1
                                firstsigflag = 1
                        if firstpeakflag == 0 and t[lag, sX, sY] > t[lag - 1, sX, sY]and t[lag, sX, sY] > t[ lag + 1, sX, sY]:
                            helper3(sX, sY, lag)
                            firstpeakflag = 1
                        if t[lag, sX, sY] > tcharlagmaxpeak[sX, sY]:
                            helper2(sX, sY, lag)
                            tvsizerocharlagmaxpeak[sX, sY] = tvsizero[lag, sX, sY ]
                # check the last lag
                lag = nlags - 1
                if t[lag, sX, sY ] > sigthresht[sX, sY]:
                            helper1(sX, sY, lag)
                            if firstsigflag == 0:
                                firstsiglag[sX, sY] = lag+1
                                firstsigflag = 1
                            if firstpeakflag == 0 and t[lag, sX, sY ] > t[lag - 1, sX, sY ]:
                                # charLagFirst[sX,sY] = lag;
                                # TcharLagFirst[sX,sY] = t[lag,sX,sY];
                                firstpeakflag = 1
                            if t[lag, sX, sY] > tcharlagmaxpeak[sX, sY]:
                                helper2(sX, sY, lag)
                                tvsizerocharlagmaxpeak[sX, sY] = tvsizero[lag, sX, sY]

    return abinary, awtd, awtdcut, charlagfirstpeak, tcharlagfirstpeak, charlagmaxpeak, tcharlagmaxpeak,\
           tvsizerocharlagmaxpeak, nsiglags, firstsiglag, lastsiglag
