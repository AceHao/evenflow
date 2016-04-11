import numpy as np

def replace(A, S1, S2):

    #the engine
    copy_A = np.copy(A)
    tf = np.empty(np.shape(A))
    tf.fill(False)


    if np.shape(S2) == ():
        S = np.empty(np.shape(A))
        S.fill(S2)
        S2 = S
    if np.isnan(A).any() and np.isnan(S1).any():
        copy_A[np.isnan(A)] = S2[np.isnan(S1)]
        tf[np.isnan(A)] = True
    for s1, s2 in zip(S1, S2):
        copy_A[np.where(A == s1)] = s2
        tf[np.where(A == s1)] = True

    return copy_A , tf
