import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from replace_15 import *


#Test 1
A = np.array([1,1,2,3,4,4])
S1 = np.array([1,3])
S2 = np.array([0,99])
result , tf= replace(A, S1, S2)
assert (result  == np.array([0,0,2,99,4,4])).all()
assert all(tf == [True, True, False, True, False, False])

#Test2
A = np.array(list(range(1,11)))
S1 = np.array([3,5,6,8])
S2 = np.array(np.nan)
result, tf = replace(A, S1, S2)
masked_result = np.ma.masked_where(np.isnan(result), result)
target = np.array([1,2,np.nan,4,np.nan,np.nan,7,np.nan,9,10])
masked_target = np.ma.masked_where(np.isnan(target), target)

assert (masked_result == masked_target).all()
assert all(tf == [False, False, True, False, True, True, False, True, False, False])

#Test 3
A = np.array([1, np.NaN, np.Inf, 8, 99])
S1 = np.array([np.NaN, np.Inf, 99])
S2 = np.array([12, 13, 14])
result , tf = replace(A, S1, S2)
assert (result == np.array([1,12,13,8,14])).all()
assert all( tf == [False, True, True, False, True])
