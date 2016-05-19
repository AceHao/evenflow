import sys
import os
import numpy as np
from infoflow import quantentr as qtn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ''))

in_path = 'in_quantentr/'
out_path = 'out_quantentr/'

x = np.genfromtxt(sys.path[0]+in_path+'X.csv', delimiter=',').reshape((1, 202))
q = np.genfromtxt(sys.path[0]+ in_path+'Q.csv', delimiter=',')
xq = np.genfromtxt(sys.path[0]+out_path+'Xq.csv', delimiter=',').reshape((1, 202))

assert np.all((xq == qtn.quantentr(x, q)))
