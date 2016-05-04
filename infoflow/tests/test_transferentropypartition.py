import sys
import os
import numpy as np
from infoflow import transferentropypartition as tep
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ''))

in_path = 'in_transferentropypartition/'
out_path = 'out_transferentropypartition/'

x = np.genfromtxt(sys.path[0]+in_path+'X1.csv', delimiter=',').reshape((1, 202))
y = np.genfromtxt(sys.path[0]+in_path+'Y1.csv', delimiter=',').reshape((1, 202))
t = int(np.genfromtxt(sys.path[0]+in_path+'t.csv', delimiter=',').reshape((1, 1))[0][0])
w = int(np.genfromtxt(sys.path[0]+in_path+'w.csv', delimiter=',').reshape((1, 1))[0][0])
t2 = np.genfromtxt(sys.path[0]+out_path+'T2(i).csv', delimiter=',').reshape((1, 1))[0][0]
npar = np.genfromtxt(sys.path[0]+out_path+'nPar.csv', delimiter=',').reshape((1, 1))[0][0]
dimpar = np.genfromtxt(sys.path[0]+out_path+'dimPar.csv', delimiter=',').reshape((38, 1))


print(tep.transferentropypartition(x,y,t,w)[0], t2)
