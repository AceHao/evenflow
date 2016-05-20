import sys
import os
import numpy as np
from infoflow import transferentropyrank as ter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ''))

in_path = 'in_transferentropypartition/'
out_path = 'out_transferentropyrank/'

x1 = np.genfromtxt(sys.path[0]+in_path+'X1.csv', delimiter=',').reshape((1, 202))
y1 = np.genfromtxt(sys.path[0]+ in_path+'Y1.csv', delimiter=',').reshape((1, 202))

assert ter.transferEntropyRank(x1, y1, 1, 1, 2, 2, 10) == 2.020571079434638
