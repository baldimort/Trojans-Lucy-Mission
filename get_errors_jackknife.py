from __future__ import division
import numpy as np
import sys,time
from scipy.stats import norm


'''Takes jacknifed data and calculates the mean and std on parameters'''

M,a,e,peri,node,incl = [],[],[],[],[],[]

l = open('{}_Parameters.txt'.format(time.strftime('%Y-%m-%d_%H%M%S')),'w')
try: f = open(str(sys.argv[1]),'r')
except Exception:
	print '\n\n************ Jack knife data not supplied/incorrect! ****************\n\n'
	raise Exception
	
while True:
	line = f.readline()
	
	if line == '': break
	line = line.split()
	
	for i,item in enumerate(line):
		if item == 'M': M.append(line[i+1])
		elif item == 'a': a.append(line[i+1])
		elif item == 'e': e.append(line[i+1])
		elif item == 'Peri.': peri.append(line[i+1])
		elif item == 'Node': node.append(line[i+1])
		elif item == 'Incl.': incl.append(line[i+1])

parameters = [
{'param':'Mean Anomaly','data': np.array(M,dtype='float64')},
{'param':'Long. of perihelion','data': np.array(peri,dtype='float64')},
{'param':'Long. of ascending node','data': np.array(node,dtype='float64')},
{'param':'Inclination','data': np.array(incl,dtype='float64')},
{'param':'Eccentricity','data': np.array(e,dtype='float64')},
{'param':'Semi-major axis','data': np.array(a,dtype='float64')},
]


for param in parameters:

	mean,std = norm.fit(np.array(param['data']))
	l.write('{:.16f} +/- {:.16f} - {:s}\n'.format(mean,std,param['param']))
	print '{:s} = {:.16f} +/- {:.16f}'.format(param['param'],mean,std)

