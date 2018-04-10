from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times']
rcParams['mathtext.fontset'] = 'cm'

sys.path.insert(0, '../')

import SAVEFIG

'''Calculates fractional residual between FIND_ORBS and JPLS'''

M,a,e,peri,node,incl = [],[],[],[],[],[]


try: f = open(str(sys.argv[1]),'r')
except Exception:
	print '\n\n************ find orb params data not supplied/incorrect! ****************\n\n'
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
{'param':r'$\mu$','data': np.array(M,dtype='float64')},
{'param':r'$\omega$','data': np.array(peri,dtype='float64')},
{'param':r'$\Omega$','data': np.array(node,dtype='float64')},
{'param':r'$i$','data': np.array(incl,dtype='float64')},
{'param':r'$e$','data': np.array(e,dtype='float64')},
{'param':r'$a$','data': np.array(a,dtype='float64')},
]
params = np.array([M,peri,node,incl,e,a],dtype='float64').T

JPL_params = np.loadtxt('JPL_params.dat',delimiter=',')
f = open('Parameters.dat','w')

frac_errors = abs(params - JPL_params)/JPL_params
#print frac_errors
fig = plt.figure(figsize=(6.4*3/4,6.4*3/4))
ax = fig.add_subplot(111)

#titles = ['Mean anomaly','Ascending node','Semi-major axis','Eccentricity','Inclination','Long. of Perihelion']

def fit(x,m,c,k):
	return np.exp(-k*x+m)+c

for i,val in enumerate(parameters):
	
	#line, = ax.plot(data[:,0],abs(frac_errors[:,i]),'o',label=titles[i],ms=3)
	ax.plot(np.arange(frac_errors.shape[0])+3,frac_errors[:,i],'o-',ms=2,label=val['param'])
	
#ax.plot(data[:,0],np.mean(abs(frac_errors[:,:]),axis=1),'--')

ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
ax.set_xlabel('N observations',fontsize=12)
ax.set_ylabel('Absolute Fractional Residual',fontsize=12)
ax.set_yscale('log')
ax.legend()
plt.tight_layout()
SAVEFIG.main('save? (y/n)','JPL_FINDORB_CONVERGENCE')
plt.show()
