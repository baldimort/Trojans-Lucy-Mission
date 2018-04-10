from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import rcParams
import sys

sys.path.insert(0, '../')

import SAVEFIG

rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times']
rcParams['mathtext.fontset'] = 'cm'

'''Calculates fractional residual between our data and JPLS'''

data = np.loadtxt('convergence.dat')
params = data[:,2:]
JPL_params = np.loadtxt('JPL_params.dat')
f = open('Parameters.dat','w')

frac_errors = (params - JPL_params)/JPL_params

fig = plt.figure(figsize=(6.4*3/4,6.4*3/4))
ax = fig.add_subplot(111)

titles = [r'$\mu$',r'$\Omega$',r'$a$',r'$e$',r'$i$',r'$\omega$']

def fit(x,m,c,k):
	return np.exp(-k*x+m)+c

for i,val in enumerate(titles):
	fitparams = curve_fit(fit,data[:,0],abs(frac_errors[:,i]))[0]
	line, = ax.plot(data[:,0],abs(frac_errors[:,i]),'-o',label=titles[i],ms=3)
	#ax.plot(data[:,0],fit(data[:,0],fitparams[0],fitparams[1],fitparams[2]),color=line.get_color())
	f.write('{}	{}\n'.format(params[-1,i],frac_errors[-1,i]))
	print 'After {} Observations, {} Frac Error = {}'.format(int(data[-1,0]),val,abs(frac_errors[-1,i]))
	
#ax.plot(data[:,0],np.mean(abs(frac_errors[:,:]),axis=1),'--')
f.close()
	
ax.set_xlabel('N observations')
ax.set_ylabel('Absolute Fractional Residual')
ax.set_yscale('log')
ax.legend()
ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
plt.tight_layout()
SAVEFIG.main('save? (y/n)', 'PAT_LIT_CONVERGENCE')
plt.show()
