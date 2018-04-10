from __future__ import division
import numpy as np
import sys,time
from scipy.stats import norm
import matplotlib.pyplot as plt
import subprocess
from mpl_toolkits.mplot3d import Axes3D
import find_xyz_func
import xyz
import pandas as pd

#N_positions = 10000
#parameters = np.genfromtxt(sys.argv[1],usecols=(0,2))
date = str(sys.argv[1])
epoch = '2011 06 08'

def gen_Param_File(parameters):
	
	f = open('orbital_elements_working','w')
	f.write(\
'Object = Patroclus\n\
Epoch of osulation = {}\n\
Mean anomaly = {}\n\
Argument of perihelion = {}\n\
Long. of ascending node = {}\n\
Inclination = {}\n\
Eccentricity = {}\n\
Semimajor axis = {}'.format(str(epoch),str(parameters['MA']),str(parameters['AoP']),str(parameters['LoAN']),str(parameters['i']),str(parameters['e']),str(parameters['a'])))
	f.close()

'''rand_params = np.zeros([N_positions,6],dtype='float64')

for e,(param,error) in enumerate(parameters):
	rand_params[:,e] = np.random.normal(param,error,N_positions).astype('float64')'''


rand_params = pd.read_csv('random_params.csv',index_col=False)
matts_params = pd.read_csv('sampling.txt',index_col=False)
x,y,z = np.zeros(rand_params.shape[0],dtype='float64'),np.zeros(rand_params.shape[0],dtype='float64'),np.zeros(rand_params.shape[0],dtype='float64')
#print rand_params.shape
x2,y2,z2 = np.zeros(rand_params.shape[0],dtype='float64'),np.zeros(rand_params.shape[0],dtype='float64'),np.zeros(rand_params.shape[0],dtype='float64')
#print matts_params.shape,rand_params.shape
for e in range(matts_params.shape[0]):
	if e%1000 == 0: print e
	#gen_Param_File(rand_params.iloc[e])
	
	gen_Param_File(matts_params.iloc[e])
	out = find_xyz_func.main(date,'orbital_elements_working')
	gen_Param_File(matts_params.iloc[e])
	out2 = xyz.main(date)
	if out[0] != out2[0] or out[1] != out2[1] or out[2] != out2[2]: raise Exception
	#if e%1000 == 0: print out[0],out2[0],out[1],out2[1],out[2],out2[2]
	x[e] = out[0]
	y[e] = out[1]
	z[e] = out[2]
	x2[e] = out2[0]
	y2[e] = out2[1]
	z2[e] = out2[2]
	


all_pos = np.array([x,y,z]).T
#print all_pos

out_csv = pd.DataFrame(data=all_pos,columns=['x','y','z'])
out_csv.to_csv('data_xyz.csv')

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(x,y,z,s=1)
ax.plot(x2,y2,z2,'o',ms=1)


max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(x.max()+x.min())
Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(y.max()+y.min())
Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(z.max()+z.min())
# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
   ax.plot([xb], [yb], [zb], 'w')

'''
fig =plt.figure()

axs = []
for i in range(6):
	axs.append(fig.add_subplot(611+i))
	axs[-1].hist(rand_params[:,i],30)

plt.tight_layout()
plt.subplots_adjust(hspace=0)'''
plt.show()	

