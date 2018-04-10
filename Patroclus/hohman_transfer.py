from __future__ import division
import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import sys
from matplotlib import rcParams
import time

rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times']
rcParams['mathtext.fontset'] = 'cm'

sys.path.insert(0, '../')
import SAVEFIG

G = 6.67e-11
M = 1.989e30
AU = 1.496e11


def hohmann_dv(r1,r2):
	mu = G*M
	dv1 = (mu/r1)**0.5 * ((2*r2 / (r1+r2))**0.5 -1)		# first orbital insertion
	dv2 = (mu/r2)**0.5 * (1 - (2*r1 / (r1+r2))**0.5)	# second orbital insertion
	dvtot = dv1 + dv2
	return (dv1,dv2,dvtot)

r_mins, r_maxs = [], []
dvs = []

for i in (str(sys.argv[1]),str(sys.argv[2])): #two surfaces that need to be loaded (inc/exc archival)
	xs,ys,zs = np.load(i) 

	rs = (xs**2 + ys**2 + zs**2)**0.5
	r_min, r_max = rs.min(), rs.max()
	r_mins.append(r_mins)
	r_maxs.append(r_max)
	dvs.append(hohmann_dv(AU,r_min*AU))
	dvs.append(hohmann_dv(AU,r_max*AU))
	print "Min. Delta v total: {}\nMax. Delta v total: {}\n".format(hohmann_dv(AU,r_min*AU),hohmann_dv(AU,r_max*AU))










fig1 = plt.figure(figsize=(3/4*6.4,3/4*6.4))
ax1 = fig1.add_subplot(111)

N = 100
thetas = np.linspace(0,np.pi*2,N)
ax1.plot(np.cos(thetas),np.sin(thetas),'g-')
ax1.plot(0,0,'yo')
ax1.plot(3*np.cos(thetas),3*np.sin(thetas),'b-')

newthetas = np.linspace(0,np.pi,100)
e = 3*2/(1 + 3) - 1
r = 2*(1-e**2)/(1+e*np.cos(newthetas))
x_ell,y_ell = r*np.cos(newthetas-np.pi/2), r*np.sin(newthetas-np.pi/2)

ax1.plot(x_ell,y_ell,'r-')
ax1.set_aspect('equal')
ax1.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
ax1.set_xlabel('x (AU)',fontsize=12)
ax1.set_ylabel('y (AU)',fontsize=12)
ax1.plot((x_ell[0],x_ell[-1]),(y_ell[0],y_ell[-1]),'ro')
plt.text(x_ell[0]-0.3,y_ell[0]-0.4,r'$\Delta v_1$',fontsize=12)
plt.text(x_ell[-1]-0.3,y_ell[-1]-0.4,r'$\Delta v_2$',fontsize=12)
SAVEFIG.main('save?','HOHMANN_TRANSFER_EXAMPLE') 






plt.show()
