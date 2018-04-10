from __future__ import division
import numpy as np
import pandas as pd
import sys
import find_xyz_func
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import sys
from matplotlib import rcParams
import time

rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times']
rcParams['mathtext.fontset'] = 'cm'

sys.path.insert(0, '../../')
import SAVEFIG


date = str(sys.argv[1]) #give date of the positon to be calculated
N_sets = int(sys.argv[2]) #jacknife - how many times?
parameters = str(sys.argv[3]) #give the parameters txt file
epoch_osc='2018/02/15'
epoch='2455720.5'

'''def write_file(ps):
	#print ps
	f = open('orbital_elements_working','w')
	f.write("\
Object: 617 Patroclus\n\
Epoch of osculation     =  {}\n\
Mean anomaly            =  {}\n\
Argument of perihelion  =  {}\n\
Long. of ascending node =  {}\n\
Inclination             =  {}\n\
Eccentricity            =  {}\n\
Semimajor axis          =  {}".format("2011 06 08",ps[0],ps[1],ps[2],ps[3],ps[4],ps[5]))
	f.close()'''
	
def find_pos(ps,date):
	#print ps
	#write_file(ps)
	x,y,z = find_xyz_func.main(date,ps,epoch_osc,epoch)
	return np.array((x,y,z),dtype=np.float64)
	
params = np.genfromtxt(parameters,dtype=np.float64,usecols=(0,2))
rand_params = np.zeros((N_sets,6),dtype=np.float64)

hidate, lodate = np.datetime64(date)+np.timedelta64(5,'m'),np.datetime64(date)-np.timedelta64(5,'m')
print hidate, lodate
arcdates = np.arange(lodate,hidate,np.timedelta64(30,'s'),dtype='datetime64[s]').astype(str)#+"T00:00:00.000"
#out_arc = np.apply_along_axis(find_pos2,0,arcdates,params[:,0])
#print out_arc

arcxyz = []
for i in arcdates:
	arcxyz.append(find_pos(params[:,0],i))
arcxyz=np.array(arcxyz)
#print arcxyz
'''
vfindpos = np.vectorize(find_pos,excluded=['ps'])
vfindpos.excluded.add(0)
out_arc = vfindpos(params[:,0],arcdates)
print out_arc'''

for e,i in enumerate(params):
	rand_params[:,e] = np.random.normal(i[0],i[1],N_sets)

out = np.apply_along_axis(find_pos,1,rand_params,date)
#print "{:.32f} {:.32f}".format(out[0,0],out[1,0])
#np.savetxt('params',rand_params)


fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(out[:,0],out[:,1],out[:,2],s=1)
ax.plot(arcxyz[:,0],arcxyz[:,1],arcxyz[:,2],'r-')

x,y,z = out[:,0],out[:,1],out[:,2]
means = np.mean(out,axis=0)

max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(x.max()+x.min())
Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(y.max()+y.min())
Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(z.max()+z.min())
# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
   ax.plot([xb], [yb], [zb], 'w')


#ERROR ELLISOID... :O

def rot_y(theta,vs):
	rotation = np.matrix([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
	
	return rotation * vs
	
def rot_z(theta,vs):
	rotation = np.matrix([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
	
	return rotation * vs
	
covxyz = np.cov(out.T)
lambdas, vectors = np.linalg.eig(covxyz)
rx,ry,rz = np.sqrt(lambdas) * 1

#set of all thetas
u = np.linspace(0,2*np.pi,100)
v = np.linspace(0,np.pi,100)

#convert to cartesian
x = rx * np.outer(np.cos(u), np.sin(v))
y = ry * np.outer(np.sin(u), np.sin(v))
z = rz * np.outer(np.ones_like(u), np.cos(v))

vmax = vectors[:,np.argmax(lambdas)]
thetay = -np.arctan(vmax[2]/vmax[1])
#print vmax[2],vmax[0]
thetaz = np.arctan(vmax[1]/vmax[0])

x,y,z = np.cos(thetay)*x+np.sin(thetay)*z, y, -np.sin(thetay)*x+np.cos(thetay)*z
x,y,z = np.cos(thetaz)*x-np.sin(thetaz)*y,np.sin(thetaz)*x+np.cos(thetaz)*y,z

ax.plot_surface(x+means[0], y+means[1], z+means[2], alpha = 0.25, rstride=4,cstride=4, color='g')

np.save('{}_ellipse_surface'.format(time.strftime('%Y%m%d_%H%M%S')),np.array([x+means[0], y+means[1], z+means[2]]))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

#ax.plot([means[0],means[0]+1e-5*vmax[0]],[means[1],means[1]+1e-5*vmax[1]],[means[2],means[2]+1e-5*vmax[2]],'y-')

fig2 = plt.figure(figsize=(3/4*6.4,3/4*3*6.4))

axs = []
for i in range(3):
	axs.append(fig2.add_subplot(311+i))

#PLOTTING ERROR ELLIPSES


for i in range(1,1+int(sys.argv[4])): #tell it how many ellipses you want
	#XY
	covxy = np.cov(out[:,0],out[:,1])
	eig_numxy, eig_vecxy = np.linalg.eig(covxy)
	sigmaxy = np.sqrt(eig_numxy)
	ell0 = Ellipse(xy=(means[0],means[1]),width=2*i*sigmaxy[1],
			height=2*i*sigmaxy[0],angle=np.rad2deg(np.arctan(eig_vecxy[0,0]/eig_vecxy[0,1])),edgecolor='red',zorder=5+i)
	ell0.set_facecolor('none')
	axs[0].add_artist(ell0)

	#YZ
	covyz = np.cov(out[:,1],out[:,2])
	eig_numyz, eig_vecyz = np.linalg.eig(covyz)
	sigmayz = np.sqrt(eig_numyz)
	ell1 = Ellipse(xy=(means[1],means[2]),width=2*i*sigmayz[1],
			height=2*i*sigmayz[0],angle=np.rad2deg(np.arctan(eig_vecyz[0,0]/eig_vecyz[0,1])),edgecolor='red',zorder=5+i)
	ell1.set_facecolor('none')
	axs[1].add_artist(ell1)

	#XZ
	cov = np.cov(out[:,0],out[:,2])
	eig_num, eig_vec = np.linalg.eig(cov)
	sigma = np.sqrt(eig_num)
	ell2 = Ellipse(xy=(means[0],means[2]),width=2*i*sigma[1],
			height=2*i*sigma[0],angle=np.rad2deg(np.arctan(eig_vec[0,0]/eig_vec[0,1])),edgecolor='red',zorder=5+i)
	ell2.set_facecolor('none')
	axs[2].add_artist(ell2)
	
	
	covxyz = np.cov(np.array([out[:,0],out[:,1],out[:,2]]))
	sigmax,sigmay,sigmaz = covxyz[0,0]**0.5,covxyz[1,1]**0.5,covxyz[2,2]**0.5
	
	if i==1:
		print "Means xyz:",means
		print "sigma xyz:{},{},{}".format(sigmax,sigmay,sigmaz)
		print "Ellipsoid volume: {}".format(4./3*np.pi*rx*ry*rz)
		#print rx,ry,rz

for a in axs: 
	a.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
	a.xaxis.set_major_locator(plt.MaxNLocator(5))
	a.yaxis.set_major_locator(plt.MaxNLocator(5))
	a.ticklabel_format(style='sci',scilimits=(0,0),axis='both')
	#a.set_aspect('equal')

axs[0].plot(out[:,0],out[:,1],'o',ms=1,zorder=1)

axs[0].set_xlabel(r'$x$ (AU)')
axs[0].set_ylabel(r'$y$ (AU)')

axs[1].plot(out[:,1],out[:,2],'o',ms=1,zorder=1)

axs[1].set_xlabel(r'$y$ (AU)')
axs[1].set_ylabel(r'$z$ (AU)')

axs[2].plot(out[:,0],out[:,2],'o',ms=1,zorder=1)

axs[2].set_xlabel(r'$x$ (AU)')
axs[2].set_ylabel(r'$z$ (AU)')
plt.tight_layout()
SAVEFIG.main('save?','PREDICTED_POSITION_PAT')

plt.show()
