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

sys.path.insert(0, '../')
import SAVEFIG


date = "2033-03-02" #give date of the positon to be calculated
N_sets = 1000 #bootstrap - how many times?
ps = ("2018-03-23_124333_Parameters_inc_archival.dat","2018-03-23_124333_Parameters_exc_archival.dat")
#give the parameters txt file
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

fig = plt.figure(figsize=(3/4*6.4,3/4*6.4*2))


for i,parameters in enumerate(ps):
	ax = fig.add_subplot(211+i,projection='3d')
	params = np.genfromtxt(parameters,dtype=np.float64,usecols=(0,2))
	rand_params = np.zeros((N_sets,6),dtype=np.float64)

	'''hidate, lodate = np.datetime64(date)+np.timedelta64(5,'m'),np.datetime64(date)-np.timedelta64(5,'m')
	print hidate, lodate
	arcdates = np.arange(lodate,hidate,np.timedelta64(30,'s'),dtype='datetime64[s]').astype(str)#+"T00:00:00.000"'''
	#out_arc = np.apply_along_axis(find_pos2,0,arcdates,params[:,0])
	#print out_arc

	'''arcxyz = []
	for i in arcdates:
		arcxyz.append(find_pos(params[:,0],i))
	arcxyz=np.array(arcxyz)
	#print arcxyz'''
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


	ax.scatter(out[:,0],out[:,1],out[:,2],s=1)
	#ax.plot(arcxyz[:,0],arcxyz[:,1],arcxyz[:,2],'r-')

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
	rx,ry,rz = np.sqrt(lambdas) * 3

	#set of all thetas
	u = np.linspace(0,2*np.pi,200)
	v = np.linspace(0,np.pi,200)

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

	ax.plot_surface(x+means[0], y+means[1], z+means[2], alpha = 0.3, rstride=4,cstride=4, color='g')

	#np.save('{}_ellipse_surface'.format(time.strftime('%Y%m%d_%H%M%S')),np.array([x+means[0], y+means[1], z+means[2]]))
	ax.xaxis.set_major_locator(plt.MaxNLocator(5))
	ax.yaxis.set_major_locator(plt.MaxNLocator(5))
	ax.zaxis.set_major_locator(plt.MaxNLocator(5))
	plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	plt.ticklabel_format(style='sci', axis='z', scilimits=(0,0))


#ax.plot([means[0],means[0]+1e-5*vmax[0]],[means[1],means[1]+1e-5*vmax[1]],[means[2],means[2]+1e-5*vmax[2]],'y-')

ax.set_xlabel('x (AU)')
ax.set_ylabel('y (AU)')
ax.set_zlabel('z (AU)')
plt.tight_layout()

SAVEFIG.main('save?','ERROR_ELLIPSOID_PAT')

plt.show()
