from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pp
import numpy as np
from matplotlib import cm
import pandas as pd
from matplotlib.patches import Ellipse


data_to_plot = pd.read_csv("data_xyz.csv", sep = ",", index_col=0)

fig = pp.figure()
ax = fig.add_subplot(111, projection='3d')
xs = data_to_plot['x']
ys = data_to_plot['y']
zs = data_to_plot['z']

ax.scatter(xs, ys, zs,s=1)

dist=0.5/5000
distz=0.005/5000 #AU
Nx=0
Ny=0
Nz=0
a=np.zeros(3)
b=np.zeros(3)
c=np.zeros(3)
xav=np.mean(xs)
yav=np.mean(ys)
zav=np.mean(zs)
'''for j in range(5000):
	for i in range(len(xs)):
		xpos=abs(-xav + xs[i])
		ypos=abs(-yav + ys[i])
		zpos=abs(-zav + zs[i])
		if j*dist < xpos < (j+1)*dist:
			Nx+=1
			if Nx == int(len(xs)*0.682):
				a[0] = (j+1)*dist
			if Nx == int(len(xs)*0.9544):
				a[1] = (j+1)*dist
			if Nx == int(len(xs)*0.997):
				a[2] = (j+1)*dist

		if j*dist < ypos < (j+1)*dist:
			Ny+=1
			if Ny == int(len(xs)*0.682):
				b[0] = (j+1)*dist
			if Ny == int(len(xs)*0.9544):
				b[1] = (j+1)*dist
			if Ny == int(len(xs)*0.997):
				b[2] = (j+1)*dist

		if j*distz < zpos < (j+1)*distz:
			Nz+=1
			if Nz == int(len(xs)*0.682):
				c[0] = (j+1)*distz
			if Nz == int(len(xs)*0.9544):
				c[1] = (j+1)*distz
			if Nz == int(len(xs)*0.997):
				c[2] = (j+1)*distz'''

#plot ellipsoid


pp.figure()

#xy plot

covxy = np.cov(xs, ys)
lambda_xy, vxy = np.linalg.eig(covxy)
lambda_xy = np.sqrt(lambda_xy)

ax1 = pp.subplot(221)
for j in xrange(1, 4):
    ell = Ellipse(xy=(xav, yav),
                  width=lambda_xy[0]*j*2, height=lambda_xy[1]*j*2,
                  angle=-np.rad2deg(np.arccos(vxy[0, 0])),color='red')
    ell.set_facecolor('none')
    ax1.add_artist(ell)
pp.scatter(xs, ys, color='black')
pp.xlim(xav-0.001,xav+0.001)
pp.ylim(yav-0.001,yav+0.001)
pp.xlabel('x distance from Sun/AU')
pp.ylabel('y distance from Sun/AU')
#yz plot

covyz = np.cov(ys, zs)
lambda_yz, vyz = np.linalg.eig(covyz)
lambda_yz = np.sqrt(lambda_yz)

ax2 = pp.subplot(222)
for j in xrange(1, 4):
    ell = Ellipse(xy=(yav, zav),
                  width=lambda_yz[0]*j*2, height=lambda_yz[1]*j*2,
                  angle=-np.rad2deg(np.arccos(vyz[0, 0])),color='red')
    ell.set_facecolor('none')
    ax2.add_artist(ell)
pp.scatter(ys, zs, color='black')
pp.xlim(yav-0.001,yav+0.001)
pp.ylim(zav-0.001,zav+0.001)
pp.xlabel('y distance from Sun/AU')
pp.ylabel('z distance from Sun/AU')

#xz plot

covxz = np.cov(xs, zs)
lambda_xz, vxz = np.linalg.eig(covxz)
lambda_xz = np.sqrt(lambda_xz)

ax3 = pp.subplot(223)
for j in xrange(1, 4):
    ell = Ellipse(xy=(xav, zav),
                  width=lambda_xz[0]*j*2, height=lambda_xz[1]*j*2,
                  angle=-np.rad2deg(-np.arccos(vxz[0, 0])),color='red')
    ell.set_facecolor('none')
    ax3.add_artist(ell)
pp.scatter(xs, zs, color='black')
pp.xlim(xav-0.001,xav+0.001)
pp.ylim(zav-0.001,zav+0.001)
pp.xlabel('x distance from Sun/AU')
pp.ylabel('z distance from Sun/AU')

pp.show()
