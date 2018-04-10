from __future__ import division
import sys,csv
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.coordinates import Angle
from matplotlib.patches import Ellipse
from matplotlib import rcParams

sys.path.insert(0, '../')

import SAVEFIG

rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times']
rcParams['mathtext.fontset'] = 'cm'


f = open(str("Anchises_180124.csv"),'rb') #1st argument is csv file containg data

file = csv.reader(f)

RA_JPL, DEC_JPL, RA_ESA, DEC_ESA, RA_DEF, DEC_DEF, RA_UCAC4, DEC_UCAC4 = [], [], [], [], [], [], [] ,[]
#DATA = np.zeros(

for line in file:
	for e,i in enumerate((RA_JPL, DEC_JPL, RA_ESA, DEC_ESA, RA_DEF, DEC_DEF, RA_UCAC4, DEC_UCAC4)):
		i.append(line[e+3])

def convert_RA_DEC_degs(inRA,inDEC):
	'''Converts list of string angles in HH:MM:SS.000 to degress'''
	ra = Angle(inRA,u.hourangle)
	dec = Angle(inDEC,u.deg)
	c = SkyCoord(ra=ra,dec=dec)
	outRA = c.ra.degree
	outDEC = c.dec.degree
	return (outRA,outDEC)
	
def convert_arcsec(inlist):
	'''converts list of string angles in HH:MM:SS.000 or numpy array in degs to arcsec'''
	'''if type(inlist)==type([6]) or type(inlist)==type((6)):
		outlist = []
		for angle in inlist:
			hours = float(angle[0:2])
			mins = float(angle[3:5])
			secs = float(angle[6:])
			outlist.append(hours/24.*360*3600 + mins*60. + secs)
		return outlist'''
	if type(inlist)==type(np.array([1,2])):
		return inlist*3600


#CONVERTS CSV TO ARRAYS IN DEGREES
RA_JPL,DEC_JPL = np.array(convert_RA_DEC_degs(RA_JPL[1:],DEC_JPL[1:]))
RA_ESA,DEC_ESA = np.array(convert_RA_DEC_degs(RA_ESA[1:],DEC_ESA[1:]))
RA_DEF, DEC_DEF = np.array(convert_RA_DEC_degs(RA_DEF[1:],DEC_DEF[1:]))
RA_UCAC4, DEC_UCAC4 = np.array(convert_RA_DEC_degs(RA_UCAC4[1:],DEC_UCAC4[1:]))

#CALCULATES RESIDUALS IN ARCSEC
DEC_res_DEF = convert_arcsec(DEC_DEF - DEC_JPL)
RA_cos_res_DEF = (convert_arcsec(RA_DEF - RA_JPL))*np.cos(DEC_JPL*np.pi/180.)

DEC_res_ESA = convert_arcsec(DEC_ESA - DEC_JPL)
RA_cos_res_ESA = (convert_arcsec(RA_ESA - RA_JPL))*np.cos(DEC_JPL*np.pi/180.)

DEC_res_UCAC4 = convert_arcsec(DEC_UCAC4 - DEC_JPL)
RA_cos_res_UCAC4 = (convert_arcsec(RA_UCAC4 - RA_JPL))*np.cos(DEC_JPL*np.pi/180.)


data = np.array([[RA_cos_res_DEF,DEC_res_DEF],[RA_cos_res_ESA,DEC_res_ESA],[RA_cos_res_UCAC4,DEC_res_UCAC4]])
medians = np.mean(data,axis=2)
print "Mean offsets RA/DEC (UCAC2,ESA,UCAC4)" 
print medians

meds3d = np.zeros(data.shape)
meds3d[0,0,:] = medians[0,0]
meds3d[0,1,:] = medians[0,1]
meds3d[1,0,:] = medians[1,0]
meds3d[1,1,:] = medians[1,1]
meds3d[2,0,:] = medians[2,0]
meds3d[2,1,:] = medians[2,1]

rms = (np.sum((data-meds3d)**2,axis=2)/data.shape[2])**0.5
print "RMS"
print rms

#PLOTTING
fig = plt.figure(figsize=(6.4*3/4,6.4*3/4))
ax = fig.add_subplot(111)


ax.plot(RA_cos_res_ESA,DEC_res_ESA,'ro',ms=2,label='ESA GAIA')
ax.plot(RA_cos_res_DEF,DEC_res_DEF,'bo',ms=2,label='UCAC2')
ax.plot(RA_cos_res_UCAC4,DEC_res_UCAC4,'go',ms=2,label='UCAC4')

colours = ['blue','red','green']
for i in range(3):
	#PLOTTING ERROR ELLIPSES
	cov = np.cov(data[i,0,:],data[i,1,:])
	eig_num, eig_vec = np.linalg.eig(cov)
	sigma = np.sqrt(eig_num)
	ell = Ellipse(xy=(medians[i,0],medians[i,1]),width=2*sigma[0],
			height=2*sigma[1],angle=np.rad2deg(np.arctan(eig_vec[1,1]/eig_vec[1,0])),edgecolor=colours[i])
	ell.set_facecolor('none')
	ax.add_artist(ell)

t = np.linspace(0,2*np.pi,1000)
#ax.plot(rms[0,0]*np.cos(t)+medians[0,0],rms[0,1]*np.sin(t)+medians[0,1],'b--')
#ax.plot(rms[1,0]*np.cos(t)+medians[1,0],rms[1,1]*np.sin(t)+medians[1,1],'r--')

ax.set_xlabel(r'$\Delta RA \times cos(Dec)\ (arcsecs)$')
ax.set_ylabel(r'$\Delta Dec \ (arcsecs)$')


#ax.set_aspect('equal')
#ax.set_aspect('equal', 'datalim')
ax.axhline(0,linestyle='--')
ax.axvline(0,linestyle='--')
ax.legend()
plt.tight_layout()
'''a = raw_input('save y/n?')
if a == 'y': plt.savefig('Astrom_compare.svg',dpi=600,format='svg')
'''
SAVEFIG.main('save fig? (y/n)','ASTROM_COMPARE')
plt.show()
