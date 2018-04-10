from __future__ import division
import sys,csv
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.coordinates import Angle
from matplotlib.patches import Ellipse
import ephem
from datetime import datetime
import pandas as pd
from matplotlib import rcParams
import SAVEFIG

rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times']
rcParams['mathtext.fontset'] = 'cm'

obs = ephem.Observer()
obs.epoch = '2000'
obs.lon = '-1:34:24'
obs.lat = '54:46:01'
obs.elevation = 119.5


elements = open(str(sys.argv[1]),'r') #Minor planet centre data for all asteroids
#archival = str(sys.argv[2]) #y/n for archival data inclusion

specific_bodys = sys.argv[3:]
if specific_bodys == []: specific_bodys = 0
#print specific_bodys

bodys = []

while True:
	line = elements.readline()
	if line == '': break
	elif line.split()[0] == '#': continue
	else: bodys.append({'name':line.split(',')[0].split()[1],'xephem':line,'ra':1,'dec':1})

fig = plt.figure(figsize=(6.4*3/4,6.4*3/4))
ax = fig.add_subplot(111)

dates = np.arange(np.datetime64('2018-01-01'),np.datetime64('2018-03-16'),np.timedelta64(1,'h'))

#dates = np.arange(np.datetime64('2000-01-01'),np.datetime64('2020-01-01'),np.timedelta64(1,'D'))

if specific_bodys == 0:
	for i in bodys:
		asteroid = ephem.readdb(i['xephem'])
		observations = pd.read_csv('CSV_positions/{}.csv'.format(i['name']),sep=',',index_col=False)
		
		if archival == 'y':
			try:
				obs_archival =  pd.read_csv('CSV_positions/{}_archival.csv'.format(i['name']),sep=',',index_col=False)
				
				obs_ras = Angle(np.concatenate((np.array(observations['RA'][:]),np.array(obs_archival['RA'][:]))),u.hourangle)
				obs_decs = Angle(np.concatenate((np.array(observations['DEC'][:]),np.array(obs_archival['DEC'][:]))),u.deg)
				c = SkyCoord(ra=obs_ras,dec=obs_decs)
			
			except Exception:
				obs_ras = Angle(observations['RA'][:],u.hourangle)
				obs_decs = Angle(observations['DEC'][:],u.deg)
				c = SkyCoord(ra=obs_ras,dec=obs_decs)
		else:
			obs_ras = Angle(observations['RA'][:],u.hourangle)
			obs_decs = Angle(observations['DEC'][:],u.deg)
			c = SkyCoord(ra=obs_ras,dec=obs_decs)
		
		ras,decs = [], []
		for time in dates:
			obs.date = time.astype(datetime)
			asteroid.compute(obs)
			ras.append(asteroid.a_ra*180./np.pi) #converts radians to degrees
			decs.append(asteroid.a_dec*180./np.pi)
		
		ras,decs = np.array(ras),np.array(decs)
		#print np.mean(abs(ras[1:] - ras[:-1])),abs(ras[1:]-ras[:-1])
		
		decs[1:][abs(ras[1:]-ras[:-1])>100 * np.mean(abs(ras[1:] - ras[:-1]))] = np.nan
		ras[1:][abs(ras[1:]-ras[:-1])>100 * np.mean(abs(ras[1:] - ras[:-1]))] = np.nan
		#stops large lines where RA>2pi or RA<0
		#print decs,ras
		
		
		i['ra'],i['dec'] = ras,decs
		l, = ax.plot(ras,decs,'-',label=i['name'],zorder=1,ms=0.1,linewidth=0.5)
		ax.plot(c.ra.deg,c.dec.deg,'o',color=l.get_color(),zorder=20,ms=2)
		ax.text(c.ra.deg[-1]*1.01,c.dec.deg[-1]*1.01,i['name'],fontsize=12)
else:
	for i in bodys:
		if (i['name'] == np.array(specific_bodys)).any(): 
			asteroid = ephem.readdb(i['xephem'])
			observations = pd.read_csv('CSV_positions/{}.csv'.format(i['name']),sep=',',index_col=False)
			
			
			if archival == 'y':
				try:
					obs_archival =  pd.read_csv('CSV_positions/{}_archival.csv'.format(i['name']),sep=',',index_col=False)
					
					obs_ras = Angle(np.concatenate((np.array(observations['RA'][:]),np.array(obs_archival['RA'][:]))),u.hourangle)
					obs_decs = Angle(np.concatenate((np.array(observations['DEC'][:]),np.array(obs_archival['DEC'][:]))),u.deg)
					c = SkyCoord(ra=obs_ras,dec=obs_decs)
				
				except Exception:
					obs_ras = Angle(observations['RA'][:],u.hourangle)
					obs_decs = Angle(observations['DEC'][:],u.deg)
					c = SkyCoord(ra=obs_ras,dec=obs_decs)
			else:
				obs_ras = Angle(observations['RA'][:],u.hourangle)
				obs_decs = Angle(observations['DEC'][:],u.deg)
				c = SkyCoord(ra=obs_ras,dec=obs_decs)
		
			ras,decs = [], []
			for time in dates:
				obs.date = time.astype(datetime)
				asteroid.compute(obs)
				ras.append(asteroid.a_ra*180./np.pi) # converts radians to degrees
				decs.append(asteroid.a_dec*180./np.pi)
			
			ras,decs = np.array(ras),np.array(decs)
			#print np.mean(abs(ras[1:] - ras[:-1])),abs(ras[1:]-ras[:-1])
			
			decs[1:][abs(ras[1:]-ras[:-1])>100 * np.mean(abs(ras[1:] - ras[:-1]))] = np.nan
			ras[1:][abs(ras[1:]-ras[:-1])>100 * np.mean(abs(ras[1:] - ras[:-1]))] = np.nan
			#stops large lines where RA>2pi or RA<0
			#print decs,ras
			
			
			i['ra'],i['dec'] = ras,decs
			l, = ax.plot(ras,decs,'-',label=i['name'],zorder=1,ms=0.1,linewidth=0.5)
			ax.plot(c.ra.deg,c.dec.deg,'o',color=l.get_color(),zorder=20,ms=2)
			#ax.text(c.ra.deg[-1]*1.1,c.dec.deg[-1]*1.1,i['name'],fontsize=12)

FONTSIZE = 12


ax.set_xlabel('RA (deg)',fontsize = FONTSIZE)
ax.set_ylabel('Dec. (deg)',fontsize = FONTSIZE)
#ax.set_xlim((130,210))
#ax.set_ylim((-7.5,33))
ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
ax.legend()
plt.tight_layout()
SAVEFIG.main('save plot? (y/n)', 'OBSERVATIONS_MAP')
plt.show()
