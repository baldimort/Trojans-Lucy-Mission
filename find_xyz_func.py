#!/usr/bin/env python
#
#         python find_xyz.py 2010-11-15T17:26:45.808
#
#           reads the "orbital_elements" file
#
# uses Pyephem
#    see http://www.rhodesmill.org/brandon/projects/pyephem.html
#
def main():
	from ephem import *
	import time, math, commands, sys, os

	date_obs = sys.argv[1]
	year = date_obs[0:4]
	month = date_obs[5:7]
	day = date_obs[8:10]
	obstime = date_obs[11:]

	elements = open(sys.argv[2])
	minor = EllipticalBody()
	minor._epoch = '2000/01/01.5'   # epoch date for J2000.0
	while 1:
		line = elements.readline()
		if line == "": break
		word = line.split()
		if line.find('Object:') >= 0: obj_name = word[2]
		if line.find('Mean anomaly') >= 0: minor._M = float(word[3])
		if line.find('Argument of perihelion') >= 0: minor._om = float(word[4])
		if line.find('Long. of ascending node') >= 0: minor._Om = float(word[5])
		if line.find('Inclination') >= 0: minor._inc = float(word[2])
		if line.find('Eccentricity') >= 0: minor._e = float(word[2])
		if line.find('Semimajor axis') >= 0: minor._a = float(word[3])
		if line.find('Epoch of osculation  ') >= 0:
		    minor._epoch_M = word[4]+'/'+word[5]+'/'+word[6]
		if line.find('Absolute magnitude') >= 0: minor._H = float(word[4])
		if line.find('Slope parameter') >= 0: minor._G = float(word[4])    

	durham = Observer()
	durham.epoch = '2000'
	durham.long = '-1:34:24'
	durham.lat = '54:46:01'
	durham.temp = 10
	durham.elev = 119.5
	durham.date = year+'/'+month+'/'+day+ ' '+obstime

	const = 180./math.pi
	minor.compute(durham)
	pos = Equatorial(minor.a_ra,minor.a_dec, epoch='2000')
	ecl = Ecliptic(pos)
	ecl_long = float(ecl.lon)
	ecl_lat  = float(ecl.lat)

	d_minor = minor.earth_distance
	x_minor = d_minor*math.cos(ecl_lat)*math.cos(ecl_long)
	y_minor = d_minor*math.cos(ecl_lat)*math.sin(ecl_long)
	z_minor = d_minor*math.sin(ecl_lat)
	#print "minor", x_minor, y_minor, z_minor
	#print " "

	xxx = Sun()
	xxx.compute(durham)
	pos = Equatorial(xxx.a_ra,xxx.a_dec, epoch='2000')
	ecl = Ecliptic(pos)
	ecl_long = float(ecl.lon)
	ecl_lat  = float(ecl.lat)
	#print ecl_long*180./math.pi , ecl_lat*180./math.pi 
	d_sun = xxx.earth_distance
	x_sun = d_sun*math.cos(ecl_lat)*math.cos(ecl_long)
	y_sun = d_sun*math.cos(ecl_lat)*math.sin(ecl_long)
	z_sun = d_sun*math.sin(ecl_lat)
	#print "Sun  ",x_sun, y_sun, z_sun
	#print " "
	return (x_sun-x_minor, y_sun-y_minor, z_sun-z_minor)

