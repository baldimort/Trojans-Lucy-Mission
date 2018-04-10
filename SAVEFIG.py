import matplotlib.pyplot as plt
import time

"""
This module was used to simplify saving of figures from matplotlib
If wou wish to use it, comment out the nulled main() function, uncomment
the function below, edit the file path in function to the desired location.

When SAVEFIG.main() is called, the two arguments required are:

message - string, the meassage to be printed to the terminal to 
prompt saving eg. "save astrometry figure? (y/n"

filetag - string, a tag inserted into the filename of the image
eg. "ASTROMETRY_PLOT"

The function will then save the figure most recently called
and save it in desired location with dpi=600 in png format.

"""
def main(null1, null2):
	return 0


'''def main(message,filetag):
	a = raw_input(message)
	if a == 'y': plt.savefig("/home/andrew/Uni/L3/Trojans/Report/Figs/"+time.strftime("%Y%m%d_%H%M%S_")+filetag+'.png',dpi=600,format='png')'''
