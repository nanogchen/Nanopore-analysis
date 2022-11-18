#!/bin/env python

import sys
# from analysis import Analysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def getInfo(fname):

	usage_info = f"""
	Specify the functions to execute:

	{fname} -all: do all calculations available as below
	{fname} -volfrac: calculate volume fraction
	{fname} -enddist: calculate end-group distribution
	{fname} -hbshell: calculate hydration shell or hydration number
	{fname} -hbonds: calculate hydrogen bonding infomation
	{fname} -freesol: calculate free sol fraction
	{fname} -hbcorr: calculate hydration shell correlation function
	"""	

	return usage_info

#-------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

	if len(sys.argv) == 1:
		print(getInfo(sys.argv[0]))
		sys.exit(0)

	elif len(sys.argv) != 6:
		print("WRONG format. RIGHT call: {} -flag groFile Rad(nm) binsize(nm) binstyle(give,auto)".format(sys.argv[0]))
		print(getInfo(sys.argv[0]))
		sys.exit(0)

	flag = sys.argv[1]
	if flag not in ['-volfrac','-enddist','-hbshell','-hbonds','-freesol','-hbcorr','-all']:
		print("Unknow flag")
		print(getInfo(sys.argv[0]))
		sys.exit(0)

	groFile = sys.argv[2]
	Rad = float(sys.argv[3])
	binsize = float(sys.argv[4])
	binstyle = sys.argv[5]

	PEOanal = Analysis(groFile, binsize)

	# # set variables
	# PEOanal.setVars(groFile, binsize)
	
	# set GLD
	PEOanal.setGLD()
	# sys.exit(1)

	# set bins
	if binstyle == 'auto':
		PEOanal.setBins()
	elif binstyle == 'give':
		PEOanal.giveBins(Rad)
	else:
		print("WRONG bin style: can only be auto or give")
		sys.exit(0)

	# set SOL
	PEOanal.setSOL()
	# sys.exit(1)
	
	# set PEO
	PEOanal.setPEO()
	# sys.exit(1)

	# print height
	print("Height: {}".format(PEOanal.height))

	# do analysis
	# get density
	soldes = PEOanal.getSOLDensity()
	with open('density.txt','w') as FO:
		FO.write("Density of peo/sol within Nanopore:{:8.3f}.\n".format(soldes))
	# print(soldes)
	# sys.exit(1)

	#--------------------------------------------------------------------------------
	# volume fraction
	#--------------------------------------------------------------------------------

	# PEO_vol_frac = PEOanal.getVolFrac(PEOanal.backbone_atoms.atom.getPos(), 'PEO-S')
	# # print(PEO_vol_frac)

	# SOL_vol_frac = PEOanal.getVolFrac(PEOanal.SOL_oxy_inpore_atoms.atom.getPos(), 'H2O')
	# # print(SOL_vol_frac)

	SOL_vol_frac = PEOanal.getVolFrac_by_bin('SOL')
	PEO_vol_frac = PEOanal.getVolFrac_by_bin('PEO')

	# create a dataframe
	vol_frac_df = pd.DataFrame(columns=['z', 'H2O', 'PEO', 'sum'])
	z = PEOanal.bin_info['Zdist'][1:]
	# z[0] = z[0]*0.5 # use the middle pt in the first bin
	vol_frac_df['z'] = np.asarray(z)
	vol_frac_df['H2O'] = SOL_vol_frac
	vol_frac_df['PEO'] = PEO_vol_frac
	vol_frac_df['sum'] = SOL_vol_frac+PEO_vol_frac

	# save
	NrepeatUnits = int((PEOanal.Natoms_per_chain - 5)/7)
	outfile = "vol_frac" + "_R" + str(PEOanal.radius) + "_nPEO" + str(NrepeatUnits) + "_N" + str(PEOanal.Nchains) 
	vol_frac_df.to_csv(outfile + ".csv", index=False, float_format='%.3f')

	# use original data for plotting
	vol_frac_df.plot(x='z', y=['H2O','PEO','sum'])
	plt.savefig(outfile + ".png", dpi=1000, bbox_inches='tight')
	# sys.exit(1)

	#--------------------------------------------------------------------------------
	# end group (last C) distribution
	#--------------------------------------------------------------------------------

	enddist = PEOanal.getEndGroup(PEOanal.backbone_atoms.getPos())
	np.savetxt("enddist.txt", enddist, fmt="%8.3f")
	# sys.exit(1)
	
	#--------------------------------------------------------------------------------
	# H-bonding and free sol
	#--------------------------------------------------------------------------------

	SOL_oxy_Hbonds,hbonds_arr = PEOanal.calHBond(PEOanal.backbone_oxy_atoms, PEOanal.SOL_oxy_inpore_atoms, PEOanal.SOL_hyd_inpore_atoms)
	# print(SOL_oxy_Hbonds)
	# np.savetxt('SOL_oxy_Hbonds.txt',SOL_oxy_Hbonds,fmt="%d")
	np.savetxt('Hbonds-PEO-SOL.txt',hbonds_arr,fmt="%d")
	# sys.exit(1)
	
	# HBond_perPEO_bin
	HBond_perPEO_bin = PEOanal.binCal(hbonds_arr[:,2], 'PEO')
	# print(HBond_perPEO_bin)
	np.savetxt('HBond_perPEOoxy_bin.txt',HBond_perPEO_bin,fmt="%.3f")

	freesol = 1.0 - PEOanal.binCal(hbonds_arr[:,0], 'SOL')
	# print(freesol)
	# np.savetxt('freesol.txt',freesol,fmt="%.3f")
	freesol_df = pd.DataFrame(columns=['z','freesol'])
	freesol_df['z'] = z
	freesol_df['freesol'] = freesol
	outfile = "freesol" + "_R" + str(PEOanal.radius) + "_nPEO" + str(NrepeatUnits) + "_N" + str(PEOanal.Nchains) 
	freesol_df.to_csv(outfile+".csv", index=False, float_format='%.3f')

	# save
	hbond_df = pd.DataFrame(columns=['z', 'Hbond-frac'])
	hbond_df['z'] = z
	hbond_df['Hbond-frac'] = HBond_perPEO_bin

	outfile = "Hbond_frac" + "_R" + str(PEOanal.radius) + "_nPEO" + str(NrepeatUnits) + "_N" + str(PEOanal.Nchains) 
	hbond_df.to_csv(outfile + ".csv", index=False, float_format='%.3f')

	# use original data for plotting
	hbond_df.plot(x='z', y=['Hbond-frac'])
	plt.savefig(outfile + ".png", dpi=1000, bbox_inches='tight')

