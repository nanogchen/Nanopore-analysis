#!/bin/env python

""" post analysis code for peo/ppo-grafted cylindrical nanopore/spherical nanoparticles"""

from groIO import groIO
from atom import Atom, AtomGrp
from func import getBox,getBoxMinmax,getRadCenter
import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np
import pandas as pd
import math
import sys

class Analysis:

	def __init__(self):
		# gro file read for the system
		self.System = groIO()

		# gld related
		self.GLD_type = None
		self.gld_atoms = AtomGrp()
		self.center = None
		self.radius = 0.0
		self.height = 0.0
		self.cutoff_hb = 5.0 # for nanoparticle

		# polymers related PEO/PPO
		self.POLY_type = None
		self.POLY_atoms = AtomGrp()
		self.backbone_atoms = AtomGrp()
		self.backbone_oxy_atoms = AtomGrp()
		self.backbone_s_atoms = AtomGrp()
		self.backbone_c_atoms = AtomGrp()
		self.Nchains = 0
		self.NrepeatUnits = 0
		self.Natoms_per_chain = 0

		# for hbond criterion: distance and angle
		self.DA_cr = 0.35
		self.HDA_deg = 30

		# solvent related
		self.SOL_atoms = AtomGrp()
		self.SOL_inpore_atoms = AtomGrp()
		self.SOL_oxy_inpore_atoms = AtomGrp()
		self.SOL_hyd_inpore_atoms = AtomGrp()

		# vol of atoms
		self.Vol_dict = {'H2O':0.03,\
						 'CH2':0.02,\
						   'O':0.02,}
		self.Vol_dict['S']=1.18*self.Vol_dict['O']

		# bin related
		self.binsize = 0.0
		self.bin_info = {}

		# bin related data
		self.backbone_oxy_atoms_binidx_dict={}
		self.backbone_s_atoms_binidx_dict={}
		self.backbone_c_atoms_binidx_dict={}
		self.SOL_oxy_inpore_atoms_binidx_dict={}

	def loadData(self, groFile, binsize, poly_type, gld_type):
		""" set the contents of the member variables using the gro file"""

		self.System.read_gro(groFile)
		self.binsize = binsize		

		########### set GLD
		self.GLD_type = gld_type
		self.gld_atoms = self.System.getAtomByResname(['GLD'])
		gld_pos = self.System.getPosByResname(['GLD'])
		self.radius, self.center = getRadCenter(self.GLD_type, gld_pos)
		self.height = self.System.box[2]

		########### set SOL
		self.SOL_atoms = self.System.getAtomByResname(['SOL'])

		if self.GLD_type == "PORE":
			bxmin,bxmax, bymin,bymax, _,_ = getBoxMinmax(gld_pos)
			self.SOL_inpore_atoms = self.SOL_atoms.selectAtoms(bxmin,bxmax, bymin,bymax)

			# only oxygen atoms within pore
			self.SOL_oxy_inpore_atoms = self.SOL_inpore_atoms.getAtomByAtomname(['OW'])

			# only hydrogen atoms within pore
			self.SOL_hyd_inpore_atoms = self.SOL_inpore_atoms.getAtomByAtomname(['HW1','HW2'])

		elif self.GLD_type == "NP":
			pass

		########### set poly
		self.POLY_type = poly_type
		if self.POLY_type == "PEO":
			
			# get PEO atoms
			self.POLY_atoms = self.System.getAtomByResname(['PE1','PE2','PE3'])
			self.Nchains = self.POLY_atoms.getAtomnames().count('SP1')

			if self.Nchains != 0:
				assert self.POLY_atoms.Natoms % self.Nchains == 0, "Total atoms of PEO should be divided by Nchains"
				self.Natoms_per_chain = int(self.POLY_atoms.Natoms / self.Nchains)
				self.NrepeatUnits = int((self.Natoms_per_chain - 5)/7)

			# get backbone
			self.backbone_atoms = self.System.getAtomByAtomname(['SP1','C21','C22','OP1','OP2','C31'])

			# get oxy atoms
			self.backbone_oxy_atoms = self.System.getAtomByAtomname(['OP1','OP2'])

			# get S atoms
			self.backbone_s_atoms = self.System.getAtomByAtomname(['SP1'])

			# get C atoms
			self.backbone_c_atoms = self.System.getAtomByAtomname(['C21','C22','C31'])

		elif self.POLY_type == "PPO":

			# get PPO atoms
			self.POLY_atoms = self.System.getAtomByResName(['PPO1','PPO2','PPO3'])
			self.Nchains = self.POLY_atoms.getAtomnames().count('SP1')

			if self.Nchains != 0:
				assert self.POLY_atoms.atom.Natoms % self.Nchains == 0, "Total atoms of PPO should be divided by Nchains"
				self.Natoms_per_chain = int(self.POLY_atoms.Natoms / self.Nchains)
				self.NrepeatUnits = int((self.Natoms_per_chain - 5)/10)

			# get backbone
			self.backbone_atoms = self.System.getAtomByAtomName(['SP1','C21','C11','OP1','CT1'])

			# get oxy atoms
			self.backbone_oxy_atoms = self.System.getAtomByAtomName(['OP1'])

			# get S atoms
			self.backbone_s_atoms = self.System.getAtomByAtomName(['SP1'])

			# get C atoms
			self.backbone_c_atoms = self.System.getAtomByAtomName(['C21','C11','C31','CT1'])
			
	def setBins(self):	

		#-----------------------------------------------
		# get parameters of bin-related 
		binsize = self.binsize
		radius = self.radius
		bz = self.height

		if self.GLD_type == "PORE":
			# from surface to the center
			MaxDist = radius

		elif self.GLD_type == "NP":
			# from gld surface to 5nm away
			MaxDist = self.cutoff_hb

		# loop the bins and the system and count the volumn in each bins
		Maxbins = int(MaxDist/binsize) + 1

		# generate the z dist array (distance to the gold surface)
		z_dist = [i*binsize for i in range(Maxbins+1)]

		# check for distance
		if z_dist[-1] > MaxDist:
			z_dist[-1] = MaxDist	

		# check if divide completely
		if abs(z_dist[-1]-z_dist[-2]) < 5.0e-2:
			z_dist.pop()
			Maxbins -= 1

		# if the dist between last two bin is less than the binsize, then merge into one bin
		if z_dist[-1] - z_dist[-2] < 0.85*binsize: # account for oscillation errors
			del z_dist[-2]
			Maxbins -= 1

		if self.GLD_type == "PORE":
			# generate the bin dist array: from surface to the center
			bin_rad = [radius - i for i in z_dist]

			# calculate the volume of each bin
			vol_bin = [math.pi*(bin_rad[i]**2 - bin_rad[i+1]**2)*bz for i in range(Maxbins)]

		elif self.GLD_type == "NP":
			# bin radius
			bin_rad = [radius + i for i in z_dist]

			# volume of each bin
			vol_bin = [4.0/3.0*math.pi*(bin_rad[i+1]**3 - bin_rad[i]**3) for i in range(Maxbins)]

		# set the bin info
		self.bin_info['Maxbins'] = Maxbins
		self.bin_info['Zdist'] = z_dist
		self.bin_info['bin_rad'] = bin_rad
		self.bin_info['bin_vol'] = vol_bin

	def giveBins(self, rad):

		# assert self.binsize == 0.3, "Not implement for other binsize except {} nm.".format(0.3)
		radius = self.radius
		bz = self.height

		#-----------------------------------------------
		# give bin-related parameters
		if rad == 2.0:
			#z_dist = [0.0, 0.4, 0.8, 1.2, 1.6, radius]
			z_dist = [0.0, 0.3, 0.6, 0.9, 1.2, 1.6, radius]

		elif rad == 2.5:
			# z_dist = [0.0, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, radius]
			# z_dist = [0.0, 0.4, 0.9, 1.2, 1.5, 1.8, 2.1, radius]
			# z_dist = [0.0, 0.4, 0.8, 1.1, 1.4, 1.7, 2.0, radius]
			z_dist = [0.0, 0.4, 0.9, 1.1, 1.5, 1.7, radius]

		elif rad == 3.0:
			z_dist = [0.0, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, radius]

		# no of bins
		Maxbins = len(z_dist)-1

		# generate the bin dist array: from surface to the center
		bin_rad = [radius - i for i in z_dist]

		# calculate the volume of each bin
		vol_bin = [math.pi*(bin_rad[i]**2 - bin_rad[i+1]**2)*bz for i in range(Maxbins)]
		
		# set the bin info
		self.bin_info['Maxbins'] = Maxbins
		self.bin_info['Zdist'] = z_dist
		self.bin_info['bin_rad'] = bin_rad
		self.bin_info['bin_vol'] = vol_bin

	def setBinIdxForAtoms(self, Atomgroups):

		# get the distance criteria
		tosur_dist = self.bin_info['Zdist']

		# get the center
		center = self.center
		radius = self.radius

		# set the bin idx for these atoms
		for iporeAtom in Atomgroups.Atoms:
			xpos = iporeAtom.x - center[0]
			ypos = iporeAtom.y - center[1]
			r = math.sqrt(xpos**2 + ypos**2)
			dist2sur = radius - r

			# get bin idx
			for i in range(1,len(tosur_dist)):
				if dist2sur < tosur_dist[i]:
					bin_idx = i
					break

				if i == len(tosur_dist)-1: # if still larger than the radius
					bin_idx = len(tosur_dist)-1

			# set bin idx
			iporeAtom.setBinIdx(bin_idx)

	def setBinIdxForAll(self):

		# for sol
		self.setBinIdxForAtoms(self.SOL_oxy_inpore_atoms)
		# self.setBinIdxForAtoms(self.SOL_hyd_inpore_atoms) # not involved in calculation

		# for polymer
		self.setBinIdxForAtoms(self.backbone_atoms) # just set for backbone with the three below accounted.
		self.setBinIdxForAtoms(self.backbone_oxy_atoms)
		self.setBinIdxForAtoms(self.backbone_s_atoms)
		self.setBinIdxForAtoms(self.backbone_c_atoms)

	def getBinCountsForAtoms(self, atoms):

		Maxbins = self.bin_info['Maxbins']
		dat_dict = {}

		#initialize
		for i in range(1, Maxbins+1):
			dat_dict[i] = 0

		# place data in
		for atom in atoms:
			dat_dict[atom.bin_idx]+=1

		return dat_dict			

	def setHbonds(self):
		""" determine H-bond between POLY-oxy and SOL-H atoms

		calculate the hydrogen bond using criteria:
		(1) D-A distance <= 3.5A (0.35 nm);
		(2) H-D-A angle <= 30 deg

		only water-oxygen atom can be Hbond donar, and POLY-oxygen atoms are acceptors.
	
		"""

		# the critiria
		DA_cr = self.DA_cr
		HDA_deg = self.HDA_deg

		# loop over SOL oxy first
		sol_oxy_idx = 0
		for sol_oxy_atom in self.SOL_oxy_inpore_atoms.Atoms:

			# pos of SOL-oxy atom
			sol_oxy_pos = sol_oxy_atom.getPos()

			# second loop: POLY-oxy
			for poly_oxy_atom in self.backbone_oxy_atoms.Atoms:

				# get its pos
				poly_oxy_pos = poly_oxy_atom.getPos()

				# account for pbc along z (vector DA->, CANNOT be in reverse!)
				DA = poly_oxy_pos - sol_oxy_pos
				if DA[2] > 0.5*self.height:
					DA[2] = DA[2] - self.height

				if DA[2] < -0.5*self.height:
					DA[2] = DA[2] + self.height							

				# distance between two oxy atoms
				if LA.norm(DA) <= DA_cr: # unit in nm

					# get Hs pos
					hyd_atom1 = self.SOL_hyd_inpore_atoms.Atoms[2*sol_oxy_idx]
					hyd_atom2 = self.SOL_hyd_inpore_atoms.Atoms[2*sol_oxy_idx+1]
					jH1pos = hyd_atom1.getPos()
					jH2pos = hyd_atom2.getPos()

					angle_list = []
					for iHpos in [jH1pos, jH2pos]:						
						DH = iHpos - sol_oxy_pos

						# account for pbc along z
						if DH[2] > 0.5*self.height:
							DH[2] = DH[2] - self.height

						if DH[2] < -0.5*self.height:
							DH[2] = DH[2] + self.height

						cosHDA = np.dot(DA, DH)/(LA.norm(DA)*LA.norm(DH))
						cosHDA=max(-1.0,cosHDA)
						cosHDA=min(1.0,cosHDA)
						angle = math.acos(cosHDA) * 180.0 / math.pi
						angle_list.append(angle)

					if angle_list[0] <= HDA_deg:	
						sol_oxy_atom.setHbondPair([hyd_atom1, poly_oxy_atom])
						poly_oxy_atom.setHbondPair([hyd_atom1, sol_oxy_atom])

					if angle_list[1] <= HDA_deg:	
						sol_oxy_atom.setHbondPair([hyd_atom2, poly_oxy_atom])
						poly_oxy_atom.setHbondPair([hyd_atom2, sol_oxy_atom])
			
			sol_oxy_idx += 1

	def setHydShell(self):
		"""get the wat in the hyd-shell for POLY-backbone atom"""

		# the critiria
		DA_cr = self.DA_cr

		# first loop: POLY-oxy
		for poly_atom in self.backbone_atoms.Atoms:		

			# pos of poly-backbone atom
			poly_pos = poly_atom.getPos()

			# second loop: 
			for sol_oxy_atom in self.SOL_oxy_inpore_atoms.Atoms:

				# get its pos
				sol_oxy_pos = sol_oxy_atom.getPos()

				# account for pbc along z (vector DA->, CANNOT be in reverse!)
				DA = poly_pos - sol_oxy_pos
				if DA[2] > 0.5*self.height:
					DA[2] = DA[2] - self.height

				if DA[2] < -0.5*self.height:
					DA[2] = DA[2] + self.height							

				# distance between two oxy atoms
				if LA.norm(DA) <= DA_cr:

					poly_atom.setHydShell(sol_oxy_atom)

	def getHBondList(self):
		"""get the hbonding list [D, H, A]"""

		# list to store the hbonds 
		hbonds_list = []

		for iatom in POREanal.SOL_oxy_inpore_atoms.Atoms:
			
			# if involved in Hbonding
			if iatom.hbond_pair_atoms:
				for hb_pair in iatom.hbond_pair_atoms:
					hbonds_list.append([iatom.atomid, hb_pair[0].atomid, hb_pair[1].atomid])

		np.savetxt("Hbonds-POLY-SOL.txt", np.array(hbonds_list), fmt="%d")

		return hbonds_list

	def getPOLYoxyHbonds_chainwise(self):
		"""get the hbond info for each oxygen in the polymer backbone"""

		# loop over POLY-oxy atom
		POLYoxyHbonds = [len(atom.hbond_pair_atoms) for atom in self.backbone_oxy_atoms.Atoms]

		# reshape into array
		POLYoxyHbonds = np.array(POLYoxyHbonds)
		NrepeatUnits = self.NrepeatUnits
		Nchains = self.Nchains
		try:
			POLYoxyHbonds = np.reshape(POLYoxyHbonds, (Nchains, NrepeatUnits))
		except:
			print("Error in getPOLYoxyHbond")

		np.savetxt("POLYoxy-hbonded.txt", POLYoxyHbonds, fmt="%2d")

	def binIdx2binCount(self, bin_idx):
		"""convert bin idx to counts in each bin"""

		# bin data
		Maxbins = self.bin_info['Maxbins']
		counts_list = []

		# list to np array
		if isinstance(bin_idx, list):
			bin_idx = np.array(bin_idx)

		for binidx in range(1,Maxbins+1):
			counts = np.count_nonzero(bin_idx ==binidx)
			counts_list.append(counts)

		return counts_list

	def getVolFrac(self):

		# get volume of each bin
		vol_bin = self.bin_info['bin_vol']
		VolH2O = self.Vol_dict['H2O']
		VolCH2 = self.Vol_dict['CH2']
		VolO = self.Vol_dict['O']
		VolS = self.Vol_dict['S']	

		# for SOL
		sol_bin_idx = [iatom.bin_idx for iatom in self.SOL_oxy_inpore_atoms.Atoms]
		sol_bin_count = self.binIdx2binCount(sol_bin_idx)

		sol_vol = VolH2O*np.array(sol_bin_count)
		sol_vol_frac = np.divide(sol_vol,np.array(vol_bin))			

		# for POLY
		S_bin_idx = [iatom.bin_idx for iatom in self.backbone_s_atoms.Atoms]
		C_bin_idx = [iatom.bin_idx for iatom in self.backbone_c_atoms.Atoms]
		O_bin_idx = [iatom.bin_idx for iatom in self.backbone_oxy_atoms.Atoms]

		S_bin_count = self.binIdx2binCount(S_bin_idx)
		C_bin_count = self.binIdx2binCount(C_bin_idx)
		O_bin_count = self.binIdx2binCount(O_bin_idx)		

		POLY_vol = VolS*np.array(S_bin_count) + VolCH2*np.array(C_bin_count) + VolO*np.array(O_bin_count)
		POLY_vol_frac = np.divide(POLY_vol,np.array(vol_bin))

		# out data
		cols = ['SOL','POLY','SUM']
		data = [sol_vol_frac, POLY_vol_frac, sol_vol_frac+POLY_vol_frac]

		dat_dict = dict(zip(cols, data))
		self.binData_to_csv("vol_frac.csv", dat_dict, AddBinIdx_bool=True)

		# return sol_vol_frac, POLY_vol_frac
	
	def getSOLDensity(self):
		""" get the SOL density in the pore"""

		if self.GLD_type == "PORE":

			Natoms = self.SOL_inpore_atoms.Natoms
			assert Natoms % 3 == 0, "Total SOL atoms should be divided by 3 (H2O)"
			NSOLs = int(Natoms / 3)

			# get mols of SOL
			NA = 6.022e23
			nmols = NSOLs/NA

			# get mass of SOL in gram
			mass_sol = nmols * 18.0154

			# get mols of Polymer PPO/PEO
			nmols_poly = self.Nchains/NA
			NrepeatUnits = self.NrepeatUnits
			if self.POLY_type == "PEO":
				molmass = 32.06+12.011*(2*NrepeatUnits+1)+15.9994*(NrepeatUnits)+1.008*(4*NrepeatUnits+3)
			elif self.POLY_type == "PPO":
				molmass = 32.06+12.011*(3*NrepeatUnits+1)+15.9994*(NrepeatUnits)+1.008*(6*NrepeatUnits+3)

			# polymer mass
			mass_poly = nmols_poly * molmass

			# get volume of pore in cm^3
			vol = math.pi*(self.radius)**2*(self.height)
			vol = vol*(1e-7)**3 # nm to cm: 10^7

			# get density
			density = (mass_sol+mass_poly)/vol

			with open('density.txt','w') as FO:
				FO.write("Density of peo/sol within Nanopore:{:8.3f}.\n".format(density))

			return density

	def getFreeSOLFrac(self):

		# free wat and hb wat
		free_wat_binwise = {}
		hb_wat_binwise = {}

		# initialize
		Maxbins = self.bin_info['Maxbins']
		for i in range(1, Maxbins+1):
			free_wat_binwise[i] = []
			hb_wat_binwise[i] = []

		# place data in
		for atom in self.SOL_oxy_inpore_atoms.Atoms:
			
			# hb wat
			if atom.hbond_pair_atoms:
				hb_wat_binwise[atom.bin_idx].append(atom.atomid)

			else:
				free_wat_binwise[atom.bin_idx].append(atom.atomid)

		# loop over each bin, and count freesol
		freesol_frac = []
		for key in free_wat_binwise.keys():
			frac = len(free_wat_binwise[key])/(len(free_wat_binwise[key]) + len(hb_wat_binwise[key]))
			freesol_frac.append(frac)

		# to csv
		cols = ['freesol']
		dat = [freesol_frac]
		data_dict = dict(zip(cols, dat))
		self.binData_to_csv("freesol.csv", data_dict, AddBinIdx_bool=True)

	def getHbondsFrac(self):

		# free wat and hb wat
		hbonds_binwise = {}
		POLY_oxy_binwise = {}

		# initialize
		Maxbins = self.bin_info['Maxbins']
		for i in range(1, Maxbins+1):
			hbonds_binwise[i] = 0
			POLY_oxy_binwise[i] = 0

		# place data in
		for atom in self.backbone_oxy_atoms.Atoms:

			# POLY-oxy
			POLY_oxy_binwise[atom.bin_idx]+=1
			
			# POLY-oxy-hbonds
			if atom.hbond_pair_atoms:
				hbonds_binwise[atom.bin_idx]+=len(atom.hbond_pair_atoms)

		# loop over each bin, and count freesol
		hbonds_frac = []
		for key in POLY_oxy_binwise.keys():
			frac = hbonds_binwise[key]/POLY_oxy_binwise[key]
			hbonds_frac.append(frac)

		# to csv
		cols = ['Hbonds']
		dat = [hbonds_frac]
		data_dict = dict(zip(cols, dat))
		self.binData_to_csv("Hbonds_perPEO.csv", data_dict, AddBinIdx_bool=True)

	def getHydNumber(self):
		"""get the hydration number"""
		
		# find wat-shell for polymer backbone atoms
		self.setHydShell()
		Maxbins = self.bin_info['Maxbins']

		# dict for the chain level count
		hyd_shell_counts_binwise = {}
		hyd_number = {}
		for i in range(1,1+Maxbins):
			hyd_shell_counts_binwise[i] = 0
			hyd_number[i] = 0

		# const
		Natoms_per_backbone = int(len(self.backbone_atoms.Atoms)/self.Nchains)

		# loop over each chain (hydration shell: a concept for a polymer chain)
		for ichain in range(self.Nchains):
			idx_start,idx_end = ichain*Natoms_per_backbone, (ichain+1)*Natoms_per_backbone

			hyd_shell_sol_atoms_ichain = []
			# loop over each atom in backbone chain
			for atom in self.backbone_atoms.Atoms[idx_start:idx_end]:
				if atom.Hshell_wat_atoms:
					hyd_shell_sol_atoms_ichain.extend(atom.Hshell_wat_atoms)

			# unique wat sol within the shell
			hyd_shell_sol_unique_atoms_ichain = list(set(hyd_shell_sol_atoms_ichain))
			hshell_count_dict = self.getBinCountsForAtoms(hyd_shell_sol_unique_atoms_ichain)
			
			# add sol number to the bix counts
			for i in range(1,1+Maxbins):
				hyd_shell_counts_binwise[i] += hshell_count_dict[i]

		# get poly-oxy counts binwise
		POLY_oxy_counts = self.getBinCountsForAtoms(self.backbone_oxy_atoms.Atoms)

		# calculate hydnumber
		for i in range(1,1+Maxbins):
			hyd_number[i] = hyd_shell_counts_binwise[i]/POLY_oxy_counts[i]

		cols = ['hydNum']
		dat = [list(hyd_number.values())]
		self.binData_to_csv("hydration_number.csv", dict(zip(cols, dat)), AddBinIdx_bool=True)

	def getHydShellCorr(self, shellrad=1.0):
		"""get the Hbonded sol within the radius of rad for water residence correlation function"""

		# pos of sol-oxy
		sol_pos = np.array([atom.getPos() for atom in self.SOL_oxy_inpore_atoms.Atoms])
		sol_idx = np.array([atom.atomid for atom in self.SOL_oxy_inpore_atoms.Atoms])
		sol_hb_idx = np.array([atom.atomid for atom in self.SOL_oxy_inpore_atoms.Atoms if atom.hbond_pair_atoms ])

		# center
		center = self.center
		center = np.expand_dims(center, axis=0)

		# get sols within the shell
		dist_pos = sol_pos - center
		dist = LA.norm(dist_pos[:,:2], axis=1)
		in_shell_bool =  dist <= shellrad

		# get idx
		in_shell_idx = sol_idx[in_shell_bool]
		in_shell_hbonded_idx = set(in_shell_idx).intersection(set(sol_hb_idx))
		in_shell_hbonded_idx = np.expand_dims(np.array(list(in_shell_hbonded_idx)), axis=1)

		# save dat
		np.savetxt("in_shell_hbonded_sol_idx.txt", in_shell_hbonded_idx, fmt="%d")
		np.savetxt("in_shell_sol_idx.txt", np.expand_dims(in_shell_idx, axis=1), fmt="%d")

	def getEndGroup(self):
		"""get the end group distance distribution to the surface"""
		
		# get the center pts
		center = self.center

		# initialize the arrays
		EndDist = []
		pos = []
		
		# the end atom C31
		for atom in self.backbone_c_atoms.Atoms:
			if atom.atomname == 'C31':
				pos.append(atom.getPos())		

		# get the pos of the end group atoms
		if self.GLD_type == "PORE":
			EndDist = [self.radius - LA.norm(ipos[:2]-center[:2]) for ipos in pos]
		
		elif self.GLD_type == "NP":
			EndDist = [LA.norm(ipos-center) - self.radius for ipos in pos]

		return np.array(EndDist)

	# def getSurfaceAtomFrac(self):
	# 	"""get the frac of POLY atoms near the surface, i.e., within 3.5A dist to surface"""

	# 	if self.GLD_type == "PORE":

	# 		# get the pos of poly-atoms
	# 		pos = np.array([atom.getPos() for atom in self.backbone_atoms.Atoms])
	# 		Natoms = pos.shape[0]

	# 		# reshape data into [chains, Natoms_per_chain, 3]
	# 		assert Natoms % self.Nchains == 0, "Total backbone atoms should be divided by Nchains!"
	# 		Natoms_per_backbone = int(Natoms/self.Nchains)
	# 		pos_0 = np.reshape(pos, (self.Nchains, Natoms_per_backbone, 3))

	# 		# get the center pts
	# 		center = getBoxCenter(self.gld_atoms.atom.getPos())
	# 		center = np.expand_dims(center, axis=(0,1))

	# 		# dist of each atom only in (x,y)
	# 		pos_0 = pos_0 - center
	# 		dist = LA.norm(pos_0[:,:,:2], axis=2)

	# 		# check how many of them are within the near-surface
	# 		criteria = self.radius - 0.35
	# 		surAtom = dist[dist >= criteria]
	# 		surAtom_Natoms = surAtom.size

	# 		return surAtom_Natoms/Natoms

	# 	else:

	# 		return 0.0

	def getRadialCOM(self):
		"""get the radial distance of center of mass for each chain"""

		pos = np.array([atom.getPos() for atom in self.backbone_atoms.Atoms])
		Natoms = pos.shape[0]
		Nchains = self.Nchains

		# reshape data into [Nchains, Natoms_per_chain, 3]
		assert Natoms % Nchains == 0, "Total backbone atoms should be divided by Nchains!"
		Natoms_per_backbone = int(Natoms/Nchains)

		# pos in 3d tensor
		polypos = np.reshape(pos, (Nchains, Natoms_per_backbone, 3))

		# calculate the radial distance
		center = self.center
		if self.GLD_type == "PLANAR":
			# use only z
			center = np.expand_dims(center[2], axis=(0,1))
			vect = polypos[:,:,2] - center[:,:,2]		

		elif self.GLD_type == "PORE":
			# use [x,y]
			center = np.expand_dims(center[:2], axis=(0,1))
			vect = polypos[:,:,:2] - center[:,:,:2]
			
		elif self.GLD_type == "NP":
			# use [x,y,z]
			center = np.expand_dims(center, axis=(0,1))
			vect = polypos - center

		# calculate com
		dist = np.linalg.norm(vect, axis=2)
		Rcom = np.mean(dist, axis=1)
	
		return Rcom

	def getBrushHeight(self, volfrac, bh_type):
		"""get the brush height from its vol. frac. data

		refs: Grafted polymers inside cylindrical tubes: Chain stretching vs layer thickness (Eq.23);
			  Monte Carlo simulation of polymer brushes in narrow pores (Eq.19)
		
		"""

		# get the bin dist
		bin_dist = self.bin_info['Zdist'][1:]
		bin_dist = np.array(bin_dist)
		Radius = self.radius

		if bh_type == "firstMoment": # h_{FM}

			# integrate the eq. using trapzoidal rule
			denominator = integrate(bin_dist, volfrac)
			nominator = integrate(bin_dist, np.multiply(bin_dist, volfrac))			

		elif bh_type == "monomerDist": # h

			# integrate the eq. using trapzoidal rule and monomer No. distr.

			# obtain n_c(h) first
			nch = []
			for h in bin_dist:

				# get the value of volfrac(R-h) using interpolation
				phi_R_h = getInterpolate(bin_dist, volfrac, Radius-h)
				nch_i = (1.0 - h/Radius)*phi_R_h
				nch.append(nch_i)

			denominator = integrate(bin_dist, nch)
			nominator = integrate(bin_dist, np.multiply(bin_dist, nch))

		# calculate the value
		brushHeight = nominator/denominator

		return brushHeight

	def getBrushHeight_h(self):
		"""get the brush height using the other forms:		
		<h> = f1/f2
		f1 = int_0^R (h * (R-h) * phi(R-h)) dh
		f2 = int_0^R (    (R-h) * phi(R-h)) dh

		by change of variables -->
		<h> = R - [int_0^R (r^2 * phi(r)) dr]/[int_0^R (r * phi(r)) dr]
		"""

		z_dist = self.bin_info['Zdist']
		Maxbins = self.bin_info['Maxbins']

		r_dist = z_dist[::-1]
		r_dist = [self.radius-i for i in r_dist]
		bz = self.height
		bin_vol_r = [math.pi*(r_dist[i+1]**2 - r_dist[i]**2)*bz for i in range(Maxbins)]

		# volume
		VolCH2 = self.Vol_dict['CH2']
		VolO = self.Vol_dict['O']
		VolS = self.Vol_dict['S']

		# start from center to the gold surface, divide POLY backbone atoms into diff bins
		if self.GLD_type == "PORE":

			# original bin idx: from surface to center
			S_bin_idx = self.backbone_s_atoms.bin_idx
			O_bin_idx = self.backbone_oxy_atoms.bin_idx
			C_bin_idx = self.backbone_c_atoms.bin_idx

			# from center to surface idx
			S_bin_idx_r = [Maxbins-i+1 for i in S_bin_idx]
			O_bin_idx_r = [Maxbins-i+1 for i in O_bin_idx]
			C_bin_idx_r = [Maxbins-i+1 for i in C_bin_idx]

			# bin_count
			S_bin_count = self.binIdx2binCount(np.array(S_bin_idx_r))
			O_bin_count = self.binIdx2binCount(np.array(O_bin_idx_r))
			C_bin_count = self.binIdx2binCount(np.array(C_bin_idx_r))			
			
			# calcualte the volume fraction in each bin: phi(r)
			mat_vol = VolS*np.array(S_bin_count) + VolCH2*np.array(C_bin_count) + VolO*np.array(O_bin_count)
			mat_vol_frac = np.divide(mat_vol, np.array(bin_vol_r))

			# do the calculation for brush height
			int_r = integrate(r_dist[1:], np.multiply(np.multiply(r_dist[1:],r_dist[1:]), mat_vol_frac))/integrate(r_dist[1:], np.multiply(r_dist[1:], mat_vol_frac))
			h = self.radius - int_r

		else:
			print("not yet implemented! Exiting ...")

		return h	

	def binData_to_csv(self, fname, dat_dict, AddBinIdx_bool):

		df = pd.DataFrame.from_dict(dat_dict)

		# add dist
		if AddBinIdx_bool:
			tosur_dist = self.bin_info['Zdist']
			dist = np.array(tosur_dist[1:])
			df.insert (0, "h", dist)

		df.to_csv(fname, index=False, float_format='%.3f')

	def saveBinNdx(self,fname,bin_dict):
		"""save the oxy atoms in each bin of PEO and SOL"""

		# assert type(bin_dict) == dict, "Only dict type is allowed for bin ndx!"

		with open(fname, 'w') as fo:

			# header
			fo.write("[ Counts ]\n")
			for idx in bin_dict.keys():
				fo.write("{:5d}".format(len(bin_dict[idx])))
			fo.write("\n")
			fo.write("\n")

			for idx in bin_dict.keys():

				# write content of each bin
				fo.write("[ bin-{} ]\n".format(idx))

				# write idx of oxy atoms
				idx_list = bin_dict[idx]
				idx_list.sort()
				counter = 0
				for item in idx_list:
					counter+=1
					fo.write("{:5d} ".format(item))

					if counter == 10:
						fo.write("\n")
						counter = 0

				fo.write("\n")
				fo.write("\n")

#---------------------------------------- other functions for efficiency

def getInterpolate(x,y,xp):
	"""given x and y arrays, get the interpolation value at xp"""

	for i in range(len(x)):
		if xp <= x[i]:

			if i > 0:
				yp = y[i-1] + (y[i]-y[i-1])/(x[i]-x[i-1])*(xp-x[i-1])
				break

			else:
				yp = y[i]
				break

	return yp

def integrate(x,f):
	"""integrate f(x) using trapzoidal rule"""

	int_sum = 0.0

	for i in range(len(x)-1):
		i_area = 0.5*(f[i]+f[i+1])*(x[i+1]-x[i])
		int_sum += i_area

	return int_sum

# --------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
	if len(sys.argv) != 7:
		print("WRONG format. RIGHT call: {}  groFile Rad(nm) binsize(nm) binstyle(give,auto) poly_type(PEO,PPO) gld_type(PLANAR, PORE, NP)".format(sys.argv[0]))
		sys.exit(0)

	groFile = sys.argv[1]
	Rad = float(sys.argv[2])
	binsize = float(sys.argv[3])
	binstyle = sys.argv[4]
	poly_type = sys.argv[5]
	assert poly_type in ['PEO','PPO'], f"ONLY PEO and PPO are supported currently!"
	gld_type = sys.argv[6]
	assert gld_type in ['PLANAR','NP','PORE'], f"ONLY PLANAR, NP (nanoparticle) and PORE (nanopore) are supported currently!"

	POREanal = Analysis()
	POREanal.loadData(groFile, binsize, poly_type, gld_type)

	# set bins
	if binstyle == 'auto':
		POREanal.setBins()
	elif binstyle == 'give':
		POREanal.giveBins(Rad)
	else:
		print("WRONG bin style: can only be auto or give")
		sys.exit(0)

	# set Bin idx
	POREanal.setBinIdxForAll()	

	# # set Hbonding info
	POREanal.setHbonds()

	POREanal.getHBondList()

	# check
	hb_sol_counts=0
	sol_counts=0
	for atom in POREanal.SOL_oxy_inpore_atoms.Atoms:
		
		if atom.bin_idx == 1:
			sol_counts+=1
			# print(f"{atom.atomid}: {len(atom.hbond_pair_atoms)}")

			if atom.hbond_pair_atoms:
				hb_sol_counts+=1

			else:
				print(f"{atom.resid:5d}, {atom.atomid:5d}")

	print(f"Total sols: {sol_counts}")
	print(f"Hb-sols: {hb_sol_counts}")
	print(f"Free sol fraction: {1-hb_sol_counts/sol_counts}")

	# # poly-oxy
	# poly_oxy_counts=0
	# poly_oxy_hb_counts=0
	# poly_oxy_hb_bin1_counts=0
	# for atom in POREanal.backbone_oxy_atoms.Atoms:		
	# 	if atom.bin_idx == 1:
	# 		poly_oxy_counts+=1

	# 		if atom.hbond_pair_atoms:

	# 			for ipair in atom.hbond_pair_atoms:
	# 				print(ipair[1].bin_idx)
	# 				if ipair[1].bin_idx == 1:
	# 					poly_oxy_hb_bin1_counts+=1
	# 					break

	# 			poly_oxy_hb_counts+=1

	# print(f"POLY-oxy counts: {poly_oxy_counts}")
	# print(f"POLY-oxy-hbonded counts: {poly_oxy_hb_counts}")
	# print(f"POLY-oxy-hbonded counts in bin1: {poly_oxy_hb_bin1_counts}")

	# POREanal.getVolFrac()

	# POREanal.getFreeSOLFrac()

	# POREanal.getHbondsFrac()

	# hbonds_list = POREanal.getHBondList()
	# # print(hbonds_list)
	# POREanal.getPOLYoxyHbonds_chainwise()

	# enddist = POREanal.getEndGroup()
	# print(enddist)

	# POREanal.getHydShellCorr()

	# POREanal.getHydNumber()

	# # do analysis
	# # get density
	# soldes = POREanal.getSOLDensity()
	# # with open('density.txt','w') as FO:
	# # 	FO.write("Density of peo/sol within Nanopore:{:8.3f}.\n".format(soldes))
	# print(soldes)

	# #--------------------------------------------------------------------------------
	# # volume fraction
	# #--------------------------------------------------------------------------------

	# PEO_vol_frac = POREanal.getVolFrac(POREanal.backbone_atoms.getPos(), 'PEO-S')
	# # print(PEO_vol_frac)

	# SOL_vol_frac = POREanal.getVolFrac(POREanal.SOL_oxy_inpore_atoms.getPos(), 'H2O')
	# # print(SOL_vol_frac)	

	# # create a dataframe
	# vol_frac_df = pd.DataFrame(columns=['z', 'H2O', 'PEO', 'sum'])
	# vol_frac_df['z'] = POREanal.bin_info['Zdist'][1:]
	# vol_frac_df['H2O'] = SOL_vol_frac
	# vol_frac_df['PEO'] = PEO_vol_frac
	# vol_frac_df['sum'] = SOL_vol_frac+PEO_vol_frac

	# # save
	# NrepeatUnits = int((POREanal.Natoms_per_chain - 5)/7)
	# outfile = "vol_frac" + "_R" + str(POREanal.radius) + "_nPEO" + str(NrepeatUnits) + "_N" + str(POREanal.Nchains) 
	# vol_frac_df.to_csv(outfile + ".csv", index=False, float_format='%.3f')

	# # use original data for plotting
	# vol_frac_df.plot(x='z', y=['H2O','PEO','sum'])
	# plt.savefig(outfile + ".png", dpi=1000, bbox_inches='tight')

	# #--------------------------------------------------------------------------------
	# # conformation
	# #--------------------------------------------------------------------------------

	# backbone_Ree = POREanal.getRee(POREanal.backbone_atoms.getPos())
	# np.savetxt("backbone_Ree_arr.txt", backbone_Ree, fmt="%8.3f")
	# # print(backbone_Ree)

	# backbone_Rg = POREanal.getRg(POREanal.backbone_atoms.getPos())
	# np.savetxt("backbone_Rg_arr.txt", backbone_Rg, fmt="%8.3f")
	# # print(backbone_Rg)

	# backbone_Ree_div_Rg = np.divide(backbone_Ree, backbone_Rg)
	# np.savetxt("backbone_Ree_div_Rg.txt", backbone_Ree_div_Rg, fmt="%8.3f")

	# #--------------------------------------------------------------------------------
	# # H-bonding
	# #--------------------------------------------------------------------------------

	# SOL_oxy_Hbonds = POREanal.calHBond(POREanal.backbone_oxy_atoms.getPos(), POREanal.SOL_oxy_inpore_atoms.getPos(), POREanal.SOL_hyd_inpore_atoms.getPos())
	# # print(SOL_oxy_Hbonds)
	# np.savetxt('SOL_oxy_Hbonds.txt',SOL_oxy_Hbonds,fmt="%d")
	
	# HBond_perPEO_bin = POREanal.getHbondFrac(POREanal.PEO_oxy_bin_idx,POREanal.SOL_oxy_bin_idx,SOL_oxy_Hbonds)
	# # print(HBond_perPEO_bin)

	# # save
	# hbond_df = pd.DataFrame(columns=['z', 'Hbond-frac'])
	# hbond_df['z'] = POREanal.bin_info['Zdist'][1:]
	# hbond_df['Hbond-frac'] = HBond_perPEO_bin

	# outfile = "Hbond_frac" + "_R" + str(POREanal.radius) + "_nPEO" + str(NrepeatUnits) + "_N" + str(POREanal.Nchains) 
	# hbond_df.to_csv(outfile + ".csv", index=False, float_format='%.3f')

	# # use original data for plotting
	# hbond_df.plot(x='z', y=['Hbond-frac'])
	# plt.savefig(outfile + ".png", dpi=1000, bbox_inches='tight')
