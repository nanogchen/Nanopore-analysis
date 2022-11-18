#!/bin/env python

""" custom functions called from the other classes"""

import numpy.linalg as la
from numpy.linalg import norm
from numba import njit, prange
import numpy as np
import math

# used in nanobuilder: place peo chains
def getTransformationMat(norm_vector):
	""" get the tranformation matrix using the normal vector at a grafting point """

	# the ref vector
	ref_vec = np.asarray([1.0, 0.0, 0.0])

	# get the angle between ref and norm vector
	cos_value = ref_vec@norm_vector/(norm(ref_vec)*norm(norm_vector))
	theta = math.acos(cos_value)

	# if gft pts in the upper semi-circle, then adjust the theta angle
	if norm_vector[1] <= 0.0:
		theta = 2*math.pi - theta

	# rotation matrix
	RotMat = np.asarray([[math.cos(theta), math.sin(theta), 0], \
							  [-math.sin(theta), math.cos(theta), 0], \
							  [0, 0, 1]])

	# translation matrix: Au-S bond length 0.265 nm
	Au_S = 0.265
	TransMat = np.asarray([norm_vector[0]*Au_S, norm_vector[1]*Au_S, 0.0])
	TransMat = np.expand_dims(TransMat, axis=0)

	return RotMat,TransMat

def checkPOLYxyz(center, radius, xyz):
	""" check the generated polymer's coordinate: within the nanopore"""

	# Natoms
	Natoms = xyz.shape[0]
	Nbadatoms = 0	

	for i in range(Natoms):
		vec = xyz[i,:2] - center[:2]
		if norm(vec) > radius:			
			Nbadatoms += 1

	assert Nbadatoms == 0,"Bad generation of the PEO system: {} atoms are in the gld region!".format(Nbadatoms)

@njit()
def getNN(xyz, lattice=0.408):
	"""
	get the nearest bond info of given atoms
	"""

	# nearest dist
	dist = lattice/math.sqrt(2)
	tol = lattice*0.1

	# bonds
	Natoms = xyz.shape[0]
	Max = Natoms*(Natoms-1)
	bonds = np.empty((Max, 2), dtype=np.int_)	

	# loop
	nbonds = 0
	for i in prange(Natoms):
		for j in prange(i+1,Natoms):

			vec_ij = xyz[i] - xyz[j]

			if la.norm(vec_ij) < dist + tol:
				bonds[nbonds,:] = np.asarray([i,j])
				nbonds += 1

	return bonds

@njit() # donot use "parallel=True"!
def getNN_two(pos1, pos2):

	"""
	get the nearest neighbor (atom2) of atom 1
	"""

	atom_idx = np.empty((pos1.shape[0]), dtype=np.int_)

	# loop each sulfur atom
	for i in prange(pos1.shape[0]):
		ipos = pos1[i,:]
		dist = 100
		idx = -1

		# loop each gold atom
		for j in prange(pos2.shape[0]):
			jpos = pos2[j,:]
			if la.norm(jpos-ipos) < dist:
				dist = la.norm(jpos-ipos)
				idx = j

		atom_idx[i] = idx

	return pos2[atom_idx,:], atom_idx+1

def getRadCenter(GLD_type, pos):
	""" get the radius of the nanopore from its pos """

	# get the center pts
	center = getBoxCenter(pos)

	# use np for efficiency: get the disp vector
	pos = pos - center		

	# for pore 
	if GLD_type == "PORE":
		dist = norm(pos[:,:2], axis=1)
		dist = np.sort(dist) # from the smallest to the largest: use the minimum
		rad = np.mean(dist[:72])

	elif GLD_type == "NP":
		dist = norm(pos, axis=1)
		dist = np.sort(dist)
		rad = np.mean(dist[-6:]) # from the smallest to the largest: use the maximum

	elif GLD_type == "PLANAR":
		print("NOT IMPLEMENT yet!")
		sys.exit(0)

	return rad,center

def getBox(pos):
	"""get box of the pos"""

	(xmin, ymin, zmin) = np.amin(pos, axis=0)
	(xmax, ymax, zmax) = np.amax(pos, axis=0)
	
	return np.asarray([xmax-xmin, ymax-ymin, zmax-zmin])

def getBoxMinmax(pos):
	"""get box of the pos"""

	(xmin, ymin, zmin) = np.amin(pos, axis=0)
	(xmax, ymax, zmax) = np.amax(pos, axis=0)
	
	return np.asarray([xmin,xmax, ymin,ymax, zmin,zmax])

def getBoxCenter(pos):
	"""get box center of the pos"""

	(xmin, ymin, zmin) = np.amin(pos, axis=0)
	(xmax, ymax, zmax) = np.amax(pos, axis=0)
	
	return np.asarray([(xmin+xmax)*0.5, (ymin+ymax)*0.5, (zmin+zmax)*0.5])

def getRad(pos):
		""" get the radius of the nanopore from its pos """

		# get the center pts
		center = getBoxCenter(pos)

		# find 8 pts and do average
		Natoms = pos.shape[0]

		# initialize rad
		rad = 1.0e2

		for i in range(Natoms):
			ipos = pos[i,:]
			dist = norm(ipos-center)

			if dist < rad:
				rad = dist

		return rad

def getCorner(pos, shift, lattice=0.408):
	"""get the corner pts of a gro file, return the idx"""

	(xmin, ymin, zmin) = np.amin(pos, axis=0)
	(xmax, ymax, zmax) = np.amax(pos, axis=0)

	# N atoms
	Natoms = pos.shape[0]
	idx_list = []

	# 8 corners
	corner_pts = np.array([[xmin,ymin,zmin],\
							[xmax,ymin,zmin],\
							[xmax,ymax,zmin],\
							[xmin,ymax,zmin],\
							[xmin,ymin,zmax],\
							[xmax,ymin,zmax],\
							[xmax,ymax,zmax],\
							[xmin,ymax,zmax]])

	# loop over each atom
	for icor in corner_pts:
		for i in range(Natoms):

			ipos = pos[i,:]		
			dist_vect = ipos - icor

			# if norm(dist_vect) < lattice*0.5: # not the best as multiple pts suffice
			if norm(dist_vect) < 1e-5:
				idx_list.append(i+1)

	# save
	np.savetxt('cornerPts.txt',np.asarray(idx_list)+shift,fmt="%d")

		