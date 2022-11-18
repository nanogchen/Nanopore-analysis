import sys
import numpy as np
from atom import Atom, AtomGrp

class groIO:

	def __init__(self):
		self.atom_lists = AtomGrp()
		self.box = np.array([None, None, None])

	def getBox(self):
		return self.box

	def setBox(self, box):
		self.box = box

	def read_gro(self, fname):
		"""
		add element by reading gro file
		format = %5d%-5s %5s %5d %8.3f%8.3f%8.3f %8.4f%8.4f%8.4f
		"""

		# spacing for the data
		space_list = [5,5,5,5,8,8,8,8,8,8]

		# file read
		with open(fname, 'r') as FI:

			# the content
			groFile = FI.readlines()

			# get the box size from the last line
			box = groFile[-1].split()
			bx = float(box[0])
			by = float(box[1])
			bz = float(box[2])
			self.box = np.array([bx, by, bz])

			# add atoms into the list
			for iline in groFile[2:-1]:

				# split the string
				resnb =  iline[:sum(space_list[:1])].strip()
				resnm =  iline[sum(space_list[:1]):sum(space_list[:2])].strip()
				atomnm = iline[sum(space_list[:2]):sum(space_list[:3])].strip()
				atomnb = iline[sum(space_list[:3]):sum(space_list[:4])].strip()
				x = 	 iline[sum(space_list[:4]):sum(space_list[:5])].strip()
				y = 	 iline[sum(space_list[:5]):sum(space_list[:6])].strip()
				z = 	 iline[sum(space_list[:6]):sum(space_list[:7])].strip()
				u = 	 iline[sum(space_list[:7]):sum(space_list[:8])].strip()
				v = 	 iline[sum(space_list[:8]):sum(space_list[:9])].strip()
				w = 	 iline[sum(space_list[:9]):sum(space_list[:])].strip()
				
				# to float
				(x,y,z) = map(float, (x,y,z))
				try:
					(u,v,w) = map(float, (u,v,w))
				except:
					u,v,w =	None, None, None		

				# gro atom
				atom = Atom()
				atom.setResId(int(resnb))
				atom.setResName(resnm)				
				atom.setAtomName(atomnm)
				atom.setAtomId(int(atomnb))
				atom.setX(x)
				atom.setY(y)
				atom.setZ(z)
				atom.setU(u)
				atom.setV(v)
				atom.setW(w)

				# add into the lists
				self.atom_lists.append(atom)

	def getPosByAtomname(self, atomname_list):
		pos = []

		for iatom in self.atom_lists.Atoms:
			if iatom.atomname in atomname_list:
				pos.append([iatom.x, iatom.y, iatom.z])

		return np.array(pos)

	def getPosByResname(self, resname_list):
		pos = []

		for iatom in self.atom_lists.Atoms:
			if iatom.atomname in resname_list:
				pos.append([iatom.x, iatom.y, iatom.z])

		return np.array(pos)

	def getAtomTypes(self):
		""" return the unique atom types of the atoms"""

		atomtypes = []
		for iatom in self.atom_lists:
			atomtypes.append(iatom.atomname)

		uniq = list(set(atomtypes))
		uniq.sort()

		return uniq

	def getResTypes(self):
		""" return the unique residue types of the atoms"""

		restypes = []
		for iatom in self.atom_lists.Atoms:
			restypes.append(iatom.resname)

		uniq = list(set(restypes))
		uniq.sort()

		return uniq

	def getAtomByResname(self, resname_list):
		"""return atoms by res name"""

		atom_lists = AtomGrp()
		for iatom in self.atom_lists.Atoms:
			if iatom.resname in resname_list:
				atom_lists.append(iatom)

		return atom_lists

	def getAtomByAtomname(self, atomname_list):
		"""return atoms by atom name"""

		atom_lists = AtomGrp()
		for iatom in self.atom_lists.Atoms:
			if iatom.atomname in atomname_list:
				atom_lists.append(iatom)

		return atom_lists

	def selectAtoms(self,bxmin,bxmax, bymin,bymax):
		""" select atoms satisfying conditions"""

		atom_lists = AtomGrp()
		for iatom in self.atom_lists.Atoms:
			if iatom.x>bxmin and iatom.x<bxmax:
				if iatom.y>bymin and iatom.y<bymax:
					atom_lists.append(iatom)

		return atom_lists

	def to_gro(self,fname):
		"""write out gro file given atoms"""

		if len(self.atom_lists) == 0:
			print("No atoms to write")
			sys.exit(0)

		# write out gro file for VMD visualization
		with open(fname, 'w') as FO:

			# add header
			FO.write("Write by gmxpy \n")
			FO.write("%5d\n" % (len(self.atom_lists)))

			for iatom in self.atoms_list:
				FO.write("{:5d}{:5s}{:5s}{:5d}{:8.3f}{:8.3f}{:8.3f}\n".format(iatom.resid, \
					iatom.resname, iatom.atomname, iatom.atomid, iatom.x, iatom.y, iatom.z, \
					iatom.u, iatom.v, iatom.w))

			# add tailer
			FO.write("{:10.5f}{:10.5f}{:10.5f}\n".format(self.box[0],self.box[1],self.box[2]))

if __name__ == '__main__':

	if len(sys.argv) != 2:
		print("Right call: {} groFile".format(sys.argv[0]))
		sys.exit(0)
	
	fname=sys.argv[1]
	groObj = groIO()
	groObj.read_gro(fname)
	print(len(groObj.atom_lists))
	print(groObj.box)
	print(groObj.getAtomTypes())
	print(groObj.getResTypes())

	peo_atoms = groObj.getAtomByResname(['PE1','PE2','PE3'])
	print(len(peo_atoms))

	gld = groObj.getAtomByAtomname(['GLD'])
	print(len(gld))
	print(type(gld))

