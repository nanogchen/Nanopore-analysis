import numpy as np

class Atom:
	"""
	atom as a basic data type for post analysis
	"""

	def __init__(self):

		# basic
		self.atomname = None
		self.atomid = None
		self.atomtype = None
		self.charge = None
		self.chargeid = None
		self.mass = None
		self.resname = None
		self.resid = None
		self.x = None
		self.y = None
		self.z = None		
		self.u = None
		self.v = None
		self.w = None

		# for post cal
		self.bin_idx = None
		self.Hshell_wat_atoms = [] # a list of atoms giving the wat in hyd-shell for poly
		self.hbond_pair_atoms = [] # a list of atoms giving the donor-acceptor pair

	def setAtomITP(self,atomid,atomtype,resid,resname,atomname,chargeid,charge,mass):
		self.atomname = atomname
		self.atomid = atomid
		self.atomtype = atomtype
		self.charge = charge
		self.chargeid = chargeid
		self.mass = mass
		self.resname = resname
		self.resid = resid

	def setAtomName(self, atomname):
		self.atomname = atomname

	def setAtomId(self, atomid):
		self.atomid = atomid

	def setResName(self, resname):
		self.resname = resname

	def setResId(self, resid):
		self.resid = resid

	def getPos(self):
		return np.array([self.x, self.y, self.z])

	def setX(self, x):
		self.x = x

	def setY(self, y):
		self.y = y

	def setZ(self, z):
		self.z = z

	def setU(self, u):
		self.u = u

	def setV(self, v):
		self.v = v

	def setW(self, w):
		self.w = w

	def setBinIdx(self, idx):
		self.bin_idx = idx

	def setHbondPair(self, atom):
		"""a list for donor-acceptor pairs
		[] for no hbonding
		[163] for single hbonds
		[1,2] for double hbonds
		"""
		self.hbond_pair_atoms.append(atom)

	def setHydShell(self, atom):
		"""a list for wat id within hyd-shell"""

		self.Hshell_wat_atoms.append(atom)

# -------------------------------------------------------------------------------------

class AtomGrp:

	"""
	atom groups: a list of atoms (pdb, gro, pore), so that each atom object can be looped over
	"""
	
	def __init__(self):
		self.Atoms = []
		self.Natoms = 0

	def form_AtomGrp_from_list_of_atoms(self, atom_list):
		
		Atoms = AtomGrp()
		for iatom in atom_list:
			Atoms.append(iatom)

		return Atoms

	def append(self, atom):
		""" add atom to the atom group"""

		self.Natoms += 1
		self.Atoms.append(atom)

	def getUniqueAtomTypes(self):
		""" return the unique atom types of the atoms"""

		atomtypes = []
		for iatom in self.Atoms:
			atomtypes.append(iatom.atomtype)

		uniq = list(set(atomtypes))
		uniq.sort()

		return uniq

	def getUniqueAtomNames(self):
		""" return the unique atom types of the atoms"""

		atomtypes = []
		for iatom in self.Atoms:
			atomtypes.append(iatom.atomname)

		uniq = list(set(atomtypes))
		uniq.sort()

		return uniq

	def getResTypes(self):
		""" return the unique atom types of the atoms"""

		restypes = []
		for iatom in self.Atoms:
			restypes.append(iatom.resname)

		uniq = list(set(restypes))
		uniq.sort()

		return uniq

	def getAtomnames(self):
		return [iatom.atomname for iatom in self.Atoms]

	def getAtomids(self):		
		return [iatom.atomid for iatom in self.Atoms]

	def getResnames(self):
		return [iatom.resname for iatom in self.Atoms]

	def getResids(self):
		return [iatom.resid for iatom in self.Atoms]

	def getPos(self):
		""" return the pos of the atoms"""

		pos = []
		for iatom in self.Atoms:
			pos.append([iatom.x, iatom.y, iatom.z])		

		return np.array(pos)

	def getAtomByResname(self, resname_list):
		"""return atoms by res name"""

		atoms_list = AtomGrp()
		for iatom in self.Atoms:
			if iatom.resname in resname_list:
				atoms_list.append(iatom)

		return atoms_list

	def getAtomByAtomname(self, atomname_list):
		"""return atoms by atom name"""

		atoms_list = AtomGrp()
		for iatom in self.Atoms:
			if iatom.atomname in atomname_list:
				atoms_list.append(iatom)

		return atoms_list

	def selectAtoms(self,bxmin,bxmax,bymin,bymax):
		""" select atoms satisfying conditions"""

		atoms_list = AtomGrp()
		for iatom in self.Atoms:
			if iatom.x>bxmin and iatom.x<bxmax:
				if iatom.y>bymin and iatom.y<bymax:
					atoms_list.append(iatom)

		return atoms_list
