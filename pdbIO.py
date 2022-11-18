import sys
from atom import pdbAtom,AtomGrp

class pdbIO(AtomGrp):

	def __init__(self):
		super().__init__()

	def formPDBGrp(self, atomobj):
		pass

	def readPDB(self, fname):
		""" read pdb file 12 items"""

		# open file
		with open(fname, 'r') as FI:

			content = FI.read().splitlines()

			# total lines
			Nlines = len(content)

			# loop
			for i in range(Nlines):
				iline = content[i]

				if iline.split()[0] in ['HETATM', 'ATOM']:
					# split the string
					i_list = iline.split()
					
					rectype = i_list[0]
					atomid =  i_list[1]
					atomnm =  i_list[2]
					resname = i_list[3]
					resid   = i_list[4]
					x = 	  i_list[5]
					y = 	  i_list[6]
					z = 	  i_list[7]
					occ = 	  i_list[8]
					tempfac = i_list[9]
					element = i_list[10]
					
					# to float
					(x,y,z,occ,tempfac) = map(float, (x,y,z,occ,tempfac))
					(atomid,resid) = map(int, (atomid,resid))

					# read the record
					atom = pdbAtom(rectype,atomid,atomnm,resname,resid,x,y,z,occ,tempfac,element)
					self.atoms_list.append(atom)
					
		self.Natoms = len(self.atoms_list)					

if __name__ == '__main__':

	if len(sys.argv) != 2:
		print("Right call: {} pdbFile".format(sys.argv[0]))
		sys.exit(0)

	fname=sys.argv[1]
	pdbObj = pdbIO()
	pdbObj.readPDB(fname)
	print(pdbObj.Natoms)

	print(pdbObj.getAtomTypes())
	print(pdbObj.getResTypes())
	print(pdbObj.getPos().shape)

	PEO = pdbObj.getAtomByResname(['PE1','PE2','PE3'])
	print(PEO.Natoms)

	gld = pdbObj.getAtomByAtomname(['GLD'])
	print(gld.Natoms)