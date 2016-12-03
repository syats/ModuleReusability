'''
Author:          Victor Mireles - Freie Universitaet Berlin
Email:           syats.vm@gmail.com

Latest version:  https://github.com/syats/ModuleReusability

Distributed under GPL V3.0
'''


from scipy.sparse import csc_matrix
import numpy as np
import random as ran
import math
from scipy.stats import binom
from scipy.stats import entropy
import bisect
import measuresForDecomposableMatrix as mfDM
import copy

import utilsDecomposableMatrix as udm

# This class provides a sparse binary matrix with the aditional restriction that columns are always sorted with respect to the order provided by the hashing of them.  This binary matrix can be altered by adding or removing rows. Furthermore, a distance matrix is included such that if it is added to its tranpose, then its j1,j2'th entry encodes the dot product of the j1 and j2'th colums of the binary matrix. This distance matrix is kept up to date under the allowed modifications of the binary matrix.


class decomposableMatrix:
	# --------------------------------------------------------------------------
	# Initialzes a Decomposable Matrix. This matrix has the following attributes:
	#    theMatrix    - a sparse column oriented binary matrix, whose columns are sorted
	#    theHashes    - a hash for every column in the binary matrix
	#    theCoincidM  - a matrix that indicates the dot product between any two columns of the binary matrix
	#    theSizes     - a list with the number of non-zero entries in each column the binary matrix
	#	 totalHash    - the hash of all the hashes
	#
	#  if a cMatrix argument is provided, then
	#    intersectionReuseMatrix - a matrix that indicates the reusability in cMatrix of the intersection between any two columns of the binary matrix. That is, its j1,j2 entry is the number of columns of cMatrix that are all ones in the rows in which columns j1 and j2 of the binary matrix are both one

	def __init__(self,denseMo=[],wVector = [],cMatrix=None):
		reload(udm)
		# Check that it is propperly oriented
		if len(denseMo) == 0:
			return;

		if denseMo.shape[0] >= denseMo.shape[1]:
			denseM = denseMo;
		else:
			denseM = denseMo.T;

		self.theMatrix       = csc_matrix(denseM,dtype=np.int8);
		self.theMatrix.data  = np.ones(self.theMatrix.data.shape,dtype=np.int8);
		self.m = self.theMatrix.shape[0];
		self.k = self.theMatrix.shape[1];
		self.alsoSmaller = False;
		self.C = None;
		self.intersectionReuseMatrix = None;
		if self.C != None:
			if self.C.shape[0] < self.C.shape[1]:
				self.C = self.C.T;

		# Order the columns by hash
		perm = self.orderColumns()
		self.totalHash  = hash(str(self.theHashes));

		# If it's not given, compute the weight vector
		self.weightVector = wVector;
		if len(wVector) == 0:
			self.weightVector = np.ones(self.m);

		# Permute the original matrix and compute coincidense matrix
		denseM = denseM[:,perm];
		self.theCoincidM = self.computeDistances(denseM);
		self.theSizes = self.computeSizes();

		self.sortIndicesWithinColumns();

	def getMatrix(self):
		return self.theMatrix

	def copy(self):
		newObject = decomposableMatrix()
		newObject.theMatrix = self.theMatrix.copy();
		newObject.theHashes = list(self.theHashes);
		newObject.theCoincidM = self.theCoincidM.copy();
		newObject.totalHash = self.totalHash;
		newObject.k = self.k;
		newObject.m = self.m;
		newObject.theSizes = list(self.theSizes);
		newObject.weightVector = self.weightVector.copy();
		newObject.alsoSmaller  = self.alsoSmaller;
		newObject.C = self.C;
		newObject.intersectionReuseMatrix = self.intersectionReuseMatrix;

		return newObject

	# --------------------------------------------------------------------------
	# Determines if this matrix is equal to another.
	# To make things more efficient, by default not the whole matrix is checked, only some random entries and hashes and the total hash. If you want a percise check then pass percise=True as argument
	def equals(self,dm2,percise=False):

		if (percise):
			mat = np.abs(self.theMatrix - dm2.theMatrix).sum() == 0;

			sim1 = (self.theCoincidM.indices.shape[0] - dm2.theCoincidM.indices.shape[0]) == 0;

			sim2 = True;

			return (mat and sim1 and sim2)


		hashChecks    = 22  +     ran.randint(0,10);
		indptrChecks  = 10  +     ran.randint(0,10);
		indicesChecks = 12  +   2*ran.randint(0,10);



		#First super fast checks
		if (not (self.totalHash == dm2.totalHash)):
			#print("TH");
			return False;
		if (not self.theMatrix.shape == dm2.theMatrix.shape):
			#print("MS");
			return False;
		if (not self.theMatrix.indices.shape[0] == dm2.theMatrix.indices.shape[0]):
			#print("MIS");
			return False;
		if (not self.theCoincidM.indices.shape[0] == dm2.theCoincidM.indices.shape[0]):
			#print("CIS");
			return False;

		#We check a subset of the columnhashes
		numToCheck = min([self.k , hashChecks]);
		toCheck    = [ran.randint(0,self.k-1) for i in range(numToCheck)]+[self.k-1]
		for j in toCheck:
			if (not (self.theHashes[j]==dm2.theHashes[j])):
				#print("H"+str(j));
				return False;


		#We check a subset of the indptrs to check
		numIndptrToCheck = min ([ self.theMatrix.indptr.shape[0],indptrChecks] )
		toCheck = [ran.randint(0,self.theMatrix.indptr.shape[0]-1) for i in range(numIndptrToCheck)]+[ self.theMatrix.indptr.shape[0]-1 ];
		if (not (np.abs(self.theMatrix.indptr[toCheck] - dm2.theMatrix.indptr[toCheck]).sum() == 0 ) ):
			#print("MIP");
			return False


		#We check a subset of the indices of the matrix
		numIndToCheck = min([self.theMatrix.indices.shape[0],indicesChecks]);
		toCheck = [ran.randint(0,self.theMatrix.indices.shape[0]-1) for i in range(numIndToCheck)];
		if (not (np.abs(self.theMatrix.indices[toCheck] - dm2.theMatrix.indices[toCheck]).sum() == 0 ) ):
			#print("MI");
			#print(toCheck)
			return False




		return True

	# --------------------------------------------------------------------------
	def computeSizes(self):
		sizes = np.zeros(self.k);
		for j in range(self.k):
			#sizes[j] = (self.theMatrix.getcol(j).indices).shape[0];
			sizes[j] = self.weightVector[self.theMatrix.getcol(j).indices].sum()
		return sizes.tolist()

	# --------------------------------------------------------------------------
	#Given the indices of two columns of the current theMatrix, it returns two set of indices required to create the symetric difference of the two columns (elements that are only in one of them), and the hash of this set of indices
	def symetricDifference(self,j1,j2):
		allIndices = np.zeros(self.m);

		ij1start = self.theMatrix.indptr[j1];
		ij1end = self.theMatrix.indptr[j1+1];
		indj1 = self.theMatrix.indices[ij1start:ij1end];

		ij2start = self.theMatrix.indptr[j2];
		ij2end = self.theMatrix.indptr[j2+1];
		indj2 = self.theMatrix.indices[ij2start:ij2end];

		allIndices[indj1] = 1;
		allIndices[indj2] = allIndices[indj2] + 1;

		symDiff = np.nonzero(allIndices==1)[0];
		symDiff.sort();

		symDiff.flags.writeable = False;
		hashSymDiff = hash(symDiff.data);

		return symDiff,hashSymDiff;

	# --------------------------------------------------------------------------
	# We say that column j1 is contained in column j2 if all the non zero entries of j1 are also non-zero in j2. This functions finds all such pairs j1,j2 and does the following: remove column j2 and add a new column whose non-zero entries are those which are non-zero in j2 but not in j1
	def removeContainedColumns(self):

		colsToAdd = []
		hashesOfColumnsToAdd = [];
		containedMarkers = np.zeros(self.k);
		for j in range(self.k):
			if (containedMarkers[j] != 0):
				continue


			if (self.theSizes[j]==0):
				containedMarkers[j]=1
				#print("-"),
				continue

			sj   = self.theSizes[j];

			ijstart = self.theCoincidM.indptr[j];
			ijend = self.theCoincidM.indptr[j+1];

			#colj = self.theCoincidM.getcol(j);
			indj = self.theCoincidM.indices[ijstart:ijend];
			datj = self.theCoincidM.data[ijstart:ijend];
			ne    = len(datj)


			#   j contains i if datj[i] == sizesOth[i]
			#      |Bj intersection Bi| == |Bi|
			sizesOth  = [self.theSizes[i] for i in list(indj)]
			JcontainsI = [indj[i] for i in range(ne) if datj[i] == sizesOth[i] ]

			#    i cotains j if datj[i] == sj
			#      |Bj intersection Bi| == |Bj|
			#    we want to make sure Bi != Bj
			#    we guarantee this by checking that Bj intersect Bi != |Bi|
			sizeOfJ = np.nonzero(datj == sj)[0];
			IcontainsJ  = [indj[i] for i in sizeOfJ if (datj[i] != sizesOth[i] )];

			# If we find one column containing another, we remove the largest of the two and add a new column that contains the symetric difference of them.
			if (len(JcontainsI)>0):
				containedMarkers[j] = 1;

			containedMarkers[IcontainsJ]  = 1;

			for i in (set(JcontainsI) | set(IcontainsJ)):
				s,h = self.symetricDifference(i,j);
				if not (h in hashesOfColumnsToAdd):
					hashesOfColumnsToAdd.append(h);
					colsToAdd.append(s);


			if (containedMarkers.sum() > 0):
				toRemove = list(np.nonzero(containedMarkers)[0]);
				toRemove.sort();
				toRemove.reverse();

				#if len(toRemove) != len(colsToAdd):
				#print(str(len(toRemove))+" != "+str(len(colsToAdd))+"  ")
				#for s in colsToAdd:
				#	print(s),
				#print(".")

				for j in toRemove:
					#print(str(j))
					self.removeColumn(j);

				for s in colsToAdd:
					#print("*"+str(s.shape)+"*"),
					newCol = np.zeros(self.m);
					if (len(s)) == 0:
						continue
					newCol[s] = 1;
					self.addColumn(newCol)


				#print("."),
				#resu = [self.copy()]+self.removeContainedColumns()
				#print(" lre"+str(len(resu))),
				self.removeContainedColumns()
				return [self]



		return [self];

	# --------------------------------------------------------------------------
	def addColumn(self,newColD):

		sizeNewCol = self.weightVector[np.nonzero(newColD)].sum()
		coincidence = np.zeros(self.k);
		intReuse = np.zeros_like(coincidence)
		for j in range(self.k):
			colJstarts = self.theMatrix.indptr[j];
			colJends   = self.theMatrix.indptr[j+1];
			indicesJ   = self.theMatrix.indices[colJstarts:colJends];

			'''
			intersection = newColD[indicesJ].sum()
			'''
			newColTemp = copy.deepcopy(newColD);
			newColTemp[indicesJ] += 1;

			intersection = self.weightVector[np.nonzero(newColTemp==2)[0]].sum()

			coincidence[j] = intersection;


		#sp = sparse version of newColD
		sp = csc_matrix(newColD,dtype=np.int8)
		spi = sp.getrow(0).indices;
		spi.sort();
		spi.flags.writeable = False;
		newRHash = hash(spi.data);
		place    = bisect.bisect(self.theHashes,newRHash);

		self.theHashes = self.theHashes[:place] + [newRHash] + self.theHashes[place:];
		self.totalHash  = hash(str(self.theHashes));
		self.theSizes = self.theSizes[:place] + [ sizeNewCol ] + self.theSizes[place:];

		self.theMatrix = udm.insertColumnIntoCSCMatrix(self.theMatrix,newColD,place);
		newK = self.theMatrix.shape[1];
		self.k = newK;

		coincidenceNew = np.zeros(newK);
		coincidenceNew[:place] = coincidence[:place]
		coincidenceNew[place+1:] = coincidence[place:]
		# Here coincidenceNew has index considering the new column (because we already inserted it into theMatrix)
		#this advances the indices of all existing distances (if they are greater than place) so that they also consider the new column
		self.theCoincidM = udm.insertEmtpyRowIntoCSCMatrix(self.theCoincidM,place)
		self.theCoincidM = udm.insertColumnIntoCSCMatrix(self.theCoincidM,coincidenceNew,place)

		#self.theCoincidM = self.computeDistances(self.theMatrix.todense());

		self.sortIndicesWithinColumns();

		return place

	# --------------------------------------------------------------------------
	def removeColumn(self,j):

		allCols = range(self.k);
		allCols.remove(j);
		self.k = self.k - 1;

		#  ----- First from the coincidence matrix --------
		#From coincidence matrix, remove the j'th column
		#coincidM = self.theCoincidM[:,allCols];
		coincidM = udm.getColSliceMissingCol(self.theCoincidM,j)
		#print(str(coincidM.shape)+"SIZE COINCID = ")

		#Now we must remove all references to the j'th row. We do this by hand because row-slicing is not supported on csc matrices
		newIndices = coincidM.indices.copy();
		newData    = coincidM.data.copy();
		newIndptr  = coincidM.indptr.copy();

		#Find all indices equal to j, remove them and the corresponding data
		tokeep     = np.nonzero(coincidM.indices!=j)[0]; #positions in the index array
		newIndices = newIndices[ tokeep ];
		newData    = newData[ tokeep ];

		removed    = np.nonzero(coincidM.indices == j)[0];#positions in the index array

		for jp in range(1,len(newIndptr)):
			ltj = len([ x for x in removed if x < coincidM.indptr[jp] ])
			newIndptr[jp] = newIndptr[jp] - ltj;


		#Find all indices greater than j, and substract one from them
		todecrease = np.zeros(len(newIndices))
		todecrease[np.nonzero(newIndices > j)[0] ] = 1
		td = np.nonzero(todecrease)[0];
		#print("td : "+str(td))
		newIndices[td] = newIndices[td] - 1;
		#print("NI :"+str(newIndices));

		newCoincidM = csc_matrix((newData,newIndices,newIndptr),shape=(self.k,self.k));
		self.theCoincidM = newCoincidM;


		#  ----- Now from the matrix and auxiliary stuff --------
		#self.theMatrix = self.theMatrix[:,allCols];
		self.theMatrix = udm.getColSliceMissingCol(self.theMatrix,j)
		self.theHashes = [self.theHashes[jp] for jp in allCols  ]
		self.totalHash  = hash(str(self.theHashes));
		self.theSizes  = [self.theSizes[jp]  for jp in allCols  ]


		self.sortIndicesWithinColumns();

		#print("k: "+str(self.k)+" indices:"+str(self.theCoincidM.indices.max())+ "(" + str(newIndices.max()) +") shape: "+str(self.theCoincidM.shape))

	# --------------------------------------------------------------------------
	def sortIndicesWithinColumns(self):
		if (not self.theMatrix.has_sorted_indices):
			self.theMatrix.sort_indices();

	# --------------------------------------------------------------------------
	def getListOfOverlaps(self):

		coincidenceMatrix = self.theCoincidM;

		R = coincidenceMatrix.data;
		# The list of possible overlaps
		possibleOverlaps = np.unique(R);
		possibleOverlaps.sort();


		return possibleOverlaps


	# --------------------------------------------------------------------------
	def findQthOverlap(self,q):
		coincidenceMatrix = self.theCoincidM;

		possibleOverlaps = self.getListOfOverlaps();
		if (len(possibleOverlaps)==0):
			#print("WTF!!" + str(self.theMatrix.shape)+" 0No: "+str(self.norm0()))
			return 0,0,0
		if (q >= len(possibleOverlaps)):
			q = 0;


		q = q + 1; #Because of numpys backward's indexing.. [-a:] returns the last a elements
		#		print(str(len(possibleOverlaps))+" q "+str(q))
		if len(possibleOverlaps) == 0:
			print("WTF!!")
		qthOverlap = possibleOverlaps[-q];
		if (qthOverlap == 0):
			qthOverlap = possibleOverlaps[-1]

		#print(str(q)+" "+str(len(possibleOverlaps))+" "+str(possibleOverlaps[-1:])+" "+str(qthOverlap));

		entriesWithThisOverlap = np.nonzero(coincidenceMatrix==qthOverlap);
		numEntriesThisOverlap  = len(entriesWithThisOverlap[0]);
		chosenPairIndex = ran.choice(range(numEntriesThisOverlap));
		moreLeft = numEntriesThisOverlap-1;
		best1 = entriesWithThisOverlap[0][chosenPairIndex];
		best2 = entriesWithThisOverlap[1][chosenPairIndex];

		if (best1==best2):
			print("\n!!!!!  J1 = J2  !!!!!!!\n");

		left = numEntriesThisOverlap - 1;
		if (possibleOverlaps.shape[0] == 1):
			left = 0;

		return best1,best2,left




	# --------------------------------------------------------------------------
	# This operation doesn't alter the 0-norm of the matrix. It splits the non-zero entries of one column into two of them, thereby increasing the number of columns in one and keeping the 0-norm. The column that is split is the one with more weight.
	def splitColumn(self,J=None):
		actualSizes = [self.theSizes[j] for j in range(self.k)]
		if J == None:
			j = np.argsort(actualSizes)[-1]
		else:
			j = J;

		ijstart = self.theMatrix.indptr[j];
		ijend   = self.theMatrix.indptr[j+1];
		colJ    = self.theMatrix.indices[ijstart:ijend];

		if len(colJ) == 1:
			return(j)

		spl = int(len(colJ)/2)

		colJ1New    = np.zeros(self.m)
		colJ2New    = np.zeros(self.m)
		colJ1New[colJ[:spl]] = 1;
		colJ2New[colJ[spl:]] = 1;

		self.removeColumn(j)
		self.addColumn(colJ1New)
		self.addColumn(colJ2New)

		return(j)


	# --------------------------------------------------------------------------
	def mergeOverlappingColumns(self,q=0):
		if self.norm0 == min(self.theMatrix.shape):
			return [];

		returnSmaller = self.alsoSmaller;
		result = [];

		count = 0;
		objK = self.k + 1;
		while (self.k < objK):
			count+=1;

			if (returnSmaller) and (count>=1):
				result.append(tuple([self.copy(),0]))

			'''
			if (self.k == self.m) and (self.norm() == self.m):
				print("~~~~~")
				print(str(self.norm()))
				print(str((returnSmaller) and (count>1)))
				result.append(tuple([self.copy(),0]))
				print("~~~~~")
			'''

			if min(self.theMatrix.shape)==self.norm0():
				return result
			j1,j2,left = self.findQthOverlap(q);
			stat = self.overlapToNewColumn(j1,j2)
			lsc = self.removeContainedColumns();
			result.append(tuple([self.copy(),left]))


		return result;

	# --------------------------------------------------------------------------
	# --------------------------------------------------------------------------
	def overlapToNewColumn(self,j1,j2):

		if j2 > j1:
			j1,j2 = j2,j1;
		if j1 == j2:
			#print("\n\n\n j1 == j2 \n\n\n")
			#print(str(self.theMatrix.todense().sum(axis=1)));
			#print(str(self.theMatrix.todense().sum(axis=0)));
			return 0
		#print(str(j1)+" "+str(j2)+" "+str(self.theCoincidM[j1,j2])+" "+str(self.theCoincidM[j2,j1]))
		#print("K = "+str(self.k)+"   "+str(j1)+","+str(j2))
		ij1start = self.theMatrix.indptr[j1];
		ij1end = self.theMatrix.indptr[j1+1];
		colJ1 = self.theMatrix.indices[ij1start:ij1end];
		ij2start = self.theMatrix.indptr[j2];
		ij2end = self.theMatrix.indptr[j2+1];
		colJ2 = self.theMatrix.indices[ij2start:ij2end];

		#print("colj1 : "+str(colJ1))
		#print("colj2 : "+str(colJ2))

		counts1        = np.zeros(self.m);
		counts1[colJ1] = 1;  #Has 0 everywhere except for those in colj1
		counts2        = np.zeros(self.m);
		counts2[colJ2] = 1;  #Has 0 everywhere except for those in colj2

		sizeBoth = len(colJ1)+len(colJ2);

		#counts has 1 in those belonging to only one of colj1 and colj2, and 2 for those belonging to both
		counts = counts1 + counts2;

		intersectionI = np.nonzero(counts==2)[0];
		#now counts only has 1 on those belonging to either colj1 or colj2
		del counts
		counts1[intersectionI] = 0;
		counts2[intersectionI] = 0;

		#counts+counts1 has 2 on those belonging only to colj1, and 1 on those belonging only to colj2

		colJ1NewI = np.nonzero(counts1)[0];
		colJ2NewI = np.nonzero(counts2)[0];

		del counts1
		del counts2

		#print("I: "+str(len(intersectionI))),
		normInit = self.theMatrix.indices.shape[0]
		#print("\tini "+str(self.theMatrix.indices.shape[0])),
		# remove columns j1 and j2 from theMatrix
		self.removeColumn(j1);
		#print("\tR1 "+str(self.theMatrix.indices.shape[0])),
		self.removeColumn(j2);
		#print("\tR2 "+str(self.theMatrix.indices.shape[0])),

		# add new columns
		colJ1New = np.zeros(self.m);
		colJ1New[colJ1NewI] = 1;
		colJ2New = np.zeros(self.m);
		colJ2New[colJ2NewI] = 1;
		colIntersection = np.zeros(self.m);
		colIntersection[intersectionI] = 1;

		#print("colj1New : "+str(colJ1NewI))
		#print("colj2New : "+str(colJ2NewI))
		#print("colInNew : "+str(intersectionI))

		if colJ1New.sum() > 0:
			r = self.addColumn(colJ1New);
			#print("\tA1 "+str(self.theMatrix.indices.shape[0])),
			#print("\t add1: "+str(self.k)+" "+str(r)+" s1:"+str(colJ1New.sum()))
		if colJ2New.sum() > 0:
			r = self.addColumn(colJ2New);
			#print("\tA2 "+str(self.theMatrix.indices.shape[0])),
			#print("\t add2: "+str(self.k)+" "+str(r)+" s2:"+str(colJ2New.sum()))
		if colIntersection.sum() > 0:
			r = self.addColumn(colIntersection);
			#print("\tAI "+str(self.theMatrix.indices.shape[0])),
			#print("\t addI: "+str(self.k)+" "+str(r)+" sI:"+str(colIntersection.sum()))

		normFin = self.theMatrix.indices.shape[0]

		if (normInit - normFin) != len(intersectionI):
			print("!!!!!!!!!!!!!!!!");
			print(str(j1)+" "+str(j2)+" "+str(len(intersectionI)));
			print(":"+str(len(intersectionI)) + "\n\n\n")

		lsc = self.removeContainedColumns();


		#result.append(tuple([self.copy(),left]))

		#if (any([res[0].k > objK for res in result])):
		#	print("LARGE RESULT!!!");

		return 1

	# --------------------------------------------------------------------------
	# Given Bk a dense matrix, computes the column wise coincidence between every pair of its columns, weighted by self.weightVector. The result is a dense matrix of k x k, where k is the number of columns of Bk.
	def computeDistances(self,BkO):
		Bk = np.array(BkO.copy(),dtype=np.double)
		wV = self.weightVector;
		k  = Bk.shape[1];


		weightMatrix = udm.nodeWeightMatrix(wV,k);
		#print("BK.T: "+str(Bk.T.shape)+"  weightMatrix: "+str(weightMatrix.shape))
		coincidenceMatrix = np.dot(weightMatrix*(Bk.T) , Bk );
		coincidenceMatrix = coincidenceMatrix - np.diag(np.diag(coincidenceMatrix));




		return csc_matrix(np.triu(coincidenceMatrix),dtype=np.double,shape=(k,k))

	# --------------------------------------------------------------------------
	#For every column, a hash is computed. Then, the columns are ordered by this hash. The sorted hashes are stored. The permutation of the columns that yields the sorted version is returned
	def orderColumns(self):
		k = self.k;
		cols = [self.theMatrix.getcol(j).indices for j in range(k) ];
		for co in cols:
			co.sort();
			co.flags.writeable = False;
		hashes = [hash(co.data) for co in cols];
		hashO  = udm.argSort(hashes);

		#self.theMatrix.indices = self.theMatrix.indices[hashO];
		newIndices = np.zeros(self.theMatrix.indices.shape,dtype=np.int32);
		newIndptr  = np.zeros(self.theMatrix.indptr.shape,dtype=np.int32);
		newData    = np.zeros(self.theMatrix.data.shape,dtype=np.int8);

		startNextColumnIndex = 0;
		newIndptr[0] = 0;
		self.theHashes = range(len(hashes));
		for J in range(k):   #J is the column index in the new matrix
			j = hashO[J];    #j is the column index in the old (unsorted) matrix
			numElems = self.theMatrix.indptr[j+1]-self.theMatrix.indptr[j];
			newIndptr[J+1] = startNextColumnIndex+numElems;
			so = self.theMatrix.indptr[j];
			eo = self.theMatrix.indptr[j+1];
			newIndices[newIndptr[J]:newIndptr[J+1]] = self.theMatrix.indices[so:eo];
			newData[newIndptr[J]:newIndptr[J+1]] = self.theMatrix.data[so:eo];
			startNextColumnIndex = startNextColumnIndex+numElems;
			self.theHashes[J] = hashes[j];

		self.totalHash  = hash(str(self.theHashes));
		self.theMatrix.data    = newData;
		self.theMatrix.indices = newIndices;
		self.theMatrix.indptr  = newIndptr;

		return hashO

	def norm0(self):
		return self.theMatrix.data.shape[0]

	def norm(self):
		wI = self.weightVector[self.theMatrix.indices];
		return wI.sum();

	def saveToFile(self,fileName):
		np.savez(fileName, Mdata = self.theMatrix.data , Mindices = self.theMatrix.indices, Mindptr = self.theMatrix.indptr, Mshape = self.theMatrix.shape , hashes = np.array(self.theHashes) , Cdata = self.theCoincidM.data , Cindices = self.theCoincidM.indices , Cindptr = self.theCoincidM.indptr , Cshape = self.theCoincidM.shape , totHash = np.array(self.totalHash) , sizes = np.array(self.theSizes) , wv = self.weightVector  )

	def loadFromFile(self,fileName):
		loader = np.load(fileName)

		newObject = decomposableMatrix()
		newObject.theMatrix = csc_matrix((loader['Mdata'], loader['Mindices'], loader['Mindptr']), shape = loader['Mshape']);
		newObject.theHashes = loader['hashes'].tolist();
		newObject.theCoincidM = csc_matrix((loader['Cdata'], loader['Cindices'], loader['Cindptr']), shape = loader['Cshape']);
		newObject.totalHash = loader['totHash'].tolist();
		newObject.k = loader['Mshape'][1];
		newObject.m = loader['Mshape'][0];
		newObject.theSizes = loader['sizes'].tolist();
		newObject.weightVector = loader['wv'];
		newObject.alsoSmaller = False;

		return newObject

	def realSizes(self,j):
		ijstart = self.theMatrix.indptr[j];
		ijend = self.theMatrix.indptr[j+1];
		indj = self.theMatrix.indices[ijstart:ijend];
		sizeB = self.weightVector[indj].sum();

		return sizeB;

	#For a given matrix C (full, m x n). S is a matrix such that D=BS has the same sparcity pattern as C, where B is this object (self). This is equivalent to saying C = sign(D)
	def createSandDforC(self,Co):
		if Co.shape[0]<Co.shape[1]:
			C = Co.T;
		else:
			C = Co;

		n = C.shape[1];

		D = np.zeros([self.m,n])
		S = np.zeros([self.k,n])
		for j in range(self.k):
			ijstart = self.theMatrix.indptr[j];
			ijend = self.theMatrix.indptr[j+1];
			indj = self.theMatrix.indices[ijstart:ijend];

			lindj = len(indj);
			if lindj == 0:
				continue

			#for every tissue, which of the genes of j it cotains
			Cj = C[indj,:];
			#print("Cj shape "+str(Cj.shape)),
			sCj = Cj.sum(axis=0);
			sCj = np.squeeze(np.asarray(sCj))
			#tissues that contain all of the genes in bj
			Sj = np.nonzero(sCj==lindj)[0];
			#print("\tsCj: \t "+str(sCj)+" fit:"+str(Sj))
			S[j,Sj] = 1;
			for i in Sj:
				#print(str(indj)+" \t\t "+str(i))
				D[indj,i] += 1;

		return S,D

	def findSuperflousEntries(self,C,SS=None,DD=None):
		if (SS==None) or (DD==None):
			S,Do = self.createSandDforC(C);
		else:
			S = SS;
			Do = DD;
		D = Do.copy()
		R = np.zeros([self.m,self.k])
		for j in range(self.k):
			ijstart = self.theMatrix.indptr[j];
			ijend = self.theMatrix.indptr[j+1];
			indj = self.theMatrix.indices[ijstart:ijend];
			for x in indj:
				tempR = S[j,:]*D[x,:];
				if len(tempR[np.nonzero(tempR)]) > 0:
					R[x,j] = (tempR[np.nonzero(tempR)]).min()
					#if R[x,j] > 1:
					#	D[x,:] = D[x,:] - S[j,:];

		return np.nonzero(R>1),R,S,Do


	def removeSuperflousIncreasingReuse(self,Co):
		if Co.shape[0]<Co.shape[1]:
			C = Co.T;
		else:
			C = Co;
		totRemoved = 0;
		sup,R,S,D = self.findSuperflousEntries(C);
		D2 = np.zeros_like(D);
		D2[np.nonzero(D)]=1;
		#print("lensup "+str(len(sup[0]))+" Eo:"+str(np.abs(D2-C).sum()))
		if len(sup[0])==0:
			print("*"),
			return None;

		BDensM = copy.deepcopy(self.theMatrix.todense());
		Bdense = np.array(BDensM,dtype=np.float64)
		numChanges = 0;
		while(len(sup[0]) > 0):
			numChanges += 1;

			maxDelta = -1;
			maxPair  = None;
			for j in range(self.k):
				supForJ = np.nonzero(Bdense[:,j]*R[:,j]>1)[0]
				if len(supForJ)==0:
					continue
				delta = self.deltaReuse(supForJ,j,C)
				maxDeltaThisJ = max(delta);
				if maxDeltaThisJ > maxDelta:
					maxDelta = maxDeltaThisJ;
					maxPair = (supForJ[delta.index(maxDelta)],j);


			if maxPair == None:
				break
			Bdense[maxPair[0],maxPair[1]] = 0;
			D[maxPair[0],np.nonzero(S[maxPair[1],:])[0]] -= 1;
			totRemoved += 1;
			sup,R,S,D = self.findSuperflousEntries(C,S,D);

		#print("totRemoved ="+str(totRemoved))
		#print("> "+str(numChanges)+"chgs "+str(totRemoved))
		if totRemoved == 0:
			return None

		newBk = decomposableMatrix(Bdense,wVector=np.array(self.weightVector.copy()),cMatrix=self.C);

		return newBk

	def deltaReuse(self,X,j,C):
		ijstart = self.theMatrix.indptr[j];
		ijend   = self.theMatrix.indptr[j+1];
		indj    = self.theMatrix.indices[ijstart:ijend];
		rOrig   = self.singleModuleReusability(C,indj);

		delta = [0 for i in range(len(X))];
		for xi in range(len(X)):
			x = X[xi]
			propJ = indj.copy().tolist();
			propJ.remove(x);
			rx = self.singleModuleReusability(C,np.array(propJ));
			delta[xi] = rx-rOrig;

		return delta;



	def removeSuperflous(self,Co):
		if Co.shape[0]<Co.shape[1]:
			C = Co.T;
		else:
			C = Co;

		sup,R,S,D = self.findSuperflousEntries(C);
		D2 = np.zeros_like(D);
		D2[np.nonzero(D)]=1;
		#print("lensup "+str(len(sup[0]))+" Eo:"+str(np.abs(D2-C).sum()))
		if len(sup[0])==0:
			return self,False,S,D;

		#We convert self's matrix to dense and remove superflous
		BDensM = copy.deepcopy(self.theMatrix.todense());
		Bdense = np.array(BDensM,dtype=np.float64)

		#Maybe a module is now empty? let's remove empty modules
		sB = Bdense.sum(axis=0);
		sB = np.squeeze(np.asarray(sB))
		Bdense = Bdense[:,np.nonzero(sB>0)[0]]



		Bdense[sup] = 0;

		#print(str(self.theMatrix.shape)+":  "+str(self.theMatrix.sum())+"\t "+str(Bdense.shape)+":"+str(Bdense.sum())+" \t "+str(self.weightVector.shape)),

		newBk = self.copy();
		newBk = decomposableMatrix(Bdense,wVector=np.array(self.weightVector.copy()),cMatrix=self.C);
		#print("--> "+str(newBk.theMatrix.shape)+" "+str(newBk.theMatrix.sum())),
		supNew,RNew,SNew,DNew = newBk.findSuperflousEntries(C);
		'''
		DNew2 = np.zeros_like(DNew);
		DNew2[np.nonzero(DNew)]=1;
		#print("\t D<C: "+str(len(np.nonzero(DNew2 < C)[0])  )+"  D>C"+str(len(np.nonzero(DNew2 > C)[0])  ))



		newModuleUsage = SNew.sum(axis=1);
		newModuleUsage = np.squeeze(np.asarray(newModuleUsage))
		unUsedModules  = np.nonzero(newModuleUsage==0)[0].tolist();
		if len(unUsedModules)>0:
			print("UNUSED  "+str(len(unUsedModules)))
			unUsedModules.sort();
			unUsedModules.reverse()
			for jj in unUsedModules:
				newBk.removeColumn(jj);


		newBk.removeContainedColumns()
		countOverlap = 0;
		while (newBk.k < self.k) and (countOverlap < 30):
			newBk.mergeOverlappingColumns()
			countOverlap += 1;
		supNew,RNew,SNew,DNew = newBk.findSuperflousEntries(C);
		'''
		return newBk,True,SNew,DNew


	def singleModuleReusability(self,C,indj):

		ri = 0;
		lindj = len(indj);

		#for every tissue, which of the genes of j it cotains
		Cj = C[indj,:];
		sCj = Cj.sum(axis=0);
		sCj = np.squeeze(np.asarray(sCj))
		#tissues that contain at least one of the genes in bj
		tj = len(np.nonzero(sCj>0)[0]);
		ri = len(np.nonzero(sCj==lindj)[0]);

		return ri

	def perModuleReusability(self,C,alsoWeights=False):
		return mfDM.perModuleReusability(self,C,alsoWeights)

	def reusability(self,C,pmr=[],returnList = False,normalized=False):
		if (pmr == []):
			perModReuse,perModIntersect = self.perModuleReusability(C)
			if normalized:
				perModReuse = (perModReuse/perModIntersect);
		else:
			perModReuse = pmr;
		if (not returnList):
			return perModReuse.sum()
		else:
			rs = np.argsort(perModReuse);
			perModReuse  = perModReuse[rs];


			res   = [perModReuse.sum()]
			avSizes = []
			numQuantiles = 10;
			size = int(len(perModReuse)/numQuantiles)
			for i in range(numQuantiles):
				lim = i*size;
				lis = perModReuse[lim:]
				sis = [self.realSizes(j) for j in rs[lim:]]
				if (len(lis)>0):
					res.append(lis.sum()/float(len(lis)))
					avSizes.append(np.mean(sis))
			return res,avSizes;

	#The likelihood
	def reusabilityLikelihood(self,C, displayEach=False,pmr=[],pmi=[]):
		if True:
			return 0;
		reus = 0;
		totLik = 0;
		D = C.sum(axis=0)*float(1.0/C.shape[0]);
		d = np.mean(D);

		N = np.min(C.shape);

		if (pmr==[]) or (pmi==[]):
			perModReuse,perModIntersect = self.perModuleReusability(C)
		else:
			perModReuse = pmr;
			perModIntersect = pmi;

		for j in range(self.k):
			ri = perModReuse[j];
			sizeB = self.realSizes(j);
			bino = binom(N,math.pow(d,sizeB));
			likJ = bino.pmf(ri);
			if (likJ == 0):
				likJ = -np.inf;
			else:
				likJ = np.log(likJ)
			if(displayEach):
				#lik = math.log(C.shape[1]) + math.log(math.pow(D[i],sizeB));
				print(str(ri)+"("+str(likJ)+") "),

			totLik += likJ;
			tj = perModIntersect[j]
			reus += ri/float(tj);
		#print(str(totLik)+" "),
		return totLik;


	def KLToRandom(self,C,pmr=[],pmi=[]):
		D = C.sum(axis=0)*float(1.0/C.shape[0]);
		d = np.mean(D);
		N = np.min(C.shape);

		if True:
			return 0

		#We compute the sizes of all modules
		M = self.weightVector.sum();
		numModulesOfEachSize = np.zeros(M);
		for j in range(self.k):
			sizeK = self.realSizes(j)
			numModulesOfEachSize[sizeK]+=1;
		#Given the sizes of modules and the density of C, we can compute the estimated distribution of reusabilities as follows:

		# E_C[f] is the expected number of modules of reuse f, under the assumptions that modules sizes obey the distribution encoded in numModulesOfEachSize

		def P_C(sizeB,reuse):
			bino = binom(N,math.pow(d,sizeB));
			prob = bino.pmf(reuse);
			return prob;

		E_C = np.zeros(N+1)
		for f in range(N+1):  #reuse
			Ef = 0.0;  #Expected number of reuse f
			for l in range(self.m):  #size
				Kl = numModulesOfEachSize[l];
				if (Kl > 0):
					Ef += Kl*P_C(l,f);
			E_C[f] = Ef;

		#So far E_C[f] stores the expected number of modules with reusability f;

		if (pmr==[]) or (pmi==[]):
			perModReuse,perModIntersect = self.perModuleReusability(C)
		else:
			perModReuse = pmr;
			perModIntersect = pmi;

		observedDistributionOfReuse = np.zeros(N+1);
		for j in range(self.k):
			observedDistributionOfReuse[perModReuse[j]]+=1;

		#We now compare the observed distribution with the expected one via de KL Divergence.
		#print("observed: "+str(observedDistributionOfReuse))
		#print("expected" +str(E_C))
		KL = entropy(pk = observedDistributionOfReuse,qk = E_C);

		return KL

	def zScoreToRandom(self,C,pmr=[],pmi=[]):
		if True:
			return 0,0;
		reus = 0;
		D = C.sum(axis=0)*float(1.0/C.shape[0]);
		d = np.mean(D);
		toZScore = 0;
		toZScoreNorm = 0;
		N = np.min(C.shape);

		if (pmr==[]) or (pmi==[]):
			perModReuse,perModIntersect = self.perModuleReusability(C)
		else:
			perModReuse = pmr;
			perModIntersect = pmi;

		for j in range(self.k):
			ri = perModReuse[j];
			sizeB = self.realSizes(j);

			#These are the parameters for the bionomial dist of the reusability of modules of size sizeB
			n = N;
			p = math.pow(d,sizeB);
			#Then the mean and std deviation are:
			mu    = n*p;
			sigma = math.sqrt(n*p*(1-p));
			zSj   = (ri - mu) / sigma;

			maxZS = (n-mu) / sigma;
			minZS = -mu / sigma;
			zSjN  = (zSj - minZS) / (maxZS - minZS);

			toZScore 	 += zSj;
			toZScoreNorm += zSjN;
			tj = perModIntersect[j]
			reus += ri/float(tj);
		#print(str(totLik)+" "),
		return toZScore,toZScoreNorm;
