'''
Author:          Victor Mireles - Freie Universitaet Berlin
Email:           syats.vm@gmail.com

Latest version:  https://github.com/syats/ModuleReusability

Distributed under GPL V3.0
'''


import numpy as np

def uncleanMatrix(B,translationOfRows):
	m = len(translationOfRows);

	doTranspose = False;
	k = B.shape[1];
	r = B.shape[0];
	wB = B;
	if (k > r):
		doTranspoe = True;
		k = B.shape[0];
		r = B.shape[1];
		wB = B.T;

	largeB = np.zeros([m,k]);
	for x in range(m):
		row = translationOfRows[x];
		largeB[x,:] = wB[row,:]

	if doTranspose:
		return largeB.T
	else:
		return largeB

#-----------------------------------------------------
#
# Clean input matrix
#
# For a given m times n binary matrix C (n<m), a new matrix is generated that satisfies the followint properties:
#  1) No two rows are the same
#  2) There is no colmn whos set of non-zero entries is a subset of the set of non-zero entries of another column
# 3) No row is all zeros
#
#  Because 1) reduces the number of rows, a mapping from the original to the new rows is also provided

def cleanInputMatrix(oC,alsoReturnTranslationToOriginal = False):
	# ------------- Make sure it is correctly oriented numColumns < numRows
	#                    That is C is m by n  n < m

	origC = oC
	if (origC.shape[1] > origC.shape[0]):
		origC = np.transpose(oC)


	m = origC.shape[0]
	n = origC.shape[1]
	C = np.array(origC.copy(),dtype=np.double)

	#print("INPUT: "+str(m)+" "+str(n))

	# ------------- Remove columns that are contained in another
	# For every column in the original matrix, save the cols of the new one that make it up

	a = np.zeros(n)
	for i1 in range(n):
		for i2 in range(n):
			if (i1 == i2):
				continue
			#print(str(np.dot(C[:,i1],C[:,i2]))+" "),
			if np.dot(C[:,i1],C[:,i2]) == np.dot(C[:,i1],C[:,i1]):
				#print(str(i1)+" contained in "+str(i2))
				if np.dot(C[:,i1],C[:,i2]) != np.dot(C[:,i2],C[:,i2]):
					a[i1] += 1
				else:
					if (i2 < i1):
						a[i1] += 1

	emptyTissues = np.nonzero(C.sum(axis=0)==0)[0];
	a[emptyTissues] = 1;
	goodTissues = np.nonzero(a==0)[0]
	C = C[:,goodTissues]

	m = C.shape[0]
	n = C.shape[1]
	#print("RANK REDUCTION: "+str(m)+" "+str(n))

	# ---- ------------- Remove empty Genes
	geneUsage = C.sum(axis=1)
	emptyGenes = np.nonzero(geneUsage==0)[0]
	a = np.ones(C.shape[0]);
	a[emptyGenes] = 0;
	goodGenes = np.nonzero(a)[0];

	C = C[goodGenes,:];

	m = C.shape[0]
	n = C.shape[1]

	#print("EMPTY GENE REMOVAL: "+str(m)+" "+str(n))

	# x is a gene.. d(x) is the numbering of this gene before the empty genes were removed
	de = np.zeros(m)
	for x in range(m):
		de[x] = x + len([y for y in emptyGenes if y <= x])
	origC = C.copy()


	# ------------- Remove repeated Rows
	#  For every row in the original matrix, save the row number in the new one

	diff = set()
	for x in range(m):
		diff.add(tuple(origC[x,:]))

	r = len(diff);
	diff = list(diff)

	C = np.zeros([r,n])
	for x in range(r):
		C[x,:] = diff[x];

	translationOfRows = np.zeros(max(origC.shape));
	#print("GROUPING OF GENES: "+str(m)+" "+str(n))
	for x in range(max(origC.shape)):
		if x in emptyGenes:
			translationOfRows[x] = -1;
			continue
		translationOfRows[x] = diff.index(tuple(origC[x,:]))


	#Let mx \in {1....r} be one of the metagenes (rows) described in the output matrix. Then detr(mx) \subset {1....m} is the set of genes (rows) of the input matrix that map to mx
	detr = dict()
	for mx in range(r):
		detr[mx]  = [de[x] for x in np.nonzero(translationOfRows==mx)[0]]


	#Up to here we have:
	# origC                the original matrix read
	# C                    a new matrix such that no two rows are the same
	# translationOfRows    whose x'th entry is the index that the x'th row of origC occupies in C


	if alsoReturnTranslationToOriginal:
		return emptyGenes,detr,translationOfRows,C



	return translationOfRows,C
