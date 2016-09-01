'''
Author:          Victor Mireles - Freie Universitaet Berlin
Email:           syats.vm@gmail.com

Latest version:  https://github.com/syats/ModuleReusability

Distributed under GPL V3.0
'''


import numpy as np
from scipy.sparse import csc_matrix

# --------------------------------------------------------------------------
def nodeWeightMatrix(nodeWeights,k):
	return np.array([nodeWeights,]*k,dtype='uint8');


# --------------------------------------------------------------------------
def getColSliceMissingCol(mat,col):

	colStart      = mat.indptr[col];
	nexColStart   = mat.indptr[col+1];
	#print("->>>>**"+str(col)+": "+str(colStart)+" "+str(nexColStart));
	colSize = nexColStart - colStart;


	k = mat.shape[1];
	m = mat.shape[0];


	indptr  = mat.indptr.tolist();
	data    = mat.data.tolist();
	indices = mat.indices.tolist();

	newIndptr  = indptr[:col]+[x - colSize for x in indptr[col+1:]];


	newData    = data[:colStart]+data[nexColStart:];
	newIndices = indices[:colStart]+indices[nexColStart:];

	newdata = np.array(newData,dtype=mat.dtype);
	newindices = np.array(newIndices,dtype=np.int32);
	newindptr = np.array(newIndptr,dtype=np.int32);


	newMat = csc_matrix((newdata,newindices,newindptr),shape=(m,k-1));
	#print("SUMMARY :"+str([len(indptr),len(data),len(indices)])+" -> "+str([len(newindptr),len(newdata),len(newindices)])+" ColSize = "+str(colSize)+" ColToRemove"+str(col)+" "+str(mat.shape)+" "+str(newMat.shape))

	return newMat

# --------------------------------------------------------------------------
def argSort(seq):	#http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
	#by unutbu
	return sorted(range(len(seq)), key=seq.__getitem__)


# --------------------------------------------------------------------------
#  column is a numpy array
def insertColumnIntoCSCMatrix(mat,column,position):

	k = mat.shape[1];
	m = mat.shape[0];

	indptr  = mat.indptr.tolist();
	data    = mat.data.tolist();
	indices = mat.indices.tolist();

	colIndices  = np.nonzero(column)[0].tolist();
	colData     = column[colIndices].tolist();
	numElems    = len(colIndices);

	newColBegin = indptr[position];
	newColEnd   = newColBegin+numElems;

	newData    = data[:newColBegin] + colData + data[newColBegin:]
	newIndices = indices[:newColBegin] + colIndices + indices[newColBegin:]
	newIndptr  = indptr[:position+1] + [newColEnd] +                                      [x + numElems for x in indptr[position+1:]]

	newdata = np.array(newData,dtype=mat.dtype);
	newindices = np.array(newIndices,dtype=np.int32);
	newindptr = np.array(newIndptr,dtype=np.int32);

	newMat = csc_matrix((newdata,newindices,newindptr),shape=(m,k+1));

	#print("}"+str(len(colIndices))+","),
	return newMat;

# --------------------------------------------------------------------------
def insertEmtpyRowIntoCSCMatrix(mat,position):
	indices = mat.indices.tolist();
	newIndices = indices;
	over = [i for i in range(len(indices)) if indices[i] >= position]
	for i in over:
		newIndices[i] += 1;

	newindices = np.array(newIndices,dtype=np.int32);

	k = mat.shape[1];
	m = mat.shape[0];
	newMat = csc_matrix((mat.data,newindices,mat.indptr),shape=(m+1,k));

	return newMat;


# --------------------------------------------------------------------------
def removeRowFromCSCMatrix(mat,position):
	indices = mat.indices.tolist();
	newIndices = indices;

	over = [i for i in range(len(indices)) if indices[i] > position]
	for i in over:
		newIndices[i] -= 1;

	newindices = np.array(newIndices,dtype=np.int32);

	k = mat.shape[1];
	m = mat.shape[0];
	newMat = csc_matrix((mat.data,newindices,mat.indptr),shape=(m-1,k));

	return newMat;
