'''
Author:          Victor Mireles - Freie Universitaet Berlin
Email:           syats.vm@gmail.com

Latest version:  https://github.com/syats/ModuleReusability

Distributed under GPL V3.0
'''


import aux.auxGeneratingModules as agm
#import random as ran
import numpy as np
import multiprocessing

def singlePropagationDM(tu):
	mat = tu[0];
	Q   = tu[1];
	m2 = mat.copy();
	R = m2.mergeOverlappingColumns(Q);
	if (type(R) is list):
	#R is a list of matrix,left  pairs
		return [tuple([m,l,Q,mat])  for m,l in R]
	return []



def FFtupleL(x):
	B = x[0];
	Q = x[1];
	weightMatrix = x[2]
	return agm.oneMoreBlockRemovingQthOverlap(B,weightMatrix,Q)

def propagateMultiCore(Qs,numCores):
	po = multiprocessing.Pool(numCores)
	prop2 = po.map(FFtupleL,Qs)
	po.close()

	return prop2

def lexSortMultiCore(listOfMatrices,numCores):
	po = multiprocessing.Pool(numCores)
	prop2 = po.map(lexS,listOfMatrices)
	po.close()
	return prop2

def lexS(b):
	a = b.T;
	return (a[:,np.lexsort(a)]).T


# input   = tuple([mat,q])
# output  = tuple([newmat,areleft,Q,mat])
#  unpacking and packing suitable for map operation


class auxForMultiCore:
	def propagateMultiCoreDM(self,Qs):
		#print("LEN:  "+str(len(Qs)))
		#print(Qs)
		#print("\n\n");
		prop2 = self.mpPool.map(singlePropagationDM,Qs)
		return prop2

	def __init__(self,numCores):
		self.nCores = numCores;
		self.mpPool = multiprocessing.Pool(numCores,maxtasksperchild=15);

	def kill(self):
		self.mpPool.close();
		del self.mpPool;
